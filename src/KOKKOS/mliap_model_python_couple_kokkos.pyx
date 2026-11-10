# cython: language_level=3
# distutils: language = c++

cimport cython

from libc.stdint cimport uintptr_t

import pickle

# For converting C arrays to numpy arrays
import numpy

import warnings
warnings.filterwarnings("ignore", message=".*Overriding a previously registered kernel.*")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")
import torch

mode = "cpu"
try:
    import cupy
    mode = "nvidia"
except ImportError:
    pass

try:
    import dpctl
    import dpctl.memory as dpmem
    import dpnp
    from cpython.pycapsule cimport PyCapsule_New
    mode = "intel"
except ImportError:
    pass

# For converting void * to integer for tracking object identity
from libc.stdint cimport uintptr_t

from libcpp.string cimport string


cdef extern from "mliap_data_kokkos.h" namespace "LAMMPS_NS":
    cdef cppclass MLIAPDataKokkosDevice:
        # Array shapes
        int nlistatoms
        int nlocal
        int ndescriptors

        # Input data
        int * ielems                # types for all atoms in list
        double * descriptors       # descriptors for all atoms in list

        # Output data to write to
        double * betas             # betas for all atoms in list
        double * charge_betas
        double * charges
        double * eatoms             # energy for all atoms in list
        double * energy

        # Method to update charges for electrostatic solver
        void update_charges()

        # Reference to sycl queue for intel
        void * sycl_queue

cdef extern from "mliap_model_python_kokkos.h" namespace "LAMMPS_NS":
    cdef cppclass MLIAPModelPythonKokkosDevice:
        void connect_param_counts()


class MLIAPPYKokkosModelNotLinked(Exception): pass


LOADED_MODELS = {}

cdef object c_id(MLIAPModelPythonKokkosDevice * c_model):
    """
    Use python-style id of object to keep track of identity.
    Note, this is probably not a perfect general strategy but it should work fine with LAMMPS pair styles.
    """
    return int(<uintptr_t> c_model)

cdef object retrieve(MLIAPModelPythonKokkosDevice * c_model) with gil:
    try:
        model = LOADED_MODELS[c_id(c_model)]
    except KeyError as ke:
        raise KeyError("Model has not been loaded.") from ke
    if model is None:
        raise MLIAPPYKokkosModelNotLinked("Model not linked, connect the model from the python side.")
    return model

cdef public int MLIAPPYKokkos_load_model(MLIAPModelPythonKokkosDevice * c_model, char* fname) with gil:
    str_fname = fname.decode('utf-8') # Python 3 only; not Python 2 not supported.
    if str_fname == "LATER":
        model = None
        returnval = 0
    else:
        if str_fname.endswith(".pt") or str_fname.endswith(".pth"):
            import torch

            device = torch.device("cpu")
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.xpu.is_available():
                device = torch.device("xpu")
            elif torch.mps.is_available():
                device = torch.device("mps")

            model = torch.load(str_fname, map_location=device, weights_only=False)

            model.device = device
            model.model = model.model.to(device)
            if mode == "intel" or mode == "nvidia":
                model.device_arrays = True

        else:
            with open(str_fname,'rb') as pfile:
                model = pickle.load(pfile)
        returnval = 1
    LOADED_MODELS[c_id(c_model)] = model
    return returnval

def load_from_python(model):
    unloaded_models = [k for k, v in LOADED_MODELS.items() if v is None]
    num_models = len(unloaded_models)
    cdef MLIAPModelPythonKokkosDevice * lmp_model

    if num_models == 0:
            raise ValueError("No model in the waiting area.")
    elif num_models > 1:
            raise ValueError("Model is amibguous, more than one model in waiting area.")
    else:
        c_id = unloaded_models[0]
        LOADED_MODELS[c_id]=model
        lmp_model = <MLIAPModelPythonKokkosDevice *> <uintptr_t> c_id
        lmp_model.connect_param_counts()


cdef public void MLIAPPYKokkos_unload_model(MLIAPModelPythonKokkosDevice * c_model) with gil:
    del LOADED_MODELS[c_id(c_model)]

cdef public int MLIAPPYKokkos_nparams(MLIAPModelPythonKokkosDevice * c_model) with gil:
    return int(retrieve(c_model).n_params)

cdef public int MLIAPPYKokkos_nelements(MLIAPModelPythonKokkosDevice * c_model) with gil:
    return int(retrieve(c_model).n_elements)

cdef public int MLIAPPYKokkos_ndescriptors(MLIAPModelPythonKokkosDevice * c_model) with gil:
    return int(retrieve(c_model).n_descriptors)

class _USMArrayInterface:
    def __init__(self, interface):
        self.__sycl_usm_array_interface__ = interface

cdef create_array(void *pointer, shape, is_int, q=None):
    size=1
    for i in shape:
        size = size*i

    if (mode == "nvidia"):
        mem = cupy.cuda.UnownedMemory(ptr=int( <uintptr_t> pointer), owner=None, size=size)
        memptr = cupy.cuda.MemoryPointer(mem, 0)
        type=cupy.double
        if (is_int):
            type=cupy.int32
        return cupy.ndarray(shape, type, memptr=memptr)
    elif (mode == "intel"):
        usm_type = "device"

        # Wrap the raw pointer as unowned USM memory
        mem = dpmem.as_usm_memory(
            _USMArrayInterface({
                "data": (int(<uintptr_t> pointer), True),
                "shape": shape,
                "strides": None,
                "typestr": "<i4" if is_int else "<f8",
                "version": 1,
                "syclobj": q,
            })
        )

        dtype = dpnp.int32 if is_int else dpnp.float64
        return dpnp.ndarray(shape, dtype=dtype, buffer=mem)
    else:
        if (len(shape) == 1 ):
            if (is_int):
                return numpy.asarray(<int[:shape[0]]>pointer)
            else:
                return numpy.asarray(<double[:shape[0]]>pointer)
        else:
            if (is_int):
                return numpy.asarray(<int[:shape[0],:shape[1]]>pointer)
            else:
                return numpy.asarray(<double[:shape[0],:shape[1]]>pointer)


cdef public void MLIAPPYKokkos_compute_gradients(MLIAPModelPythonKokkosDevice * c_model, MLIAPDataKokkosDevice * data) with gil:
    if (mode == "nvidia"):
        torch.cuda.nvtx.range_push("set data fields")
    model = retrieve(c_model)
    n_d = data.ndescriptors
    n_a = data.nlistatoms

    # Handle empty neighbor/list atom sets.
    if n_a == 0:
        data.energy[0] = 0.0
        data.update_charges()
        if (mode == "nvidia"):
            torch.cuda.nvtx.range_pop()
        return

    if (mode == "intel"):
        q = dpctl.SyclQueue(PyCapsule_New(data.sycl_queue, b"SyclQueueRef", NULL))
    else:
        q = None
    # Make arrays from raw Kokkos/device pointers.
    elem_cp = create_array(data.ielems, (n_a,), True, q)
    en_cp   = create_array(data.eatoms, (n_a,), False, q)
    beta_cp = create_array(data.betas, (n_a, n_d), False, q)
    desc_cp = create_array(data.descriptors, (n_a, n_d), False, q)
    charges_cp = create_array(data.charges, (n_a,), False, q)
    charge_beta_cp = create_array(data.charge_betas, (n_a, n_d), False, q)

    if (mode == "nvidia"):
        torch.cuda.nvtx.range_pop()

    # Invoke Python model on device/host arrays.
    if (mode == "nvidia"):
        torch.cuda.nvtx.range_push("call model")

    model(elem_cp, desc_cp, beta_cp, en_cp, charges_cp, charge_beta_cp)

    if (mode == "nvidia"):
        torch.cuda.nvtx.range_pop()

    # Charge bookkeeping
    data.update_charges()

    # Get the total energy from the atom energy.
    if (mode == "nvidia"):
        energy = cupy.sum(en_cp)
    elif (mode == "intel"):
        energy = dpnp.sum(en_cp)
    else:
        energy = numpy.sum(en_cp)

    data.energy[0] = <double> energy
    return

