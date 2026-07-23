#!/bin/bash

BASE=/Users/geraldc/Desktop/lammps/build

cd ${BASE}

/opt/homebrew/bin/cmake \
    -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_INSTALL_PREFIX=$(pwd) \
    -D CMAKE_C_COMPILER=$(brew --prefix llvm)/bin/clang \
    -D CMAKE_CXX_COMPILER=$(brew --prefix llvm)/bin/clang++ \
    -D CMAKE_CXX_FLAGS=" -DMLIAP_ACE " \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_STANDARD_REQUIRED=ON \
    -D BUILD_MPI=ON \
    -D BUILD_SHARED_LIBS=ON \
    -D PKG_KOKKOS=ON \
    -D Kokkos_ENABLE_OPENMP=ON \
    -D Kokkos_ENABLE_SERIAL=ON \
    -D BUILD_OMP=ON \
    -D PKG_MOLECULE=ON \
    -D PKG_KSPACE=ON \
    -D PKG_ML-IAP=ON \
    -D PKG_ML-SNAP=ON \
    -D MLIAP_ENABLE_PYTHON=ON \
    -D PKG_ML-PACE=ON \
    -D PKG_PYTHON=ON \
    -D Python_FIND_VIRTUALENV=ONLY \
    -D PYTHON_EXECUTABLE=/Users/gcollom/Desktop/lammps/.venv/bin/python3.11 \
    -D PYTHONPATH=/Users/gcollom/Desktop/lammps/.venv/lib/python3.11/site-packages \
    ../cmake


#    -D OpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp" \
#    -D OpenMP_CXX_LIB_NAMES="omp" \
#    -D OpenMP_omp_LIBRARY=/opt/homebrew/opt/libomp/lib/libomp.dylib \
#    -D OpenMP_CXX_INCLUDE_DIR=/opt/homebrew/opt/libomp/include \
#    -D Python_INCLUDE_DIR=/Users/geraldc/Desktop/lammps/.venv/include/python3.12 \
#    -D PYTHON_LIBRARY=/Users/geraldc/Desktop/lammps/.venv/lib/libpython3.12.dylib \
#    -D PYTHON_INLCUDE_DIR=/Users/geraldc/Desktop/lammps/.venv/include/python3.12 \
#    -D PYTHON_EXECUTABLE=/opt/homebrew/bin/python3 \
#    -D BUILD_OMP=ON \
#    -D OpenMP_CXX_FLAGS="-fopenmp -I/opt/homebrew/opt/libomp/include" \
#    -D OpenMP_CXX_LIB_NAMES="omp" \
#    -D OpenMP_ROOT=/opt/homebrew/opt/libomp \
#    -D OpenMP_omp_LIBRARY=$OpenMP_ROOT/lib/libomp.dylib \

#    -D CMAKE_CXX_COMPILER=mpicxx-mpich-gcc14 \
#    -D CMAKE_C_COMPILER=mpicc-mpich-gcc14 \
