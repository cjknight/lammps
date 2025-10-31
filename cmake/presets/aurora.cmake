# preset that enables KOKKOS and selects SYCL compilation with OpenMP
# enabled as well. Also sets some performance related compiler flags.
#set(BUILD_OMP ON CACHE BOOL "" FORCE)

set(PKG_OPENMP ON CACHE BOOL "" FORCE)
set(PKG_GPU ON CACHE BOOL "" FORCE)
set(GPU_API OPENCL CACHE STRING "" FORCE)
set(USE_STATIC_OPENCL_LOADER NO CACHE BOOL "" FORCE)

set(FFT "MKL" CACHE STRING "" FORCE)
set(FFT_SINGLE ON CACHE BOOL "" FORCE)

unset(USE_INTERNAL_LINALG)
unset(USE_INTERNAL_LINALG CACHE)
set(BLAS_VENDOR "Intel10_64_dyn")

set(CMAKE_CXX_COMPILER icpx CACHE STRING "" FORCE)
set(CMAKE_C_COMPILER icx CACHE STRING "" FORCE)
set(CMAKE_Fortran_COMPILER ifx CACHE STRING "" FORCE)
set(MPI_CXX_COMPILER "mpicxx" CACHE STRING "" FORCE)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -xSAPPHIRERAPIDS -g -O0 -ffp-model=fast -qoverride-limits -qopt-zmm-usage=high " CACHE STRING "" FORCE)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -xSAPPHIRERAPIDS -g -O0 -ffp-model=fast -qoverride-limits -qopt-zmm-usage=high " CACHE STRING "" FORCE)
