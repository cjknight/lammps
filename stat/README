## clone chris’ repo or just the aurora_stat branch, either or

$ git clone https://github.com/cjknight/lammps.git
$ cd lammps
$ git checkout aurora_stat

## If building with GPU package (OpenCL backend)
$ mkdir build
$ cd build
$ cmake -C ../cmake/presets/aurora.cmake ../cmake
$ make -j 16

# submit batch job: Replace 'Catalyst' with ALCF project name
$ cd ../stat
$ qsub -l select=4 -A Catalyst ./submit.sh


## If building with Kokkos package (SYCL backend)
$ cd src
$ make yes-KOKKOS
$ make aurora_kokkos -j 32

# submit batch job: Replace 'Catalyst' with ALCF project name
$ cd ../stat
$ qsub -l select=4 -A Catalyst ./submit_kokkos.sh

# Edit LMP_STAT_# environment variables in submit.sh to trigger hang

#export LMP_STAT_HANG_RANK=11 # MPI rank 11 will hang on CPU
export LMP_STAT_HANG_RANK_GPU=11 # MPI rank 11 will hang on GPU
export LMP_STAT_HANG_MINUTES=1 # rank will sleep for 1 minutes if hang on host (gpu hangs indefinitely)
export LMP_STAT_STEP=9 # hang will occur on step 9
