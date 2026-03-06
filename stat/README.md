This is a modified version of LAMMPS that will hang or segfault in GPU kernels to help with testing debug tools. The workload can be run on any node count for any time duration as a means of exploring how typical users might encounter issues and rely on debug tools to resolve them.


# Obtain LAMMPS code
This is easy. Just clone chris’ repo and use the `aurora_stat` branch. Note, this is a slightly outdated version of LAMMPS. This does not impact the ability to test hangs and segfaults with debug tools, but is important to keep in mind that there are differences with the latest develop version of LAMMPS (e.g. newer version of Kokkos and support for building with Makefiles dropped).

```
$ git clone https://github.com/cjknight/lammps.git
$ cd lammps
$ git checkout aurora_stat
```

For GPU-accelerated systems, users will leverage one of two code paths/packages depending on the specific capabilities they need for their simulation: Kokkos and GPU. The Kokkos package implements exactly what the name suggests, and the SYCL backend is primarily used on Aurora. The GPU package implements a similar abstraction for a smaller fraction of LAMMPS features, and the OpenCL backend is primarily used on Aurora.


## Running workloads with the Kokkos/SYCL backend

The LAMMPS executable can be compiled simply as follows using the default software environment on Aurora.

```
$ cd lammps/src
$ make yes-KOKKOS
$ make aurora_kokkos -j 32
```

If any changes are needed for compiling and linking LAMMPS, then those can be made in the `lammps/src/MAKE/MACHINES/Makefile.aurora_kokkos`. That Makefile could also easily be copied to create a new file, such as `lammps/src/MAKE/MACHINES/Makefile.aurora_blah`, and then compiled as follows.

```
cd lammps/src
make aurora_blah -j 32
```

The `make yes-KOKKOS` command only needs to be run one time. That command simply copies files from `lammps/src/KOKKOS` into `lammps/src`. 

*Important* : The `lammps/src/KOKKOS/Install.sh` script was modified to only include a minimal subset of source files. This is required because of the `-g` compiler flag that is required by some of the debugging tools. 

## Running the Kokkos workload
The `lammps/stat` directory has a modified version of the `bench/in.lj` workload that is setup to run different sized systems for different number of timesteps, both of which are specified in the job submission script `submit_kokkos.sh`. One could submit that job script as-is in a batch job using the following commands, but care is needed as that will hang indefinitely.

```
$ cd ../stat
$ qsub -l select=4 -A Catalyst ./submit_kokkos.sh
```

Specifying details of the LAMMPS workload

* The size of the simulation workload is weak-scaled up depending on the number of compute nodes requested. LAMMPS has good weak-scaling performance, so the runtime should be close to or slightly increase as the node count increases.
* The `NSTEPS` variable in the job script can be used to increase the runtime. The runtime increases roughly linearly with the number of simulation steps. 

Environment variables are exported to control how LAMMPS will hang or segfault on CPU or GPU.

* `export LMP_STAT_STEP=9` # hang/segfault will occur on step 9 of simulation

* `#export LMP_STAT_HANG_RANK=11` # MPI rank 11 will hang on CPU. This is usually commented because focus is on GPU hangs and segfaults.
* `#export LMP_STAT_HANG_MINUTES=1` # rank will sleep for 1 minute(s) if hang on CPU (gpu hangs indefinitely)

* `export LMP_STAT_HANG_RANK_GPU=11` # MPI rank 11 will hang on GPU

These environment variables are exported in the batch job script to trigger the desired behavior.

## Source code modifications

High-level changes are made in the verlet.cpp and verlet_kokkos.cpp files to change behavior based on values of the environment variables. Search for `LMP_STAT_` in those files. 

Changes in the GPU kernel can be found in the `pair_lj_cut_kokkos.cpp` file in the `PairLJCutKokkos<DeviceType>::compute()` function.

```
  if(stat_hang) {
    using member_type = Kokkos::TeamPolicy<DeviceType>::member_type;
    Kokkos::TeamPolicy<DeviceType> policy(1024, Kokkos::AUTO());
    Kokkos::parallel_for("HangForever", policy, KOKKOS_LAMBDA(member_type team_member){
                      if(team_member.team_rank() == 0) team_member.team_barrier();
                    }
                    );
  }
```

This code can be modifed as desired to achieve the desired outcome.

## Running workloads with the GPU/OpenCL backend

# If building with GPU package (OpenCL backend)
```
$ mkdir build
$ cd build
$ cmake -C ../cmake/presets/aurora.cmake ../cmake
$ make -j 16
```

## submit batch job: Replace 'Catalyst' with ALCF project name
```
$ cd ../stat
$ qsub -l select=4 -A Catalyst ./submit.sh
```

