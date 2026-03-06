#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:20:00
#PBS -q prod
#PBS -A Catalyst
#PBS -l filesystems=home:flare

cd ${PBS_O_WORKDIR}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12
NTHREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PLACES=cores

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

EXE=../src/lmp_aurora_kokkos

INPUT=in.lj
NSTEPS=10

#X=2
#Y=2
#Z=2
#EXE_ARG="-in in.lj -var nsteps ${NSTEPS} -var x ${X} -var y ${Y} -var z ${Z} "

XYZ=`python3 decompose_global_problem_xyz.py ${NNODES} 0 2`
EXE_ARG="-in in.lj -var nsteps ${NSTEPS} ${XYZ} "
EXE_ARG+=" -k on g 1 -sf kk -pk kokkos newton off neigh full gpu/aware on "

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:53:54:55:56:57:58"

AFFINITY=""
AFFINITY=" gpu_tile_compact.sh "

#export LMP_STAT_HANG_RANK=11 # MPI rank 11 will hang on CPU
#export LMP_STAT_HANG_RANK_GPU=11 # MPI rank 11 will hang on GPU
#export LMP_STAT_SEGFAULT_RANK_GPU=11 # MPI rank 11 will hang on GPU
#export LMP_STAT_HANG_MINUTES=1 # rank will sleep for 1 minutes if hang on host (gpu hangs indefinitely)
export LMP_STAT_STEP=9 # hang will occur on step 9

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ${EXE} ${EXE_ARG}"
echo "COMMAND= ${COMMAND}"
${COMMAND}
