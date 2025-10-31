#!/bin/bash -l
#PBS -l select=1
#PBS -l place=scatter
#PBS -l walltime=0:20:00
#PBS -q prod
#PBS -A Catalyst
#PBS -l filesystems=home:flare

cd ${PBS_O_WORKDIR}

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=24
NTHREADS=1

export MPICH_GPU_SUPPORT_ENABLED=1
export OMP_NUM_THREADS=${NTHREADS}
export OMP_PLACES=cores

NTOTRANKS=$(( NNODES * NRANKS_PER_NODE ))

EXE=../build/lmp

INPUT=in.lj
NSTEPS=1000

#X=2
#Y=2
#Z=2
EXE_ARG="-in in.lj -var nsteps ${NSTEPS} -var x ${X} -var y ${Y} -var z ${Z} "

XYZ=`python3 decompose_global_problem_xyz.py ${NNODES} 0 2`
EXE_ARG="-in in.lj -var nsteps ${NSTEPS} ${XYZ} "

EXE_ARG+=" -pk gpu 1 -sf gpu " 
#EXE_ARG+=" -pk gpu 1 -pk omp ${NTHREADS} -sf hybrid gpu omp " 

MPI_ARG="-n ${NTOTRANKS} --ppn ${NRANKS_PER_NODE} "
#MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:53:54:55:56:57:58"
MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:7:8:9:10:11:12:53:54:55:56:57:58:59:60:61:62:63:64"
#MPI_ARG+="--cpu-bind list:1:2:3:4:5:6:7:8:9:10:11:12:13:14:15:16:17:18:19:20:21:22:23:24:53:54:55:56:57:58:59:60:61:62:63:64:65:66:67:68:69:70:71:72:73:74"

AFFINITY=""
AFFINITY=" gpu_tile_compact.sh "

export LMP_STAT_HANG_RANK=11 # MPI rank 11 will hang
export LMP_STAT_HANG_MINUTES=1 # rank will sleep for 1 minutes
export LMP_STAT_STEP=99 # hang will occur on step 99

COMMAND="mpiexec ${MPI_ARG} ${AFFINITY} ${EXE} ${EXE_ARG}"
echo "COMMAND= ${COMMAND}"
${COMMAND}
