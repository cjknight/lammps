#!/bin/bash

#EXE=jlse_kokkos_sycl
EXE=jlse_kokkos_sycl_aot

#rm lmp_${EXE}*

TEAM=32

#for VECLEN in 8 16 32
#for VECLEN in 32
for VECLEN in 16
#for VECLEN in 8
do
#  touch kokkos_type.h 

  #for TILE in 1 2 4 8 16
#  for TILE in 8
  for TILE in 2
#  for TILE in 1
  do
#    touch pair_snap_kokkos.h 
#    touch pair_snap_kokkos_impl.h
    echo "Compiling VECLEN= ${VECLEN}  TILE= ${TILE}  TEAM= ${TEAM}"

    make clean-${EXE}
    make -j 32 ${EXE} CHRIS_EXTRA="-DCHRIS_VECLEN=${VECLEN} -DCHRIS_TILE=${TILE} -DCHRIS_TEAM=${TEAM}" 

    mv lmp_${EXE} lmp_${EXE}-${VECLEN}-${TILE}
  done

done
