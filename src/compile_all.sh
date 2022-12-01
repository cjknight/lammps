#!/bin/bash

EXE=jlse_kokkos_sycl_aot

#rm lmp_${EXE}*

for VECLEN in 8 16 32
do
  touch kokkos_type.h 

  for TILE in 1 2 4 8 16
  do
    touch pair_snap_kokkos.h 
    echo "Compiling VECLEN= ${VECLEN}  TILE= ${TILE}"

    make -j 16 ${EXE} CHRIS_EXTRA="-DCHRIS_VECLEN=${VECLEN} -DCHRIS_TILE=${TILE}" 

    mv lmp_${EXE} lmp_${EXE}-${VECLEN}-${TILE}
  done

done
