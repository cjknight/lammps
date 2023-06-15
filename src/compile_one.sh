#VERSION=jlse_kokkos_sycl_aot
#VERSION=jlse_kokkos_sycl_aot_vtune
#VERSION=polaris_gnu_kokkos
VERSION=polaris_oneapi_kokkos

make -j 32 ${VERSION}
#make -j 32 ${VERSION} CHRIS_EXTRA="-DCHRIS_VECLEN=16 -DCHRIS_TILE=2 -DCHRIS_TEAM=32 -DCHRIS_TILE_YI=8"
