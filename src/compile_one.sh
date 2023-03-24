make -j 1 jlse_kokkos_sycl_aot CHRIS_EXTRA="-DCHRIS_VECLEN=16 -DCHRIS_TILE=2 -DCHRIS_TEAM=32"
mv lmp_jlse_kokkos_sycl_aot lmp_jlse_kokkos_sycl_aot-16-2

