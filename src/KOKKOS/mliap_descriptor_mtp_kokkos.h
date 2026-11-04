#ifndef LMP_MLIAP_DESCRIPTOR_MTP_KOKKOS_H
#define LMP_MLIAP_DESCRIPTOR_MTP_KOKKOS_H

#include "mliap_descriptor_kokkos.h"
#include "mliap_descriptor_mtp.h"
#include "mliap_data_kokkos.h"

#include <Kokkos_Core.hpp>

namespace LAMMPS_NS {

template <class DeviceType>
class MLIAPDescriptorMTPKokkos :
    public MLIAPDescriptorMTP,
    public MLIAPDescriptorKokkos<DeviceType>, 
    virtual protected Pointers {
 public:
  typedef ArrayTypes<DeviceType> AT;

  MLIAPDescriptorMTPKokkos(class LAMMPS *);
  ~MLIAPDescriptorMTPKokkos() override;
  
  void compute_descriptors(class MLIAPData *) override;
  void compute_forces(class MLIAPData *) override;
  void compute_force_gradients(class MLIAPData *) override;
  void compute_descriptor_gradients(class MLIAPData *) override;
  void init() override;
  double memory_usage() override;
  void compute_forces_from_coeffs(
    MLIAPDataKokkos<DeviceType> *,
    DAT::tdual_double_2d_lr &coeffs,
    typename AT::t_double_1d phi = typename AT::t_double_1d());

 private:
  Kokkos::View<double ****, DeviceType> k_radial_coeffs;
  Kokkos::View<int *, DeviceType> k_specs_type;
  Kokkos::View<int **, DeviceType> k_specs_mu;
};

}  // namespace LAMMPS_NS

#endif