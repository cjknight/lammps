/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_MLIAP_DESCRIPTOR_MTP_H
#define LMP_MLIAP_DESCRIPTOR_MTP_H

#include "mliap_descriptor.h"

namespace LAMMPS_NS {

class MLIAPDescriptorMTP : public MLIAPDescriptor, virtual protected Pointers {
 public:
  MLIAPDescriptorMTP(class LAMMPS *);
  ~MLIAPDescriptorMTP() override;

  void init() override;

  void compute_descriptors(class MLIAPData *) override;
  void compute_descriptor_gradients(class MLIAPData *) override;
  void compute_forces(class MLIAPData *) override;
  void compute_forces_from_coeffs(class MLIAPData *, double **coeffs, double *phi = nullptr);
  void compute_force_gradients(class MLIAPData *) override;
  bool printed = false;

  static constexpr int MAX_N_RADIAL = 12;
  static constexpr int MAX_N_RF = 8;

 protected:
  enum BasisType {
    NU0,
    NU1_DOT,
    NU2_FROB,
    NU0_X_NU1SQ,
    NU0_X_NU1_NU1,
    V_T_V,
    NU0_X_NU0,
    NU0_SQ
  };

  struct BasisSpec {
    BasisType type;
    int mu[4];
  };

  double cutoff;
  double rmin;

  int n_radial;
  int n_rf;

  int max_nu;
  int max_level;

  std::vector<std::string> species;
  std::vector<BasisSpec> basis_specs;

  // radial_coeffs[mu][itype][jtype][n]
  std::vector<std::vector<std::vector<std::vector<double>>>> radial_coeffs;

  void build_basis_index();

  double cutoff_function(double r);

  void chebyshev_basis(double r,
                       std::vector<double>& basis,
                       std::vector<double>& dbasisdr);

  void compute_radial_functions(double r,
                                int itype,
                                int jtype,
                                std::vector<double> &fmu);

};    // namespace LAMMPS_NS

}
#endif
