/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(baoab/hrespa,FixBAOABHRespa);
// clang-format on
#else

#ifndef LMP_FIX_BAOAB_HRESPA_H
#define LMP_FIX_BAOAB_HRESPA_H

#include "fix_baoab_tether_base.h"

#include <vector>

namespace LAMMPS_NS {

// Hessian-built RESPA for systems without a molecular split (crystals, glasses, ...).
// Inner level = the harmonic model F_h(x) = f_ref - K (x - x_ref) of the whole force field,
// with K cut at "cut" and the acoustic sum rule restored; it is integrated with velocity
// Verlet substeps (one forward_comm each, no force-field calls). Outer level = the full force
// minus F_h, applied as half kicks at the ends of the step. The Langevin O step (gamma = 1/damp)
// is applied once per step, mid-loop.
// Phase 1 layout: the reference (x_ref, f_ref, K as CSR rows by atom ID) is replicated on every
// rank, built once by finite differences of whole-system recomputes (6N) or read from a file.

class FixBAOABHRespa : public FixBAOABTetherBase {
 public:
  FixBAOABHRespa(class LAMMPS *, int, char **);
  ~FixBAOABHRespa() override;

  void init() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  double memory_usage() override;

 private:
  int inner_user, nin;
  double cut, wmax, eps;
  char *file_in, *file_out;
  bool ready;
  bigint natoms_ref;
  std::vector<double> xref, fref, mass_id;   // 3 (or 1) per atom ID, index = ID
  std::vector<bigint> rowptr;                // CSR rows by atom ID (size natoms + 2)
  std::vector<tagint> col;                   // column atom ID of each 3x3 block
  std::vector<double> K;                     // 9 per block, (a, b) row-major: K_{ia, jb}
  std::vector<double> fh;                    // harmonic force on owned atoms (3 per atom)

  void setup_model();
  void build_model_fd();
  void read_model(const char *);
  void write_model(const char *) const;
  void finish_model(std::vector<tagint> &, std::vector<tagint> &, std::vector<double> &);
  void harmonic_force();
};

}    // namespace LAMMPS_NS

#endif
#endif
