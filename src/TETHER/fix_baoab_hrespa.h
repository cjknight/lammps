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
// Optional student (a second, cheap pair style, keywords student_style/student_coeff): the inner
// force is then either the student itself (inner_force student) or the "graft" (default with a
// student) F_h + F_s - F_hs, where F_hs is the student's own harmonic model at the same
// reference: the teacher's curvature plus only the student's anharmonicity. The graft is stored
// as one harmonic model with K - K_s and f_ref - f_s,ref.

class FixBAOABHRespa : public FixBAOABTetherBase {
 public:
  FixBAOABHRespa(class LAMMPS *, int, char **);
  ~FixBAOABHRespa() override;

  void init() override;
  void post_run() override;
  void setup(int) override;
  void initial_integrate(int) override;
  void final_integrate() override;
  double memory_usage() override;

 private:
  enum { HARMONIC, STUDENT, GRAFT };
  int inner_user, nin, inner_kind;
  double cut, wmax, eps;
  char *file_in, *file_out;
  bool ready;
  bigint natoms_ref;
  std::vector<double> xref, fref, mass_id;   // 3 (or 1) per atom ID, index = ID
  struct HModel {
    std::vector<bigint> rowptr;              // CSR rows by atom ID (size natoms + 2)
    std::vector<tagint> col;                 // column atom ID of each 3x3 block
    std::vector<double> K;                   // 9 per block, (a, b) row-major: K_{ia, jb}
  };
  HModel hm;                                 // the harmonic part of the inner force
  std::vector<double> fin, fs;               // inner force, student force (owned atoms, 3 each)
  bigint fin_step;                           // step at whose end fin was computed (-1 = none)

  // student
  std::string student_name;
  std::vector<std::string> student_settings, student_coeff;
  class Pair *student;
  bigint n_student_calls, n_student_calls_run0;

  void setup_model();
  void build_model_fd(class Pair *, std::vector<double> &, std::vector<tagint> &,
                      std::vector<tagint> &, std::vector<double> &);
  void read_model(const char *, std::vector<tagint> &, std::vector<tagint> &,
                  std::vector<double> &);
  void write_model(const char *) const;
  void finish_model(std::vector<tagint> &, std::vector<tagint> &, std::vector<double> &, HModel &);
  double fastest_frequency(const HModel &) const;
  void student_force();
  void inner_force();
};

}    // namespace LAMMPS_NS

#endif
#endif
