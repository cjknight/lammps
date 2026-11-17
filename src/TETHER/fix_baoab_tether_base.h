/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_FIX_BAOAB_TETHER_BASE_H
#define LMP_FIX_BAOAB_TETHER_BASE_H

#include "fix_baoab.h"

#include <vector>

namespace LAMMPS_NS {

// Shared layer of the fix baoab/tether family (baoab/tether, baoab/hrespa): out-of-band
// whole-system force evaluations at scratch configurations, and saving/restoring what such
// an evaluation overwrites (forces, charges set by the pair style, accumulated energies and
// virials). Not a style itself.

class FixBAOABTetherBase : public FixBAOAB {
 protected:
  // style is the command name for error messages; FixBAOAB parses only the 4 required arguments
  FixBAOABTetherBase(class LAMMPS *, int, char **, const char *style);

  bigint n_recompute_calls;    // recompute_forces_local() calls

  void force_clear_local();
  // all force contributions at the current positions (ghosts must be current), as the
  // integrator computes them, without energies or virials
  void recompute_forces_local();

  // charges set by the pair style at every compute (Pair::kspace_response_flag, e.g. an LES
  // model): a scratch recompute leaves atom->q at the scratch geometry, so callers save the
  // real-geometry charges first and restore them afterwards
  bool charges_from_pair() const;
  void save_charges(std::vector<double> &qs) const;
  void restore_charges(const std::vector<double> &qs);

  // pair and kspace energies and virials (thermo output of the current step)
  static constexpr int NACCUM = 2 + 6 + 1 + 6;
  void save_accumulators(double *) const;
  void restore_accumulators(const double *);

 private:
  static int base_narg(class LAMMPS *, int, const char *);
};

}    // namespace LAMMPS_NS

#endif
