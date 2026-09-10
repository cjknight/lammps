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
FixStyle(baoab/tether,FixBAOABTether);
// clang-format on
#else

#ifndef LMP_FIX_BAOAB_TETHER_H
#define LMP_FIX_BAOAB_TETHER_H

#include "fix_baoab.h"

namespace LAMMPS_NS {

// Stage-1 frozen-Hessian mollified integrator: each atom in the fix group
// is drift-integrated as if tethered to a frozen harmonic center, reset
// every "refresh" steps, for whichever of its 3 per-atom curvature-block
// eigenmodes are locally stiff (omega*dt > theta); soft modes and all
// non-group atoms fall back to plain free drift / the inherited B-A-O-A-B
// Ornstein-Uhlenbeck thermostat exactly as implemented by FixBAOAB.

class FixBAOABTether : public FixBAOAB {
 public:
  FixBAOABTether(class LAMMPS *, int, char **);
  ~FixBAOABTether() override;

  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;

  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

 private:
  double theta;          // dimensionless omega*dt stiff/soft threshold
  int refresh_every;     // steps between curvature-block refresh
  double eps;            // finite-difference probe displacement
  int eps_was_set;       // 1 if user gave "eps", else default from skin

  bigint last_refresh_step;
  int need_refresh;

  // per-atom persistent state (only meaningful for group atoms with a
  // valid block; zero/harmless default for everything else)
  double **x0;       // frozen harmonic center set at last refresh (nmax x 3)
  double **evecs;    // curvature-block eigenvectors, mode-major:
                      //   evecs[i][3*m+k] = k-th Cartesian component of
                      //   the m-th eigenvector of atom i's block (nmax x 9)
  double **om;        // per-mode angular frequency, 0 if unset (nmax x 3)

  void refresh_blocks();
  void force_clear_local();
  void recompute_forces_local();
  void modal_force_correction(int i, double mi, double *fcorr) const;
};

}    // namespace LAMMPS_NS

#endif
#endif
