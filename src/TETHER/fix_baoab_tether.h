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
#include "update.h"

namespace LAMMPS_NS {

// Stage-2 frozen-Hessian mollified integrator: each atom in the fix group
// is drift-integrated as if tethered to a harmonic center, for whichever of
// its 3 per-atom curvature-block eigenmodes are locally stiff
// (omega*dt > theta); soft modes and all non-group atoms fall back to plain
// free drift / the inherited B-A-O-A-B Ornstein-Uhlenbeck thermostat exactly
// as implemented by FixBAOAB. With "mollify yes" (stage 2), the center is
// the atom's INSTANTANEOUS clamped equilibrium given its current cage
// (damped-Newton solve every step, via a local partial-force kernel over
// its own neighbor list -- no mollify keyword or Newton state at all when
// "mollify no", which reduces exactly to stage 1's static-center behavior.

class FixBAOABTether : public FixBAOAB {
 public:
  FixBAOABTether(class LAMMPS *, int, char **);
  ~FixBAOABTether() override;

  void init() override;
  void init_list(int, class NeighList *) override;
  void initial_integrate(int) override;
  void final_integrate() override;

  double memory_usage() override;
  double compute_vector(int) override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;

 private:
  double theta;          // dimensionless omega*dt stiff/soft threshold
  int refresh_every;     // steps between curvature-block refresh
  double eps;            // finite-difference probe displacement
  int eps_was_set;       // 1 if user gave "eps", else default from skin

  int mollify;           // 0 = stage-1 static center, 1 = Newton mollifier
  int newton_iters;      // minimum damped-Newton iterations
  double newton_damp;    // initial Newton damping factor
  double solve_tol_rel;  // Newton convergence tol, relative to thermal amplitude

  // stage-3 adaptive-dt + event machinery (all inert when adapt == 0)
  int adapt;              // 0 = stage-1/2 fixed dt, 1 = adaptive dt + demote/quench/forgive
  double c_acc;           // target Verlet accuracy for the unresolved band
  double dt_min, dt_max;  // quantized-ladder floor/ceiling (ladder: dt_max/2^dt_lvl)
  double ke_rel;          // kinetic over-excitation threshold, multiples of kT
  double gamma_quench;    // strong-friction OU coefficient during cooldown

  double om_split;    // spectral-gap-detected stiff/soft split, one-shot (adapt==1 only)
  int om_split_set;   // 0 until compute_om_split() has run once
  int dt_lvl;          // current rung on the dt_max/2^dt_lvl ladder, -1 = uninitialized

  static constexpr int JMAX = 96;    // per-atom response-block neighbor cap

  bigint last_refresh_step;
  int need_refresh;

  class NeighList *list;    // full, non-occasional: this atom's own pair neighbors

  // per-atom persistent state (only meaningful for group atoms with a
  // valid block; zero/harmless default for everything else)
  double **x0;       // raw position at last refresh (Newton warm-start fallback,
                      // FD-probe base point)
  double **c;         // last Newton-converged (or, mollify=no, static) center (nmax x 3)
  double **evecs;    // curvature-block eigenvectors, mode-major:
                      //   evecs[i][3*m+k] = k-th Cartesian component of
                      //   the m-th eigenvector of atom i's block (nmax x 9)
  double **om;        // per-mode angular frequency, 0 if unset (nmax x 3)

  // sparse per-atom response blocks J_i (mollify=yes only): for each group
  // atom i, nJ[i] (neighbor local index, 3x3 block) pairs restricted to
  // atom i's own pair-style neighbor list -- halo-local by construction,
  // per D.10.4's own support argument (zero outside the Hessian row).
  int *nJ;                 // number of stored neighbor blocks per atom (nmax)
  tagint **Jtag;            // neighbor atom TAG per stored block (nmax x JMAX)
  double ***Jblk;           // 3x3 block per stored neighbor (nmax x JMAX x 9)

  // stage-3 per-atom event-machinery state (harmless zero default when
  // adapt == 0; grown/exchanged like x0/c/om above)
  int *demote_count;        // chronic-failure counter, exponential backoff (nmax)
  bigint *last_demote;      // step of last demotion, for forgiveness (nmax)
  bigint *demote_cool;      // step until which this atom is Verlet-exiled (nmax)
  bigint *quench_exempt;    // step until which this atom skips the quench kick (nmax)
  double *last_ke;          // previous step's max masked-modal KE, suddenness test (nmax)

  // self-evaluation guard logs (computed always when mollify=yes; only
  // acted on -- i.e. actually demote a mode -- when adapt == 1, matching
  // the Python reference's own "& self.adapt" gating exactly)
  bigint n_guard_newton, n_guard_flip, n_guard_energy;

  // stage-3 event counters (0 unless adapt == 1)
  bigint n_demoted, n_kinetic_guard;

  // halo-locality cost accounting (plan Sec 6.1 honest force-call cost):
  // partial_force() calls (Newton inner loop, mollify=yes) vs. full local
  // recompute_forces_local() calls -- exposed via compute_vector().
  bigint n_partial_force_calls, n_recompute_calls;

  void refresh_blocks();
  void force_clear_local();
  void recompute_forces_local();
  void modal_force_correction(int i, double mi, double *fcorr) const;

  // stage-2 additions
  void solve_center(int i, double mi, double kT, double *cnew);
  bool partial_force(int i, const double *xtrial, double *fout) const;
  void force_moll();

  // stage-3 additions
  inline bool mode_is_stiff(int i, int m) const
  {
    if (!adapt) return (om[i][m] * update->dt) > theta;
    if (demote_cool[i] > update->ntimestep) return false;
    double om_max = MAX(om[i][0], MAX(om[i][1], om[i][2]));
    double om_min = MIN(om[i][0], MIN(om[i][1], om[i][2]));
    if (!((om_max > om_split) && (om_min > 0.6 * om_split))) return false;
    return om[i][m] > om_split;
  }
  void compute_om_split();
  void update_dt();
  void demote_atom(int i, bigint backoff_steps);
  void kinetic_guard_and_quench(int i, double mi, double kT, bool *msk, const double *p1,
                                 bool &quench_active);
};

}    // namespace LAMMPS_NS

#endif
#endif
