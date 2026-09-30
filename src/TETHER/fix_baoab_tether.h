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
#include "safe_pointers.h"
#include "update.h"

#include <string>

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
  int setmask() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  void end_of_step() override;

  double memory_usage() override;
  double compute_vector(int) override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

 private:
  double theta;          // dimensionless omega*dt stiff/soft threshold
  int refresh_every;     // steps between curvature-block refresh
  double eps;            // finite-difference probe displacement
  int eps_was_set;       // 1 if user gave "eps", else default from skin

  // thermally-averaged effective curvature (CLAUDE.md convention #3):
  // 0 (default) = legacy per-refresh instantaneous curvature, unchanged.
  // >0 = accumulate the raw Hessian over this many refresh cycles; the
  // atom stays masked/tethered throughout using a stable working estimate
  // (om[i]/evecs[i] updated only at the first and last sample -- never
  // thrashed every refresh), then freezes the N-sample average permanently;
  // see refresh_blocks().
  int heff_samples;

  int mollify;           // 0 = stage-1 static center, 1 = Newton mollifier
  int newton_iters;      // minimum damped-Newton iterations
  double newton_damp;    // initial Newton damping factor
  double solve_tol_rel;  // Newton convergence tol, relative to thermal amplitude
  int use_local_partial_force;   // 0 = pair->single() loop, 1 = pair->local_partial_force()

  // stage-3 adaptive-dt + event machinery (all inert when adapt == 0)
  int adapt;              // 0 = stage-1/2 fixed dt, 1 = adaptive dt + demote/quench/forgive
  double c_acc;           // target Verlet accuracy for the unresolved band
  double dt_min, dt_max;  // quantized-ladder floor/ceiling (ladder: dt_max/2^dt_lvl)
  double ke_rel;          // kinetic over-excitation threshold, multiples of kT
  double gamma_quench;    // strong-friction OU coefficient during cooldown

  // excursion guard for unmasked/soft modes (Stage 5, adapt == 1 only):
  // 0 (default) = legacy behavior, kinetic_guard_and_quench() only watches
  // currently-masked modes exactly as before. >0 = also (a) include soft
  // modes in the kinetic-energy check and (b) re-tether and locally
  // sub-step an atom's remaining drift for the rest of THIS step, before
  // its next scheduled refresh, once it has drifted this far (minimum-
  // imaged) from x0[i] (its position at the last refresh) -- soft modes
  // have no analytic tether and are otherwise invisible to every existing
  // guard; see kinetic_guard_and_quench() and sub_step_free_atom().
  double refresh_skin;

  // how a drift_hot trip (refresh_skin > 0 only) is handled:
  // 0 (retether, default) = legacy Attempt 3-5 behavior, unchanged --
  // re-tether in place and locally sub-step this atom's remaining dtby2
  // via sub_step_free_atom(). 1 (preshrink) = take NO action mid-step at
  // all (not even a bookkeeping refresh -- see kinetic_guard_and_quench()'s
  // doc comment for why calling refresh_one_atom() without reprojecting
  // q1[]/p1[] is unsound); this atom finishes the current step completely
  // unperturbed, and end_of_step() lowers update->dt system-wide before
  // the NEXT step instead if the drift is still there once the step ends.
  int drift_response;

  // permanent per-atom/per-event diagnostic log (plan Sec 6.1's required
  // per-step/per-event log): empty (default) = disabled, zero overhead --
  // every log_event() call site is gated on a plain "if (event_log_fp)"
  // check via SafeFilePtr's implicit nullptr conversion. When set, each
  // MPI rank opens its OWN file (comm->me-suffixed when nprocs > 1, so
  // this stays correct/halo-local under future domain decomposition with
  // no gather/collective) and logs REFRESH/GUARD_KE/GUARD_DRIFT/DEMOTE/
  // PRESHRINK events as they occur; see log_event().
  std::string event_log_filename;
  SafeFilePtr event_log_fp;

  double om_split;    // spectral-gap-detected stiff/soft split, one-shot (adapt==1 only)
  int om_split_set;   // 0 until compute_om_split() has run once
  int dt_lvl;          // current rung on the dt_max/2^dt_lvl ladder, -1 = uninitialized

  // MACE's much larger r_cut (vs. EAM/LJ) puts a bulk Pd atom's full
  // neighbor list around ~150 within 8 A; 96 was sized for the shorter
  // pairwise cutoffs this fix was originally validated against.
  static constexpr int JMAX = 256;    // per-atom response-block neighbor cap

  bigint last_refresh_step;
  int need_refresh;

  // set by end_of_step()'s emergency dt-shrink (drift_response == preshrink
  // only) to ntimestep + a short backoff; guards update_dt()'s own relax-up
  // hysteresis so a scheduled refresh_blocks() call can't raise dt_lvl back
  // down before the atom that justified the shrink has actually cleared --
  // update_dt()'s om_unres is blind to which atom that was. -1 = no active
  // floor (default; never blocks relax-up).
  bigint dt_floor_until;

  // rank-local trigger, set by sub_step_free_atom() when a drift_hot atom's
  // local excursion would exceed the neighbor list's trust radius (see
  // sub_step_free_atom()'s doc comment) -- Allreduced to a same-on-every-
  // rank decision at the end of initial_integrate(), same pattern as
  // need_refresh above, then translated into the standard Fix::force_
  // reneighbor/next_reneighbor request so neighbor->decide() forces a real,
  // synchronized reneighboring (comm->exchange()/borders()/neighbor->build())
  // before the NEXT step's initial_integrate() runs -- not a new collective
  // call invented here, just the existing LAMMPS mechanism fix_deposit.cpp
  // etc. already use for the same purpose.
  int need_reneighbor;

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

  // thermally-averaged effective curvature accumulator (heff_samples > 0
  // only): Hacc is the running SUM of the raw (pre-diagonalization) 3x3
  // mass-normalized symmetrized Hessian, row-major 9 doubles/atom. om[i]/
  // evecs[i] are (re)diagonalized from Hacc/n_hess_samples[i] only at the
  // first sample and once more when n_hess_samples[i] reaches heff_samples
  // (frozen from then on) -- never on the samples in between, so the
  // atom's mask decision stays stable through the whole window instead of
  // thrashing every refresh; see refresh_blocks().
  double **Hacc;            // running Hessian sum (nmax x 9)
  int *n_hess_samples;      // samples accumulated so far, 0..heff_samples (nmax)

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

  // stage-4 (ghost/MPI communication) diagnostic: counts times a stored
  // sparse-block neighbor (Jtag[i][kk]) could not be resolved via
  // atom->map() because it had drifted outside the current ghost cutoff
  // since the last refresh -- the back-reaction contribution is silently
  // dropped (fails safe) rather than crashing; see force_moll(). Always 0
  // on a single rank with a generous cutoff/skin.
  bigint n_ghost_miss;

  // halo-locality cost accounting (plan Sec 6.1 honest force-call cost):
  // partial_force() calls (Newton inner loop, mollify=yes) vs. full local
  // recompute_forces_local() calls -- exposed via compute_vector().
  bigint n_partial_force_calls, n_recompute_calls;

  void refresh_blocks();
  void refresh_one_atom(int i, const double *xi, bool instantaneous);
  void force_clear_local();
  void recompute_forces_local();
  void modal_force_correction(int i, double mi, double *fcorr) const;

  // stage-2 additions
  void solve_center(int i, double mi, double kT, double *cnew);
  bool partial_force(int i, const double *xtrial, double *fout, double *fneigh = nullptr,
                      bool live_neighbor_positions = false) const;
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
  void apply_new_dt(double dt_new);
  void demote_atom(int i, bigint backoff_steps, const char *source);
  void kinetic_guard_and_quench(int i, double mi, double kT, bool *msk, double *q1, double *p1,
                                 bool &quench_active, bool &sub_stepped);
  void sub_step_free_atom(int i, double mi, const double *x_mid, const double *v_mid,
                           double remaining_dt, double kT);

  // Attempt-8 diagnostic logging: no-op when event_log_fp is unset (see
  // event_log_filename above). ke/ke_cap/drift/skin/om/dt use -1.0 as the
  // "not applicable to this event" sentinel (every logged quantity is
  // physically non-negative).
  void log_event(const char *event, tagint tag, double ke, double ke_cap, double drift,
                 double skin, const double *om, double dt, const std::string &note);
};

}    // namespace LAMMPS_NS

#endif
#endif
