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
#include "force.h"
#include "safe_pointers.h"
#include "update.h"

#include <cmath>
#include <string>
#include <vector>

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
  void setup(int) override;
  void post_run() override;
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
  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;

 private:
  double theta;          // dimensionless omega*dt stiff/soft threshold
  int refresh_every;     // steps between curvature-block refresh

  // decouples how often the one-rung dt-ladder promotion hysteresis in
  // update_dt() gets re-checked from refresh_every: 0 (default) = disabled,
  // promotion is only re-evaluated when refresh_blocks() runs, exactly as
  // before. >0 = also re-run update_dt() from end_of_step() (already called
  // every step) every promote_every steps, reusing the currently cached
  // om[] -- no new curvature probing, no change to update_dt()'s own
  // one-rung/1.25x-margin/dt_floor_until logic. This exists because
  // dt_floor_until's lockout (drift_response preshrink only) otherwise eats
  // most of the time between triggers, leaving only 1-2 refresh_every-paced
  // opportunities per cycle to climb back toward dt_max.
  int promote_every;

  // confirmation threshold for EARLY release of an active dt_floor_until
  // lockout (drift_response preshrink only): 0 (default) = disabled,
  // exactly reproduces today's fixed-duration-only floor with no behavior
  // change. >0 = once this many CONSECUTIVE *fresh, scheduled*
  // refresh_blocks()->update_dt() calls (never promote_every's stale-om[]
  // recheck -- see update_dt()'s fresh_sample parameter) in a row each
  // independently find the same 1.25x relax-up margin update_dt() already
  // requires for its own one-rung climb, the lockout is released early
  // (dt_floor_until set to the current ntimestep) so that climb can resume
  // starting next call, instead of unconditionally waiting out the fixed
  // 3*refresh_every duration. Requires settle_confirm INDEPENDENT global
  // reductions over every flagged atom's just-reprobed curvature to agree
  // -- never a single stale or single freshly-probed sample.
  int settle_confirm;
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

  // solve_center()'s guard-threshold constants, exposed as keywords instead
  // of hardcoded locals/literals -- see water_newton_guard_unit_scale_bug
  // memory and the "Parameterize solve_center()'s guard-threshold constants"
  // plan for the full derivation. Each is a dimensionless multiple of a_th
  // (the thermal amplitude of the stiffest masked mode) -- newton_tol_rel
  // directly (replacing the old confusing "20.0*0.25" indirection ported
  // from the Python reference's separately-hardcoded 20.0), flip_rel and
  // newton_cap_rel likewise; energy_guard_rel is a multiple of kT (eharm's
  // own natural scale), not of a_th directly. Defaults reproduce today's
  // pre-existing hardcoded values exactly. Negative = disabled (matches
  // this file's own newton_debug_tag/pforce_debug_tag idiom for "off",
  // since every valid value here is a strictly positive multiplier and
  // there is no finite "neutral" value for an inequality threshold or a
  // clamp the way 1.0 is neutral for a blend weight).
  double newton_tol_rel;    // g_n: res > newton_tol_rel*a_th; default 5.0
  double flip_rel;          // g_f: flip > flip_rel*a_th; default 14.0
  double energy_guard_rel;  // g_e: eharm > energy_guard_rel*kT*(mvv2e when a_th_mass_scale); default 25.0
  double newton_cap_rel;    // per-iter Newton step cap multiplier; default 8.0

  // EXPERIMENTAL, opt-in, default 0 (byte-identical to today when unset).
  // solve_center()'s a_th (thermal amplitude of the stiffest masked mode)
  // and the eharm energy guard both compare against a quantity that is
  // missing a division by the atom's own mass and uses a `kT` already
  // divided by force->mvv2e where the dimensionally-correct formula needs
  // the raw force->boltz*t_target instead -- see
  // water_newton_guard_unit_scale_bug memory for the full derivation and
  // numeric verification (confirmed on 3 independent atoms/unit styles).
  // The corrected scale was tested directly against the validated H/Pd
  // production config and found to DESTABILIZE it on a second seed (mean
  // heavyT 299K -> 23,608K) -- adapt=yes feeds these guards straight into
  // demote_atom(), and H/Pd's whole validated history rests on their
  // CURRENT (dimensionally wrong, but empirically road-tested) scale.
  // 0 (default) = today's exact formula, unconditionally, for every
  // system including this one. 1 = use the corrected,
  // mass-and-unit-consistent scale instead -- intended for systems in
  // unit styles other than metal (e.g. water in `real` units), where the
  // uncorrected scale is instead far too TIGHT (guards fire on ordinary
  // thermal bond vibration). Never flip this on for an existing validated
  // metal-units config without re-running its own multi-seed regression.
  int a_th_mass_scale;
  // eligibility keyword: 0 = all_modes (default; an atom may be analytically clamped only
  // if EVERY mode exceeds 0.6*om_split -- the validated H/Pd rattler rule), 1 = bonded
  // (atoms with a bond skip the om_min test, so one stiff mode, e.g. an O-H stretch, can be
  // clamped alone). APPROXIMATION valid only while bond topology is fixed (no reactions);
  // not usable for MLIP/reactive systems. adapt=yes only.
  int eligibility_bonded;
  inline bool bonded_ok(int i) const { return eligibility_bonded && bonded[i]; }
  int newton_res_length;   // 1: solve_center residual in Newton-step length units (see newton_res_comp)
  double newton_res_comp(double r, double mi, double om_m) const;

  // force_eval full (default off, byte-identical when unused): validation-
  // only mode that bypasses partial_force()'s local kernel entirely --
  // for EVERY caller (solve_center()'s Newton residual, refresh_blocks()/
  // refresh_one_atom()'s FD-Hessian curvature probe, sub_step_free_atom()'s
  // exile sub-stepping, diag_check_staleness()'s per-step diagnostic) --
  // substituting a genuine whole-system recompute_forces_local() per trial
  // evaluation instead (see full_system_force()). O(N) per probe and
  // single-process only -- not a production option, see
  // fix_baoab_tether.rst. Deliberately NOT performance-tuned: this exists
  // to find out whether local_partial_force()'s neighbor-shell truncation
  // of the EAM embedding term explains a curvature discrepancy against a
  // full-system reference (e.g. lj_longstep_prototype.py's FD Hessian),
  // not to be fast. diag_check_staleness() in particular runs this every
  // step for every mollified atom -- keep debug runs short.
  int force_eval_full;

  // stage-3 adaptive-dt + event machinery (all inert when adapt == 0)
  int adapt;              // 0 = stage-1/2 fixed dt, 1 = adaptive dt + demote/quench/forgive
  double c_acc;           // target Verlet accuracy for the unresolved band
  double dt_min, dt_max;  // quantized-ladder floor/ceiling (ladder: dt_max/2^dt_lvl)
  double ke_rel;          // kinetic over-excitation threshold, multiples of kT
  double gamma_quench;    // strong-friction OU coefficient during cooldown

  // Baseline thermostat for modes that are NEITHER currently masked stiff
  // NOR in a post-demotion quench window. FixBAOABTether fully overrides
  // FixBAOAB::initial_integrate()/final_integrate() (the base class's own
  // uniform Langevin O-step is never invoked through this derived class --
  // confirmed by direct test: fix baoab/tether mollify no adapt no at a
  // conventional small dt, where every mode's om*dt stays below theta so
  // NOTHING is ever masked stiff, shows clean, steady cooling away from the
  // target temperature with zero active thermostat, not NVE-like
  // conservation). Default 0.0 (off) preserves prior behavior exactly;
  // gamma_soft > 0 applies an ordinary Cartesian OU kick (same c1/c2 form as
  // FixBAOAB's own O-step, mirrors lj_longstep_prototype.py's "gamma_soft"
  // constructor parameter) to every unmasked, non-quench mode every step.
  double gamma_soft;
  int gamma_soft_was_set;   // 0: gamma_soft defaults to gamma = 1/damp (one physical relaxation time)

  // gamma_quench auto (continuous, not a warm-up calibration): when set,
  // gamma_quench is overwritten every update_dt() call from the system's
  // CURRENT global stiffest-mode frequency instead of holding the literal
  // keyword value -- see update_dt()'s doc comment. gamma_quench itself
  // keeps its ordinary default (20.0) as the bootstrap value in effect
  // before the first refresh with a nonzero om_analytic_max.
  int gamma_quench_auto;
  double gamma_quench_last_logged;  // last value written to event_log; -1.0 = never logged yet

  // gamma_quench auto calib <warmup_steps> <period_steps> (requires
  // gamma_quench_auto): a closed-loop controller layered ON TOP of plain
  // "auto" rather than replacing it -- see fix_baoab_tether.rst for the full
  // rationale (this exists specifically because a flat hardcoded multiplier
  // on om_analytic_max does not generalize across target temperatures or
  // systems). For the first gamma_quench_calib_warmup steps, behaves exactly
  // like plain "auto" while the first measurement window fills. After
  // warm-up, every gamma_quench_calib_period steps, compares the masked
  // subsystem's own measured equipartition temperature against t_target and
  // nudges gamma_quench by a clamped, confirmed, log-space step -- see
  // update_dt(). om_analytic_max remains a permanent floor (never damp
  // weaker than critical damping for the current stiffest classified mode).
  // 0 (default) = disabled, gamma_quench_auto's existing behavior is
  // completely unchanged.
  int gamma_quench_calib;
  int gamma_quench_calib_warmup;   // steps before the controller starts acting (no default, required)
  int gamma_quench_calib_period;   // steps per measurement window (no default, required)

  // control-loop stability knobs (NOT physics constants -- see the design
  // doc comment on update_dt()'s GAMMA_CALIB block): govern how cautiously
  // the loop reacts to its own measurement noise, independent of what the
  // "correct" gamma_quench actually is for this run.
  double gamma_quench_calib_gain;        // exponent applied to rho=T_masked/t_target per commit; default 0.5
  double gamma_quench_calib_max_step;    // multiplicative clamp per commit; default 2.0
  int gamma_quench_calib_confirm;        // consecutive agreeing windows required before committing; default 3
  double gamma_quench_calib_deadband_lo; // rho below this counts as "too cold" evidence; default 0.8
  double gamma_quench_calib_deadband_hi; // rho above this counts as "too hot" evidence; default 1.25
  double gamma_quench_calib_ceiling;     // absolute safety rail, not a target value; default 1e4
  int gamma_quench_calib_refine;         // 1 (default) = refine forever, 0 = freeze after first commit

  // rolling-window accumulators (reset to zero at every window close --
  // deliberately NOT cumulative-forever like ke_rel's Phase B, so the
  // controller tracks a non-stationary run instead of averaging over its
  // entire history) and streak state for the confirm-before-commit gate.
  double gamma_quench_calib_sum_ke;   // sum of masked-mode-instance KE (eV) this window, ke_hot steps excluded
  double gamma_quench_calib_sum_dof;  // count of masked-mode-instance samples this window
  int gamma_quench_calib_direction;   // -1/0/+1 = last window's evidence direction; 0 = inside deadband
  int gamma_quench_calib_streak;      // consecutive windows agreeing with gamma_quench_calib_direction
  bigint last_gamma_calib_check;      // ntimestep of the last window close, for warm-up gating

  // set permanently true the first time the control law commits an update
  // (never cleared): with gamma_quench_calib_refine == 0 (freeze), this is
  // what actually stops further window evaluation after that first commit
  // -- gamma_quench_calib itself stays 1 forever so the warm-up guard on
  // the plain "auto" block in update_dt() never re-opens and overwrites the
  // frozen value.
  int gamma_quench_calib_committed;

  // ke_rel warm-up calibration (rare-event guard threshold, NOT an ordinary
  // dynamical property like gamma_quench -- see fix_baoab_tether.rst for why
  // these use different calibration strategies). 0 (default) = disabled,
  // ke_rel keeps its literal/default value for the whole run, unchanged.
  // >0 = number of steps in Phase A (threshold-finding); Phase B (exceedance
  // accumulation) starts when Phase A closes and runs for the rest of the
  // run in both policies. See kinetic_guard_and_quench() and update_dt().
  int ke_rel_warmup_steps;
  int ke_rel_refine;           // 0 = freeze after warm-up (default), 1 = keep ratcheting up
  int ke_rel_margin_blocks;    // K in the exponential-tail extrapolation formula, default 1000
  static constexpr int KE_REL_NBINS = 256;
  double ke_rel_hist[KE_REL_NBINS];  // Phase A histogram of r = ke_max/kT, range [0, ke_rel_hist_max]
  static constexpr double KE_REL_HIST_MAX = 60.0;
  double ke_rel_u;             // Phase A threshold (empirical 90th-pct bin); -1.0 = not yet set
  double ke_rel_count_u;       // Phase B: running count of samples with r > ke_rel_u
  double ke_rel_sum_exceed;    // Phase B: running sum of (r - ke_rel_u) over those samples
  double ke_rel_n_total;       // Phase B: running count of all samples since Phase A closed
  double ke_rel_last_logged;   // last ke_rel value written to event_log; -1.0 = never logged yet

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

  // dimensionless alternative to refresh_skin (mutually exclusive, default
  // 0.0 = disabled): multiplies each atom's OWN thermal ballistic step at
  // dt_max, sqrt(kT/m_i)*dt_max, instead of using one absolute distance for
  // every species/mass. Deliberately NOT curvature(om)-based: this guard's
  // own job is to catch atoms with persistently weak/soft curvature, where
  // an om-based amplitude diverges and disables the guard it should be
  // (confirmed by direct test). See effective_skin().
  double refresh_skin_rel;

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

  // 0 (default) = legacy behavior, fully unchanged: an exiled atom
  // (demote_cool[i] > ntimestep) takes the ordinary shared per-mode drift
  // at the shared GLOBAL update->dt, and its cached curvature still feeds
  // update_dt()'s om_unres vote and the PRESHRINK drift-skin trigger --
  // meaning one chronically volatile atom can hold the WHOLE system's dt
  // down even though it has already been relegated to the soft integrator.
  // 1 = while exiled, route the atom through sub_step_free_atom()'s
  // already-safe, Allreduce-free, locally resonance-bounded
  // (dt_sub = MIN(dt_min, 2.2/om_max_new)) sub-stepping for the FULL step
  // instead (see initial_integrate()/final_integrate()), and exclude it
  // from both the om_unres vote and the PRESHRINK trigger while exiled --
  // it no longer needs, and should not be allowed to shrink, the shared
  // dt. Does not change when/how long exile itself triggers or decays
  // (demote_cool/demote_count/backoff/forgiveness untouched).
  int exile_substep;

  // 0 (default) = end_of_step()'s PRESHRINK drift-skin trigger shrinks the
  // shared GLOBAL update->dt (and locks it via dt_floor_until) exactly as
  // before, regardless of exile_substep.
  // 1 = requires exile_substep yes (checked at init()). Each atom whose
  // drift trips PRESHRINK's threshold is instead demote_atom()'d on the
  // spot (same flat 3*refresh_every backoff the kinetic (ke_hot) guard
  // uses, source="preshrink") and excluded from that step's global-shrink
  // vote entirely, so it decouples into exile_substep's already-validated
  // sub_step_free_atom() treatment instead of forcing every other atom in
  // the system down to this one atom's locally-required dt. Untested
  // against this specific (predictive, pre-close-encounter) trigger
  // before shipping -- see fix_baoab_tether.rst for the retether
  // precedent this needs to avoid reproducing.
  int preshrink_exile;

  // permanent per-atom/per-event diagnostic log (plan Sec 6.1's required
  // per-step/per-event log): empty (default) = disabled, zero overhead --
  // every log_event() call site is gated on a plain "if (event_log_fp)"
  // check via SafeFilePtr's implicit nullptr conversion. When set, each
  // MPI rank opens its OWN file (comm->me-suffixed when nprocs > 1, so
  // this stays correct/halo-local under future domain decomposition with
  // no gather/collective) and logs REFRESH/GUARD_KE/GUARD_DRIFT/DEMOTE/
  // PRESHRINK/SETTLE_RELEASE events as they occur; see log_event().
  std::string event_log_filename;
  SafeFilePtr event_log_fp;

  double om_split;    // spectral-gap-detected stiff/soft split, one-shot (adapt==1 only)
  int om_split_set;   // 0 until compute_om_split() has run once
  int dt_lvl;          // current rung on the dt_max/2^dt_lvl ladder, -1 = uninitialized

  // DIAGNOSTIC-ONLY, deliberate match to lj_longstep_prototype.py's
  // HPdEngine.hessian_blocks(): when heavy_group is set, every atom in that
  // LAMMPS group gets a single, isotropic, ONE-TIME-SAMPLED curvature
  // estimate instead of a real per-refresh FD-Hessian probe -- sample
  // heavy_sample atoms from the group once (compute_heavy_block()), average
  // their raw trace/3, and freeze that single scalar (mass-divided per atom
  // at use time, see refresh_one_atom()) for the rest of the run. This is a
  // KNOWN NON-GENERAL SHORTCUT, not validated reference physics -- see the
  // python_heavy_block_frozen_approximation memory. It exists only to get
  // C++ and Python behaviorally consistent while the real fix (making the
  // stiff-mode machinery robust to a genuinely drifting heavy atom) is
  // worked out separately. Default (heavy_group unset, heavy_groupbit == 0)
  // leaves every atom on the real, continuously refreshed per-atom probe,
  // matching the fix's general design -- not a production keyword.
  int heavy_groupbit;      // 0 = disabled (default: real probe for every atom)
  int heavy_sample;        // atoms sampled to build the frozen estimate (default 8, matches Python)
  double heavy_block_tr3;  // averaged raw trace/3 (NOT mass-divided) of the sampled blocks
  int heavy_block_set;     // 0 until compute_heavy_block() has run once

  // DIAGNOSTIC-ONLY, deliberate match to lj_longstep_prototype.py's
  // HPdEngine.hessian_rows() (lammps_hpd_phase0.py:167-168), which returns
  // an all-zeros array for every call -- commented "# J-free (validated)" --
  // so FrozenHessianIntegrator.refresh()'s self.J (the dense linear-response
  // back-reaction matrix, D.10.6) is identically zero throughout every H/Pd
  // run: force_moll()'s `g += J^T g_s` back-reaction term never contributes
  // anything on the Python side. C++'s force_moll() computes and applies a
  // REAL (if sparse/halo-local-truncated) back-reaction via Jblk/Jtag built
  // from genuine FD cross-derivatives -- a real, structural difference
  // sitting directly in the H->Pd energy-transfer pathway, and a candidate
  // mechanism for the heavyT gap under investigation. j_free=1 skips
  // building Hc/Jblk/Jtag entirely in refresh_one_atom() (nJ[i] left at 0),
  // matching Python's J=0 behavior exactly while leaving the always-active
  // stiff-modal-row zeroing in force_moll() untouched (Python does that
  // unconditionally too, independent of J). Default off (real sparse
  // back-reaction, the fix's general/intended design) -- not a production
  // keyword; see python_heavy_block_frozen_approximation memory.
  int j_free;

  // THROWAWAY DIAGNOSTIC, not a production keyword: when set (>=0), print a
  // verbose per-iteration Newton-solve trace to stderr for exactly this one
  // atom tag, every time solve_center() runs on it. Built to find why a
  // handful of chronic atoms repeatedly hit the Newton-divergence guard at
  // T300 even with heff_samples-smoothed curvature -- see
  // test_ladder_isolates_newton_divergence_at_t300 memory. -1 = off
  // (default, zero overhead).
  tagint newton_debug_tag;

  // THROWAWAY DIAGNOSTIC, not a production keyword: when set (>=0), every
  // time partial_force() is called for this one atom tag with
  // force_eval_full OFF (local_partial_force()), ALSO compute
  // full_system_force() for the identical (i, xtrial) and print both
  // fout's and the relative difference to stderr. Does not change which
  // result is actually used -- pure side-channel comparison, built to
  // localize the force_eval local-vs-full gap found in
  // force_eval_local_regression_confirmed. -1 = off (default, zero
  // overhead).
  tagint pforce_debug_tag;

  // D.14 self-distilled surrogate hierarchy (design doc), RESPA-style
  // teacher/student composition -- EXPERIMENTAL, off by default
  // (teacher_every == 0 reproduces today's single-fidelity behavior byte-
  // for-byte; see fix_baoab_tether.rst). The production pair style
  // (force->pair, set via the ordinary pair_style/pair_coeff commands) is
  // the cheap STUDENT: every inner step's mollifier machinery (Hessian
  // blocks, Newton solve, modal mask, back-reaction J) runs on it exactly
  // as today. teacher_pair is a SECOND, independently-constructed Pair
  // object (never registered as force->pair) standing in for the
  // expensive TEACHER (e.g. a foundation MLIP) -- called only once every
  // teacher_every steps, via a whole-system teacher_pair->compute(0,0) at
  // the student's own clamped configuration, mirroring force_moll()'s
  // existing move-to-center/recompute/restore pattern. See
  // d14_respa_mechanism_port memory and lj_multifidelity_experiment.py
  // (the validated Python reference this ports) for the full design.
  class Pair *teacher_pair;     // owned; nullptr unless teacher_every > 0
  std::string teacher_style_name;
  std::vector<std::string> teacher_settings_args;   // from "teacher_style <n> <name> <args...>"
  std::vector<std::string> teacher_coeff_args;      // from "teacher_coeff <n> <args...>"
  int teacher_every;            // 0 (default) = disabled; the D.14 outer ratio "m"
  bigint outer_cycle_start;     // ntimestep the current outer cycle began
  // dt-limiter diagnostics (set by update_dt(), exposed via compute_vector 12-15): the
  // highest still-unresolved (unmasked) mode frequency that sets dt_c = c_acc/om_unres, the
  // LAMMPS type of the atom that owns it, the 99th-percentile unmasked frequency on this rank
  // (max vs p99 separates one extreme atom from a whole population), and the highest masked
  // (analytic) frequency. All in the run's own inverse-time unit (1/fs in real units).
  int recoil;    // reduced-mass (partner recoil) correction to analytic mode frequencies
  // molblock_diag (DIAGNOSTIC, default off): assemble each molecule's full mass-weighted Hessian
  // (3N_mol x 3N_mol) from the own/cross blocks refresh_one_atom() already measures, diagonalize it
  // and log internal vs rigid-body mode frequencies. Changes no dynamics. Step 2 of the
  // molecule-block plan; serial / whole-molecule-on-one-rank only.
  int molblock_diag;
  // kspace_center (EXPERIMENTAL, default off): add each atom's k-space force, evaluated once per
  // solve pass at the current configuration and held frozen, to the local force used by
  // solve_center(), so the Newton center sees the same total force whose masked component
  // force_moll() later removes.
  int kspace_center;
  std::vector<double> fks;
  void kspace_snapshot();
  bool mb_collect;
  // probe data per atom (owned atoms measured, ghosts filled by forward comm at a refresh)
  std::vector<double> mb_Hown;    // nall x 9: raw symmetrized own block H_ii
  std::vector<double> mb_X;       // nall x MBMAX x 9: dF_partner/dx_i (row = partner force comp)
  std::vector<tagint> mb_tag;     // nall x MBMAX: partner tags
  std::vector<int> mb_n;          // nall: number of partners
  std::vector<tagint> mb_btop;    // nall x mb_bpa: bond partner tags (own bond list)
  std::vector<tagint> mb_atop;    // nall x 3 mb_apa: angle atom tags (own angle list)
  std::vector<int> mb_nbt, mb_nat;    // nall: number of bonds / angles listed
  void molblock_diagnostic();

  // molblock (auto-RESPA, see mb_refresh()): Hessian-built fast force field for the bonds and
  // angles of small stiff molecules, integrated with inner substeps
  static constexpr int MBA = 5;           // max atoms per molecule (atom + 4 bond/angle partners)
  static constexpr int MBN = 3 * MBA;     // max Cartesian dimension of a molecule
  struct MBMol {
    int n = 0;                            // atoms in the molecule
    int nm = 0;                           // fast modes (= nic, or 0 if the molecule is not stiff)
    bool valid = false;                   // all atoms local this step
    tagint tag[MBA];
    double m[MBA];
    double xref[MBA][3];                  // geometry (rel. to COM) at refresh
    double w[MBN];                        // GF angular frequencies of the fast set (1/time)
    double Dref[MBN][MBN];                // mass-weighted Hessian at refresh
    int nic = 0;                          // fast internal coordinates (bonds, angles)
    // 0 = bond (a,b), 1 = angle (a,b,c) with vertex b, 2..7 = linear bend (a,b,c): component
    // (type-2)%2 of b's displacement from line a-c, frame from lab axis (type-2)/2 (mb_lin_frame)
    int ic_type[MBN], ic_at[MBN][3];
    double Fic[MBN][MBN];                 // internal-coordinate force constants
    double sig0[MBN];                     // reference internal coordinates (anchored once)
    bool have_s0 = false;
    double Fold[MBN][MBN];                // force constants still in use until the next step
    bool pending = false;                 // Fold valid, switch to Fic at the next step start
    // trajectory estimate of F: sums of ds ds^T and dphi ds^T over consecutive steps since the
    // last update (phi = raw internal-coordinate force), and the previous step's s and phi
    int nd = 0;
    bool hprev = false;
    double sprev[MBN], pprev[MBN], dss[MBN][MBN], dps[MBN][MBN];
    bool has_mom = false;                 // averaged moments (finite-difference F as prior)
    double Sss[MBN][MBN], Sps[MBN][MBN];
  };
  int molblock, mb_verbose;
  int mb_inner_user, mb_nin;              // user substeps (0 = auto), current substeps
  double mb_wmax;                         // fastest refreshed frequency (1/time)
  void mb_fast_force(const MBMol &, const double (*)[3], double (*)[3], bool = false) const;
  void mb_ic_eval(const MBMol &, const double (*)[3], double *, double (*)[MBN]) const;
  int mb_stiff_modes(const MBMol &, const double (*)[3]) const;
  void mb_gf_forceconst(MBMol &, const double (*)[3], const double (*)[3]);
  void mb_gf_freqs(MBMol &, const double (*)[3]);
  void mb_ic_force(const MBMol &, const double (*)[3], const int *, double *, double *) const;
  void mb_reestimate();
  void mb_lin_sym(MBMol &, double (*)[MBN]) const;
  void mb_lin_reframe(MBMol &, const double (*)[3]) const;
  bigint mb_last_refresh;
  bigint mb_step;                         // step whose positions/forces mb_prepare() processes
  // parallel layout: every rank integrates its own copy of each molecule with an owned atom
  // (ghosts for the rest), so the inner loop needs no communication. A molecule's persistent
  // state lives on its anchor (lowest-ID) atom, travels with it (exchange) and to ghost copies
  // (borders); between reneighborings each copy evolves it identically.
  std::vector<MBMol> mbmols;              // this rank's molecule copies (rebuilt each call)
  std::vector<int> mb_aidx;               // local or ghost index of each copy's anchor atom
  std::vector<int> mb_member;
  int mb_nicmax, mb_ss, mb_bpa, mb_apa, mb_commflag, mb_probe_size, mb_nmax;
  double **mb_state;                      // per atom: [0] = 1 if anchor holding state, then MBMol
  tagint *mb_anc;                         // per atom: anchor tag of its molecule (0 = none)
  bool mb_locate(const MBMol &, int *, double (*)[3]) const;
  void mb_rel(const MBMol &, const double (*)[3], double (*)[3]) const;
  int mb_pack(const MBMol &, double *) const;
  void mb_unpack(MBMol &, const double *) const;
  void mb_gather();
  void mb_scatter();
  double mb_gauss(tagint, int) const;
  void mb_refresh();
  void mb_prepare();
  void mb_initial_integrate();
  void mb_final_integrate();
  double diag_om_unres, diag_om_unres_p99, diag_om_analytic;
  int diag_unres_type;
  bigint n_teacher_calls;       // diagnostic: teacher_pair->compute() calls (compute_vector())
  // wall-clock cost of the teacher (compute_vector() 16/17, summary in post_run()):
  // whole compute_force_delta() calls, and the teacher_pair->compute() part of them
  // (including the KOKKOS host<->device transfers it triggers). Cumulative over runs;
  // the *_run0 copies are the values at the start of the current run (setup()).
  double t_teacher_total, t_teacher_pair;
  double t_run_start, t_teacher_total_run0, t_teacher_pair_run0;
  bigint n_teacher_calls_run0;

  // carried half-kick impulse per flagged atom (nmax x 3): 0.5*teacher_every*dt*
  // F_delta/m, already modal-filtered + back-reaction-applied by
  // apply_modal_filter_and_backreaction(). Reused, leapfrog-style, as both
  // the closing half-kick of the outer cycle that just computed it and the
  // opening half-kick of the next one -- see initial_integrate()/
  // final_integrate(). Harmless all-zero default when teacher_every == 0.
  double **f_delta;

  // MACE's much larger r_cut (vs. EAM/LJ) puts a bulk Pd atom's full
  // neighbor list around ~150 within 8 A; 96 was sized for the shorter
  // pairwise cutoffs this fix was originally validated against. Bumped
  // again for flexible SPC/Fw water: lj/cut/coul/long's 9 A cutoff + 2 A
  // skin gives an 11 A full-list radius, and liquid water's atom density
  // (~0.1 atom/Angstrom^3, 3x a comparable metal's atom count per volume)
  // puts a bulk atom's full neighbor list around ~550-600 -- confirmed by
  // a direct JMAX-exceeded error at 256 on this project's own water
  // workload. Same "raise and recompile" precedent as the EAM->MACE bump;
  // not expected to need per-system tuning beyond this going forward.
  static constexpr int JMAX = 1024;    // per-atom response-block neighbor cap

  // Bonded (bond/angle) local-force support: a stiff bond's partner is
  // almost always EXCLUDED from the pairwise neighbor list by special_bonds
  // (e.g. this project's water workload uses the LAMMPS default
  // special_bonds 0/0/0, fully excluding 1-2 pairs from pairwise LJ/Coulomb)
  // -- so it is invisible to both partial_force()'s pairwise kernel and
  // refresh_one_atom()'s pairwise-jlist-sourced back-reaction unless handled
  // separately. See partial_force_add_bonded()/refresh_one_atom(). 8 is
  // generous for flexible SPC/Fw water (max 2 bonds/atom) and any small-
  // molecule system this fix is likely to see next; raise if a future
  // system needs more bonds per atom than this.
  static constexpr int BOND_JMAX = 8;

  bigint last_refresh_step;
  bigint last_promote_check;   // ntimestep of the last promote_every recheck, -1 = none yet
  int need_refresh;

  // set by end_of_step()'s emergency dt-shrink (drift_response == preshrink
  // only) to ntimestep + a short backoff; guards update_dt()'s own relax-up
  // hysteresis so a scheduled refresh_blocks() call can't raise dt_lvl back
  // down before the atom that justified the shrink has actually cleared --
  // update_dt()'s om_unres is blind to which atom that was. -1 = no active
  // floor (default; never blocks relax-up).
  bigint dt_floor_until;

  // running count of consecutive qualifying update_dt() calls toward
  // settle_confirm (see above). Reset to 0 by end_of_step()'s drift-skin
  // loop finding ANY candidate atom this step (even when dt_lvl does not
  // end up changing, e.g. already pinned at the dt_min rung) and by
  // update_dt() itself whenever a scheduled call's margin check fails or
  // no lockout is currently active. Meaningless/inert when
  // settle_confirm == 0.
  int settle_streak;

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
  // only): Hacc is a continuously-updated exponential moving average (EMA)
  // of the raw (pre-diagonalization) 3x3 mass-normalized symmetrized
  // Hessian, row-major 9 doubles/atom; heff_samples sets the EMA time
  // constant (alpha = 1/heff_samples). om[i]/evecs[i] are re-diagonalized
  // from Hacc EVERY scheduled sample (never frozen) -- see
  // refresh_one_atom() -- so the working mask decision stays continuously
  // current with the atom's recent-history curvature instead of being
  // locked to a single stale snapshot.
  double **Hacc;            // running Hessian EMA (nmax x 9)
  int *n_hess_samples;      // samples seen so far, saturates at heff_samples (nmax); 0 means "never sampled"

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
  // eligibility bonded (opt-in, adapt=yes): per-atom 1 if atom has >=1 bond (set from
  // atom->num_bond at each refresh_blocks()), forward-communicated with om[] so ghost
  // neighbors can be tested too. Only meaningful for NON-REACTIVE topology (fixed
  // bonds); see eligibility_bonded below.
  int *bonded;
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

  // Stage-20 read-only diagnostic (long-run light-subsystem heating
  // investigation): cached-vs-fresh mode-eligibility disagreement, measured
  // by diag_check_staleness() immediately after mode_is_stiff() decides the
  // cached mask and before kinetic_guard_and_quench() can mutate it. Pure
  // counters, exposed via compute_vector() alongside the existing event
  // counters above; writes nothing to om[]/evecs[]/c[]/x0[]/Jtag/Jblk.
  // Per-atom-per-step counts of MASKED-MODE COUNT agreement (0..3 masked
  // modes cached vs. freshly re-probed), not per-mode-index agreement --
  // see diag_check_staleness()'s doc comment for why per-index comparison
  // is invalid for H-in-Pd's near-degenerate modes.
  bigint n_mask_agree;
  bigint n_mask_stale_soft;  // fresh count > cached count (dangerous: fewer modes tethered than should be)
  bigint n_mask_stale_tight; // fresh count < cached count (harmless: more modes tethered than needed)

  // halo-locality cost accounting (plan Sec 6.1 honest force-call cost):
  // partial_force() calls (Newton inner loop, mollify=yes) vs. full local
  // recompute_forces_local() calls -- exposed via compute_vector().
  bigint n_partial_force_calls, n_recompute_calls;

  void refresh_blocks();
  void recoil_accumulate(int, const double *, double, double *) const;
  void refresh_one_atom(int i, const double *xi, bool instantaneous);
  double effective_skin(int i, double mi, double kT) const;
  void force_clear_local();
  void recompute_forces_local();
  void modal_force_correction(int i, double mi, double *fcorr) const;

  // stage-2 additions
  void solve_center(int i, double mi, double kT, double *cnew);
  bool neighbor_clamp_eligible(int j) const;
  bool trial_in_range(int i, const double *xtrial) const;
  bool partial_force(int i, const double *xtrial, double *fout, double *fneigh = nullptr,
                      bool live_neighbor_positions = false, tagint *bond_tag = nullptr,
                      double *bond_fneigh = nullptr, int *bond_jnum_out = nullptr);
  bool full_system_force(int i, const double *xtrial, double *fout, double *fneigh = nullptr,
                          int jnum = 0, const int *jlist = nullptr);
  void partial_force_add_bonded(int i, const double *xtrial, double *fout,
                                 bool live_neighbor_positions, tagint *bond_tag,
                                 double *bond_fneigh, int *bond_jnum_out) const;
  void force_moll();
  void apply_modal_filter_and_backreaction();

  // D.14 additions
  void teacher_outer_kick();
  void compute_force_delta();

  // stage-3 additions

  inline bool mode_is_stiff(int i, int m) const
  {
    // om[] is expressed in the natural eV/Angstrom/amu time unit (tau ~= 10.1805
    // fs), not real LAMMPS metal-units ps like update->dt -- sqrt(force->ftm2v)
    // converts it to real ps^-1 before this comparison. Only needed where om is
    // used linearly against a real-time quantity; the squared uses elsewhere
    // (mi*om^2 reconstructing H) are scale-invariant and must not get this factor.
    if (!adapt) return (om[i][m] * sqrt(force->ftm2v) * update->dt) > theta;
    if (demote_cool[i] > update->ntimestep) return false;
    double om_max = MAX(om[i][0], MAX(om[i][1], om[i][2]));
    double om_min = MIN(om[i][0], MIN(om[i][1], om[i][2]));
    if (!((om_max > om_split) && (bonded_ok(i) || (om_min > 0.6 * om_split)))) return false;
    return om[i][m] > om_split;
  }
  void compute_om_split();
  void compute_heavy_block();

  // Stage-20 read-only diagnostic: re-probes atom i's curvature at its
  // CURRENT position (FD Hessian + jacobi3, same math as refresh_one_atom())
  // without writing om[]/evecs[]/c[]/x0[]/Jtag/Jblk, re-applies the
  // mode_is_stiff() eligibility test against the fresh estimate, and
  // accumulates n_mask_agree/n_mask_stale_soft/n_mask_stale_tight against
  // msk_cached (the mask mode_is_stiff() just produced this step).
  void diag_check_staleness(int i, double mi, const bool *msk_cached);

  // fresh_sample: true (default) from refresh_blocks()'s own scheduled
  // cadence (every flagged atom's om[]/evecs[] just freshly reprobed
  // before this call); false only from end_of_step()'s promote_every
  // recheck, which reuses stale cached om[] -- settle_confirm's streak
  // only ever advances on the former.
  void update_dt(bool fresh_sample = true);
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
