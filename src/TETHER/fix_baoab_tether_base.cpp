/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_baoab_tether_base.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "domain.h"
#include "math_eigen.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "pair.h"
#include "update.h"

#include <cmath>
#include <cstring>
#include <memory>

using namespace LAMMPS_NS;

// FixBAOAB is constructed with a literal narg of 7 (it would reject the derived styles'
// keywords), so the required arguments are checked here first: without this check a too-short
// command line would make FixBAOAB read past the end of the argument array
int FixBAOABTetherBase::base_narg(LAMMPS *lmp, int narg, const char *style)
{
  if (narg < 7) utils::missing_cmd_args(FLERR, utils::lowercase(style), lmp->error);
  return 7;
}

FixBAOABTetherBase::FixBAOABTetherBase(LAMMPS *lmp, int narg, char **arg, const char *style) :
    FixBAOAB(lmp, base_narg(lmp, narg, style), arg), n_recompute_calls(0), style_name(style),
    list(nullptr), eps(0.0), eps_was_set(0), use_local_partial_force(0), force_eval_full(0),
    pforce_debug_tag(-1)
{
}

/* ---------------------------------------------------------------------- */

void FixBAOABTetherBase::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBAOABTetherBase::force_clear_local()
{
  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0], 0, 3 * nbytes);
}

/* ---------------------------------------------------------------------- */

void FixBAOABTetherBase::recompute_forces_local()
{
  n_recompute_calls++;
  force_clear_local();

  if (force->pair) force->pair->compute(0, 0);
  if (force->bond) force->bond->compute(0, 0);
  if (force->angle) force->angle->compute(0, 0);
  if (force->dihedral) force->dihedral->compute(0, 0);
  if (force->improper) force->improper->compute(0, 0);
  if (force->kspace) force->kspace->compute(0, 0);
  // force term that needs this configuration's kspace solution (charge response of an LES
  // model), exactly as the integrator adds it (Pair::compute_kspace_response())
  if (force->kspace && force->pair && force->pair->kspace_response_flag)
    force->pair->compute_kspace_response(0, 0);

  if (modify->n_pre_reverse) modify->pre_reverse(0, 0);
  if (force->newton) comm->reverse_comm();
}

/* ---------------------------------------------------------------------- */

bool FixBAOABTetherBase::charges_from_pair() const
{
  return atom->q_flag && force->pair && force->pair->kspace_response_flag;
}

void FixBAOABTetherBase::save_charges(std::vector<double> &qs) const
{
  if (!charges_from_pair()) return;
  qs.assign(atom->q, atom->q + atom->nlocal + atom->nghost);
}

void FixBAOABTetherBase::restore_charges(const std::vector<double> &qs)
{
  if (!charges_from_pair() || qs.empty()) return;
  int n = MIN((int) qs.size(), atom->nlocal + atom->nghost);
  memcpy(atom->q, qs.data(), n * sizeof(double));
  if (force->kspace) force->kspace->qsum_qsq();
}

/* ---------------------------------------------------------------------- */

void FixBAOABTetherBase::save_accumulators(double *e) const
{
  for (int k = 0; k < NACCUM; k++) e[k] = 0.0;
  e[0] = force->pair->eng_vdwl;
  e[1] = force->pair->eng_coul;
  for (int k = 0; k < 6; k++) e[2 + k] = force->pair->virial[k];
  if (force->kspace) {
    e[8] = force->kspace->energy;
    for (int k = 0; k < 6; k++) e[9 + k] = force->kspace->virial[k];
  }
}

void FixBAOABTetherBase::restore_accumulators(const double *e)
{
  force->pair->eng_vdwl = e[0];
  force->pair->eng_coul = e[1];
  for (int k = 0; k < 6; k++) force->pair->virial[k] = e[2 + k];
  if (force->kspace) {
    force->kspace->energy = e[8];
    for (int k = 0; k < 6; k++) force->kspace->virial[k] = e[9 + k];
  }
}

/* ----------------------------------------------------------------------
   checks and kernel choice for partial_force() (pair style, atom map, symmetric bond/angle
   storage); called from the init() of styles that probe or solve with it
------------------------------------------------------------------------- */

void FixBAOABTetherBase::init_probe_kernels()
{
  if (!force->pair) error->all(FLERR, "{} requires a pair style", style_name);
  // the per-atom force kernels (local_partial_force()/single()) read the
  // host-side per-atom state of the pair style (e.g. PairEAM's rho[]/fp[]),
  // which KOKKOS pair styles keep on the device instead; only
  // symmetrix/mace/kk implements its own device-aware kernel. Under -sf kk,
  // the student pair style must therefore be defined after "suffix off";
  // the teacher_style may be a KOKKOS style (see compute_force_delta()).
  if (force->pair->kokkosable && !utils::strmatch(force->pair_style, "^symmetrix/mace"))
    error->all(FLERR,
               "{} cannot use the KOKKOS pair style {} for its per-atom forces; "
               "define the pair style after a 'suffix off' command (a KOKKOS teacher_style "
               "is supported)",
               style_name, force->pair_style);
  use_local_partial_force = force->pair->has_local_partial_force;
  // force_eval full evaluates every trial force by a whole-system recompute
  // (full_system_force()), so it needs neither per-atom kernel
  if (!force_eval_full && !use_local_partial_force && !force->pair->single_enable)
    error->all(FLERR, "{} requires a pair style that supports "
                       "either single() or local_partial_force()", style_name);
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "{} requires atom_modify map (yes|array)", style_name);

  // partial_force_add_bonded() reads each atom's own bond/angle lists, which hold every bond
  // and angle the atom takes part in only with newton_bond off
  if (force->bond && force->newton_bond)
    error->all(FLERR, "{} with a bond style requires newton_bond off (needed for symmetric "
               "per-atom bond storage)", style_name);
  if (force->angle && force->newton_bond)
    error->all(FLERR, "{} with an angle style requires newton_bond off (needed for symmetric "
               "per-atom angle storage)", style_name);
}

/* ---------------------------------------------------------------------- */

bool FixBAOABTetherBase::partial_force(int i, const double *xtrial, double *fout, double *fneigh,
                                    bool live_neighbor_positions, tagint *bond_tag,
                                    double *bond_fneigh, int *bond_jnum_out)
{
  fout[0] = fout[1] = fout[2] = 0.0;
  if (bond_jnum_out) *bond_jnum_out = 0;
  if (!list) return false;

  double **x = atom->x;
  int *type = atom->type;
  int itype = type[i];
  double **cutsq = force->pair->cutsq;

  int *jlist = list->firstneigh[i];
  int jnum = list->numneigh[i];

  // force_eval full: every caller of partial_force() (not just
  // solve_center()'s two sites) now gets a genuine whole-system force
  // evaluation instead of the local kernel below -- see full_system_force()'s
  // doc comment and force_eval_full's doc comment in the header. Note this
  // silently drops the live_neighbor_positions c[j]-substitution behavior
  // (full_system_force() always uses real positions, a documented scope
  // limitation of this validation mode). fout is already bond/angle-complete
  // here (full_system_force() -> recompute_forces_local() calls force->bond/
  // force->angle unconditionally) but bond_tag/bond_fneigh are left empty --
  // no per-partner breakdown is available from a whole-system recompute, so
  // the bonded back-reaction is silently absent under force_eval_full, the
  // same documented scope limitation as the c[j] substitution above.
  if (force_eval_full) return full_system_force(i, xtrial, fout, fneigh, jnum, jlist);

  if (use_local_partial_force) {
    // Mirror the clamped-center substitution used by the single()-based
    // fallback below (real x[j] -> c[j] for a CURRENTLY stiff/rattler-
    // eligible atom, when !live_neighbor_positions): local_partial_force()
    // applies this itself, internally, via the clamp_mask/clamp_c pair,
    // indexed by atom index (not `mask[j] & groupbit` -- see
    // neighbor_clamp_eligible()'s doc comment for why that would be wrong
    // on a single whole-system group) so a multi-hop implementation (e.g.
    // PairSymmetrixMACE) can apply it to any atom it encounters, not just
    // atom i's direct jlist neighbors. Sized nlocal+nghost and built fresh
    // each call -- a real but modest O(N) cost on top of
    // local_partial_force()'s own, only paid for Newton-residual calls.
    // c[] is kept forward-comm'd/PBC-shifted to match x[]'s image (see
    // pack_forward_comm()/unpack_forward_comm()), so it is safe to hand to
    // the pair style for ghost indices too.
    // std::vector<bool> is bit-packed and has no .data(); use a plain
    // heap buffer so this can be handed to the Pair interface as bool*.
    std::unique_ptr<bool[]> clamp_mask;
    if (!live_neighbor_positions) {
      int nall = atom->nlocal + atom->nghost;
      clamp_mask = std::make_unique<bool[]>(nall);
      for (int k = 0; k < nall; k++) clamp_mask[k] = clamp_neighbor(k);
    }
    const bool *clamp_mask_ptr = clamp_mask.get();
    double *const *clamp_c = live_neighbor_positions ? nullptr : clamp_centers();
    bool ok = force->pair->local_partial_force(i, xtrial, jnum, jlist, fout, fneigh,
                                                clamp_mask_ptr, clamp_c);

    // THROWAWAY DIAGNOSTIC (pforce_debug_tag), see header comment: side-
    // channel comparison against full_system_force() for the identical
    // (i, xtrial), without altering which result (fout above) is actually
    // used by the caller. IMPORTANT: full_system_force() recomputes and
    // leaves the pair style's GLOBAL rho[]/fp[] caches in the TRIAL
    // configuration's state -- it only restores f[] (see its own doc
    // comment/implementation), not rho[]/fp[]. That is harmless under
    // normal force_eval_full operation (every call there is a fresh
    // full_system_force() call too, so nothing ever reads a stale
    // in-between cache), but calling it here, as a side channel while
    // use_local_partial_force is the ACTIVE path, would otherwise corrupt
    // the very rho[j] cache local_partial_force()'s delta-correction
    // depends on for every subsequent call this step -- silently
    // contaminating the "local" results this diagnostic is supposed to be
    // comparing against, and compounding across repeated probe/Newton
    // calls. Undo that corruption immediately with one extra no-op
    // full_system_force() call (xtrial == the real x[i], captured before
    // the comparison call moved it) before returning -- 2x the diagnostic's
    // own O(N) cost, never incurred unless pforce_debug_tag is set.
    if (pforce_debug_tag >= 0 && atom->tag[i] == pforce_debug_tag) {
      double xreal_now[3] = {x[i][0], x[i][1], x[i][2]};
      double fout_full[3];
      full_system_force(i, xtrial, fout_full, nullptr, 0, nullptr);
      // bonded terms are added to fout only AFTER this block (below), so compare the
      // pair-only local force PLUS the bonded kernel against the complete full force
      double fout_tot[3] = {fout[0], fout[1], fout[2]};
      partial_force_add_bonded(i, xtrial, fout_tot, live_neighbor_positions, nullptr, nullptr, nullptr);
      double dmag = sqrt((fout_tot[0]-fout_full[0])*(fout_tot[0]-fout_full[0]) +
                          (fout_tot[1]-fout_full[1])*(fout_tot[1]-fout_full[1]) +
                          (fout_tot[2]-fout_full[2])*(fout_tot[2]-fout_full[2]));
      double fmag = sqrt(fout_full[0]*fout_full[0] + fout_full[1]*fout_full[1] +
                          fout_full[2]*fout_full[2]);
      double disp = sqrt((xtrial[0]-xreal_now[0])*(xtrial[0]-xreal_now[0]) +
                          (xtrial[1]-xreal_now[1])*(xtrial[1]-xreal_now[1]) +
                          (xtrial[2]-xreal_now[2])*(xtrial[2]-xreal_now[2]));
      // TEMPORARY forensic check (user's hypothesis): is any neighbor BOTH
      // currently clamp-eligible AND within force cutoff of xtrial? If so,
      // local correctly clamps it (c[j]) while full_system_force() always
      // uses its real x[j] (documented scope limitation, not a bug) -- the
      // two are EXPECTED to disagree there, not a sign of a local defect.
      int n_eligible_in_range = 0;
      double min_eligible_r = -1.0;
      if (!live_neighbor_positions) {
        double **cutsq = force->pair->cutsq;
        int *type = atom->type;
        int itype = type[i];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj] & NEIGHMASK;
          if (!clamp_neighbor(j)) continue;
          double delx = xtrial[0]-x[j][0], dely = xtrial[1]-x[j][1], delz = xtrial[2]-x[j][2];
          double rsq = delx*delx+dely*dely+delz*delz;
          if (rsq < cutsq[itype][type[j]]) {
            n_eligible_in_range++;
            double r = sqrt(rsq);
            if (min_eligible_r < 0.0 || r < min_eligible_r) min_eligible_r = r;
          }
        }
      }

      utils::logmesg(lmp, "PFORCE_DEBUG step={} tag={} live={} jnum={} disp={:.6g} "
                     "n_elig_in_range={} min_elig_r={:.6g} "
                     "fout_local=({:.6g},{:.6g},{:.6g}) fout_full=({:.6g},{:.6g},{:.6g}) "
                     "|delta|={:.6g} |full|={:.6g} rel={:.6g}\n",
                     update->ntimestep, pforce_debug_tag, live_neighbor_positions, jnum, disp,
                     n_eligible_in_range, min_eligible_r,
                     fout_tot[0], fout_tot[1], fout_tot[2], fout_full[0], fout_full[1], fout_full[2],
                     dmag, fmag, fmag > 0.0 ? dmag / fmag : 0.0);

      double fout_restore_dummy[3];
      full_system_force(i, xreal_now, fout_restore_dummy, nullptr, 0, nullptr);
    }
    partial_force_add_bonded(i, xtrial, fout, live_neighbor_positions, bond_tag, bond_fneigh,
                              bond_jnum_out);
    return ok;
  }

  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj] & NEIGHMASK;
    int jtype = type[j];

    // A neighbor that is CURRENTLY stiff/rattler-eligible must contribute
    // via its own current CENTER, not its instantaneous real position --
    // otherwise atom i's clamped equilibrium responds directly to atom j's
    // fast real-time oscillation phase, a resonant parametric-coupling
    // channel between adjacent rattlers. This mirrors
    // lj_longstep_prototype.py's _solve_centers(), which places ALL
    // flagged atoms at their (jointly converging) trial centers before
    // every force evaluation, never at their real xw positions; it is the
    // analogous fix for the Newton solve itself, complementing (not
    // duplicating) the rattler-rattler Jblk exclusion in refresh_blocks(),
    // which only concerns the back-reaction operator. See
    // neighbor_clamp_eligible()'s doc comment for why this is an om-based
    // eligibility test, not `mask[j] & groupbit`.
    const double *xj = (!live_neighbor_positions && clamp_neighbor(j)) ? clamp_centers()[j] : x[j];

    double delx = xtrial[0] - xj[0];
    double dely = xtrial[1] - xj[1];
    double delz = xtrial[2] - xj[2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq >= cutsq[itype][jtype]) {
      if (fneigh) fneigh[3 * jj] = fneigh[3 * jj + 1] = fneigh[3 * jj + 2] = 0.0;
      continue;
    }

    // Honor the special-bond bits the neighbor list carries on jlist[jj]: with a kspace style
    // LAMMPS keeps excluded (bonded 1-2/1-3/1-4) pairs in the list so the pair style can apply
    // its excluded-pair correction (factor 0 => subtract the reciprocal-space part). Hard-coding
    // factors of 1.0 here evaluated those pairs as full Coulomb, a spurious ~100 kcal/mol/A
    // force on a water O-H pair in the Newton solve and FD curvature probe (found 2026-11-05;
    // force_eval full, which calls the real pair->compute(), was stable). For pair styles
    // without special neighbors the bits are 0 => factors 1.0, identical to before.
    int sb = (jlist[jj] >> SBBITS) & 3;
    double fpair = 0.0;
    force->pair->single(i, j, itype, jtype, rsq, force->special_coul[sb], force->special_lj[sb],
                        fpair);
    fout[0] += delx * fpair;
    fout[1] += dely * fpair;
    fout[2] += delz * fpair;
    if (fneigh) {
      fneigh[3 * jj + 0] = -delx * fpair;
      fneigh[3 * jj + 1] = -dely * fpair;
      fneigh[3 * jj + 2] = -delz * fpair;
    }
  }
  partial_force_add_bonded(i, xtrial, fout, live_neighbor_positions, bond_tag, bond_fneigh,
                            bond_jnum_out);
  return true;
}

/* ----------------------------------------------------------------------
   Bonded (bond/angle-style) contribution to partial_force()'s per-atom
   local force kernel: a stiff bond's partner (e.g. water's O-H) is
   typically EXCLUDED from the pairwise neighbor list by special_bonds, so
   it is invisible to the pairwise logic above -- yet it is often the
   dominant stiff mode partial_force()/solve_center()'s Newton solve needs
   to see (design doc: O-H stretch ~3700 cm^-1, the stiffest mode in
   water). The H-O-H bend (~1600 cm^-1) is softer but NOT negligible for
   the Newton solve specifically: omitting it leaves solve_center()
   searching for a force balance against an incomplete force law (bond
   only), while the real per-step force (recompute_forces_local(), via
   force_moll()) is bond+angle complete -- confirmed empirically on this
   project's water workload: bond-only produced a large, persistent
   Newton-guard rate (n_guard_newton/n_guard_flip climbing continuously),
   which is the expected symptom of chasing a root that doesn't exist for
   the (incomplete) force law being solved. A no-op (fout unchanged, no
   slots written) whenever force->bond AND force->angle are both unset, so
   this has zero effect on the existing EAM/MACE pair-only systems.

   Uses the SAME tag->local-index resolution LAMMPS's own topology-list
   builders use (ntopo_bond_all.cpp: atom->map() then
   domain->closest_image()) rather than neighbor->bondlist/anglelist,
   which are deduplicated, non-atom-indexed flat lists -- the wrong tool
   for "given atom i, find its bonded partners now". This requires every
   bond/angle to be stored symmetrically on ALL its member atoms, i.e.
   newton_bond off (checked in init(); see water_bonded_kspace_scoping
   memory).

   Applies the identical clamped-center substitution as the pairwise loop
   above (neighbor_clamp_eligible()) so a bonded partner that is itself
   currently flagged stiff doesn't inject its own fast oscillation phase
   into atom i's Newton solve -- the same resonant-coupling concern that
   substitution exists to suppress for ordinary pairwise neighbors.

   Angle support is AngleHarmonic-only (Angle::single_force() returns
   false, silently skipped, for any other angle style) -- a known, scoped
   limitation, not a general angle-style framework.

   bond_tag/bond_fneigh/bond_jnum_out are optional (nullptr skips writing
   them, matching fneigh's existing convention): fout always gets the
   bond/angle contribution added regardless, but only refresh_one_atom()
   (which needs each partner's own force response for the Jtag/Jblk
   back-reaction) passes non-null buffers; solve_center() just wants the
   correct net fout.
------------------------------------------------------------------------- */

void FixBAOABTetherBase::partial_force_add_bonded(int i, const double *xtrial, double *fout,
                                               bool live_neighbor_positions, tagint *bond_tag,
                                               double *bond_fneigh, int *bond_jnum_out) const
{
  if (!force->bond && !force->angle) return;

  double **x = atom->x;
  int nk = 0;

  auto clamped = [&](int j) -> const double * {
    return (!live_neighbor_positions && clamp_neighbor(j)) ? clamp_centers()[j] : x[j];
  };

  if (force->bond) {
    int nb = atom->num_bond[i];
    for (int k = 0; k < nb; k++) {
      if (nk >= BOND_JMAX)
        error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded local-force "
                           "partner slots -- raise BOND_JMAX and recompile", BOND_JMAX);
      int j = atom->map(atom->bond_atom[i][k]);
      if (j < 0) continue;    // lost bond partner -- fails safe, same convention as n_ghost_miss
      j = domain->closest_image(i, j);

      const double *xj = clamped(j);

      double delx = xtrial[0] - xj[0];
      double dely = xtrial[1] - xj[1];
      double delz = xtrial[2] - xj[2];
      double rsq = delx * delx + dely * dely + delz * delz;

      double fforce = 0.0;
      force->bond->single(atom->bond_type[i][k], rsq, i, j, fforce);
      fout[0] += delx * fforce;
      fout[1] += dely * fforce;
      fout[2] += delz * fforce;

      if (bond_tag && bond_fneigh) {
        bond_tag[nk] = atom->tag[j];
        bond_fneigh[3 * nk + 0] = -delx * fforce;
        bond_fneigh[3 * nk + 1] = -dely * fforce;
        bond_fneigh[3 * nk + 2] = -delz * fforce;
        nk++;
      }
    }
  }

  if (force->angle) {
    int na = atom->num_angle[i];
    tagint my_tag = atom->tag[i];
    for (int k = 0; k < na; k++) {
      tagint t1 = atom->angle_atom1[i][k];
      tagint t2 = atom->angle_atom2[i][k];
      tagint t3 = atom->angle_atom3[i][k];
      int role = (t1 == my_tag) ? 1 : (t2 == my_tag) ? 2 : 3;

      int j1 = (role == 1) ? i : atom->map(t1);
      int j2 = (role == 2) ? i : atom->map(t2);
      int j3 = (role == 3) ? i : atom->map(t3);
      if (j1 < 0 || j2 < 0 || j3 < 0) continue;
      if (role != 1) j1 = domain->closest_image(i, j1);
      if (role != 2) j2 = domain->closest_image(i, j2);
      if (role != 3) j3 = domain->closest_image(i, j3);

      const double *x1 = (role == 1) ? xtrial : clamped(j1);
      const double *x2 = (role == 2) ? xtrial : clamped(j2);
      const double *x3 = (role == 3) ? xtrial : clamped(j3);

      double f1[3], f3[3];
      if (!force->angle->single_force(atom->angle_type[i][k], x1, x2, x3, f1, f3)) continue;
      double f2[3] = {-(f1[0] + f3[0]), -(f1[1] + f3[1]), -(f1[2] + f3[2])};

      const double *fi = (role == 1) ? f1 : (role == 2) ? f2 : f3;
      fout[0] += fi[0];
      fout[1] += fi[1];
      fout[2] += fi[2];

      if (bond_tag && bond_fneigh) {
        if (role != 1) {
          if (nk >= BOND_JMAX)
            error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded "
                               "local-force partner slots -- raise BOND_JMAX and recompile",
                       BOND_JMAX);
          bond_tag[nk] = atom->tag[j1];
          bond_fneigh[3 * nk + 0] = f1[0];
          bond_fneigh[3 * nk + 1] = f1[1];
          bond_fneigh[3 * nk + 2] = f1[2];
          nk++;
        }
        if (role != 2) {
          if (nk >= BOND_JMAX)
            error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded "
                               "local-force partner slots -- raise BOND_JMAX and recompile",
                       BOND_JMAX);
          bond_tag[nk] = atom->tag[j2];
          bond_fneigh[3 * nk + 0] = f2[0];
          bond_fneigh[3 * nk + 1] = f2[1];
          bond_fneigh[3 * nk + 2] = f2[2];
          nk++;
        }
        if (role != 3) {
          if (nk >= BOND_JMAX)
            error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded "
                               "local-force partner slots -- raise BOND_JMAX and recompile",
                       BOND_JMAX);
          bond_tag[nk] = atom->tag[j3];
          bond_fneigh[3 * nk + 0] = f3[0];
          bond_fneigh[3 * nk + 1] = f3[1];
          bond_fneigh[3 * nk + 2] = f3[2];
          nk++;
        }
      }
    }
  }
  if (bond_jnum_out) *bond_jnum_out = nk;
}

/* ----------------------------------------------------------------------
   force_eval full validation mode: an exact whole-system force on atom i
   (and, when fneigh/jnum/jlist are supplied, on each of its listed
   neighbors too) if atom i were placed at xtrial. partial_force() routes
   EVERY one of its callers here -- solve_center()'s Newton residual,
   refresh_blocks()/refresh_one_atom()'s FD-Hessian curvature probe (both
   the atom-i diagonal block and the Hc_j/Jblk cross-derivative columns),
   sub_step_free_atom()'s exile sub-stepping -- whenever force_eval_full is set (see
   partial_force()'s own dispatch at its top). Unlike partial_force(),
   this does NOT substitute a flagged neighbor's clamped center c[j] for
   its real position -- every other atom (rattler or not) is evaluated at
   its actual current x[j]. That is a deliberate, known scope limitation,
   not an oversight: it reintroduces the resonant rattler-rattler coupling
   partial_force()'s c[j] substitution exists to suppress (see that
   comment above), so this mode is only a clean validation of the
   embedding-truncation hypothesis for configurations with at most one
   flagged atom in range of any other (e.g. the dilute single-H-
   interstitial H/Pd runs it was built to test) -- not a general
   substitute for the local kernel.

   Single-process only (guarded in the constructor): moving xtrial into
   atom->x[i] and recomputing requires atom i's own ghost images to be
   refreshed via comm->forward_comm() for correctness under periodic
   boundaries, but no cross-rank communication of the trial position
   itself is attempted here. Reuses recompute_forces_local() (the same
   whole-system pair/bond/.../kspace recompute force_moll() already uses
   for its own clamped-configuration force field) rather than calling
   force->pair->compute() directly, so every force contribution (not just
   the pair style) is included -- the same completeness Python's
   lj_longstep_prototype.py/lammps_hpd_phase0.py get "for free" from
   calling the real engine's full energy_forces() for every FD probe.
   O(N) per call -- expected and acceptable for a validation mode, not
   something to wire into a hot path (see force_eval_full's doc comment
   in the header for which call sites this makes very slow).

   LATENT CONSTRAINT: only f[] (and atom i's position) are saved/restored
   around the trial recompute -- the pair style's own GLOBAL rho[]/fp[]
   caches are deliberately left in the TRIAL configuration's state, not
   restored. Harmless under force_eval_full's actual (sole, exclusive) use
   -- every partial_force() call routes here when it's set, so every call
   is its own fresh, self-consistent full recompute and nothing ever reads
   a stale in-between rho[]/fp[] value. But this function must NEVER be
   called as a side comparison while use_local_partial_force's path is the
   one actually in effect without separately restoring rho[]/fp[] first
   (e.g. a second full_system_force() call at the real, unperturbed
   configuration) -- local_partial_force()'s rho_j_trial delta-correction
   reads rho[j] directly, and would otherwise silently use a corrupted
   baseline for every subsequent call this step. pforce_debug_tag's
   comparison code does this restore explicitly; see its doc comment.
------------------------------------------------------------------------- */

bool FixBAOABTetherBase::full_system_force(int i, const double *xtrial, double *fout,
                                        double *fneigh, int jnum, const int *jlist)
{
  double **x = atom->x;
  double **f = atom->f;
  int ntotal = atom->nlocal + atom->nghost;

  std::vector<double> fsave(3 * ntotal);
  if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));
  std::vector<double> qsave;
  save_charges(qsave);

  double xreal[3] = {x[i][0], x[i][1], x[i][2]};
  x[i][0] = xtrial[0];
  x[i][1] = xtrial[1];
  x[i][2] = xtrial[2];

  comm->forward_comm();
  recompute_forces_local();

  fout[0] = f[i][0];
  fout[1] = f[i][1];
  fout[2] = f[i][2];

  // read back each listed neighbor's own total force at this trial
  // configuration -- the whole-system analog of partial_force()'s fneigh
  // output, needed by refresh_one_atom()'s Hc_j/Jblk cross-derivative
  // columns. f[j] here already reflects every force contribution on j
  // (not just the i-j pair), which is the correct quantity for a
  // genuine dF_j/dx_i finite difference -- recompute_forces_local() was
  // just run for the whole local domain, not atom i alone.
  if (fneigh)
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      fneigh[3 * jj + 0] = f[j][0];
      fneigh[3 * jj + 1] = f[j][1];
      fneigh[3 * jj + 2] = f[j][2];
    }

  x[i][0] = xreal[0];
  x[i][1] = xreal[1];
  x[i][2] = xreal[2];
  comm->forward_comm();

  if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));
  restore_charges(qsave);

  return true;
}

/* ----------------------------------------------------------------------
   central-difference curvature probe of atom i about xi (step eps), forces from
   partial_force() with live neighbor positions:
     H[a][b] = -dF_i,a / dx_i,b (not symmetrized);
   with_neighbors (and a neighbor list): the columns Hc[k*9 + 3a + b] = dF_j,a / dx_i,b for every
   atom j = jlist[k] of atom i's list (jnum of them); with_bonds: the same for the bond/angle
   partners (partial_force_add_bonded()), n_bond of them with IDs bond_tag_buf, in Hc_bond.
   Extracted from fix baoab/tether's refresh_one_atom(); shared with fix baoab/molblock.
------------------------------------------------------------------------- */

void FixBAOABTetherBase::probe_atom(int i, const double *xi, bool with_neighbors, bool with_bonds,
                                    double (*H)[3], int &jnum, int *&jlist, std::vector<double> &Hc,
                                    std::vector<tagint> &bond_tag_buf, std::vector<double> &Hc_bond,
                                    int &n_bond)
{
  jnum = 0;
  jlist = nullptr;
  if (with_neighbors && list) {
    jlist = list->firstneigh[i];
    jnum = list->numneigh[i];
    if (jnum > JMAX)
      error->one(FLERR, "Fix baoab/tether: atom exceeds JMAX ({}) neighbors for the "
                         "sparse response block -- raise JMAX and recompile", JMAX);
  }
  // neighbor force-response columns: Hc[k][a][b] = dF_{jlist[k],a}/dx_{i,b}
  Hc.clear();
  if (jnum > 0) Hc.assign((size_t) jnum * 9, 0.0);

  std::vector<double> fj_plus, fj_minus;
  if (jnum > 0) {
    fj_plus.resize((size_t) jnum * 3);
    fj_minus.resize((size_t) jnum * 3);
  }

  // Bonded-partner response columns (see partial_force_add_bonded()): a
  // stiff bond's partner is typically excluded from the pairwise jlist
  // above by special_bonds, so it needs its own, separate Hc-style buffer
  // -- n_bond is the same every FD evaluation (atom i's own bond topology
  // doesn't change within one refresh_one_atom() call), so the last
  // (b==2) probe's count is authoritative. No-op allocation (BOND_JMAX is
  // a small fixed cap) when force->bond is unset.
  std::vector<double> bj_plus, bj_minus;
  n_bond = 0;
  bond_tag_buf.clear();
  Hc_bond.clear();
  if (with_bonds && force->bond) {
    bond_tag_buf.resize(BOND_JMAX);
    bj_plus.resize((size_t) BOND_JMAX * 3);
    bj_minus.resize((size_t) BOND_JMAX * 3);
    Hc_bond.assign((size_t) BOND_JMAX * 9, 0.0);
  }

  for (int b = 0; b < 3; b++) {
    double fplus[3], fminus[3];
    double xt[3] = {xi[0], xi[1], xi[2]};
    int n_bond_plus = 0, n_bond_minus = 0;
    bool have_bond = with_bonds && force->bond;

    xt[b] = xi[b] + eps;
    partial_force(i, xt, fplus, jnum > 0 ? fj_plus.data() : nullptr, true,
                  have_bond ? bond_tag_buf.data() : nullptr, have_bond ? bj_plus.data() : nullptr,
                  have_bond ? &n_bond_plus : nullptr);

    xt[b] = xi[b] - eps;
    partial_force(i, xt, fminus, jnum > 0 ? fj_minus.data() : nullptr, true,
                  have_bond ? bond_tag_buf.data() : nullptr, have_bond ? bj_minus.data() : nullptr,
                  have_bond ? &n_bond_minus : nullptr);

    for (int a = 0; a < 3; a++) H[a][b] = -(fplus[a] - fminus[a]) / (2.0 * eps);

    for (int k = 0; k < jnum; k++)
      for (int a = 0; a < 3; a++)
        Hc[(size_t) k * 9 + 3 * a + b] =
            (fj_plus[(size_t) k * 3 + a] - fj_minus[(size_t) k * 3 + a]) / (2.0 * eps);

    // n_bond_plus == n_bond_minus by construction (atom i's bond topology
    // is fixed within this call); take either, last iteration wins.
    n_bond = n_bond_plus;
    for (int k = 0; k < n_bond; k++)
      for (int a = 0; a < 3; a++)
        Hc_bond[(size_t) k * 9 + 3 * a + b] =
            (bj_plus[(size_t) k * 3 + a] - bj_minus[(size_t) k * 3 + a]) / (2.0 * eps);
  }
}

/* ----------------------------------------------------------------------
   molecule partners of atom i from one probe_atom() result: Hown = the symmetrized own block,
   then up to MOLPART partners = bond/angle partners (their bonded columns, X[p*9 + ...] =
   dF_partner/dx_i), plus any pairwise column for the same atom ID (intramolecular non-bonded
   part, e.g. the excluded-pair Coulomb correction). np = number of partners.
------------------------------------------------------------------------- */

void FixBAOABTetherBase::collect_partners(const double (*H)[3], int jnum, const int *jlist,
                                          const std::vector<double> &Hc, int n_bond,
                                          const std::vector<tagint> &bond_tag_buf,
                                          const std::vector<double> &Hc_bond, double *Hown,
                                          double *X, tagint *T, int &np) const
{
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) Hown[3 * a + b] = 0.5 * (H[a][b] + H[b][a]);
  np = 0;
  for (int s = 0; s < MOLPART * 9; s++) X[s] = 0.0;
  // molecule partners = bonded/angle partners (their bonded columns), plus any pairwise column
  // for the same tag (intramolecular non-bonded part, e.g. the excluded-pair Coulomb correction).
  for (int k = 0; k < n_bond; k++) {
    tagint t = bond_tag_buf[k];
    int p = -1;
    for (int q = 0; q < np; q++)
      if (T[q] == t) p = q;
    if (p < 0) {
      if (np >= MOLPART) continue;
      T[np] = t;
      p = np++;
    }
    for (int s = 0; s < 9; s++) X[p * 9 + s] += Hc_bond[(size_t) k * 9 + s];
  }
  for (int k = 0; k < jnum; k++) {
    tagint t = atom->tag[jlist[k] & NEIGHMASK];
    int p = -1;
    for (int q = 0; q < np; q++)
      if (T[q] == t) p = q;
    if (p < 0) continue;
    for (int s = 0; s < 9; s++) X[p * 9 + s] += Hc[(size_t) k * 9 + s];
  }
}

/* ----------------------------------------------------------------------
   second, standalone pair style
------------------------------------------------------------------------- */

int FixBAOABTetherBase::parse_counted_args(int narg, char **arg, int iarg,
                                           std::vector<std::string> &out)
{
  std::string kw = fmt::format("{} {}", utils::lowercase(style_name), arg[iarg]);
  if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, kw, error);
  int n = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
  if (n < 1) error->all(FLERR, iarg + 1, "{} {} count must be >= 1", style_name, arg[iarg]);
  if (iarg + 2 + n > narg) utils::missing_cmd_args(FLERR, kw, error);
  out.clear();
  for (int k = 0; k < n; k++) out.emplace_back(arg[iarg + 2 + k]);
  return iarg + 2 + n;
}

Pair *FixBAOABTetherBase::create_aux_pair(const std::string &name,
                                          const std::vector<std::string> &settings,
                                          const std::vector<std::string> &coeff, const char *role)
{
  // the pair style takes the active suffix like any pair_style command
  int sflag;
  Pair *p = force->new_pair(name, lmp->suffix_enable, sflag);
  if (!p) error->all(FLERR, "{}: unknown {} pair style {}", style_name, role, name);

  std::vector<char *> settings_argv;
  for (auto &s : settings) settings_argv.push_back(const_cast<char *>(s.c_str()));
  p->settings((int) settings_argv.size(), settings_argv.data());

  // some pair styles (eam/alloy, eam/fs, ...) set the per-type masses in coeff(): those of the
  // system (set by the input, possibly different from the potential file's) must stay in effect
  int ntypes = atom->ntypes;
  std::vector<double> msave;
  std::vector<int> msetsave;
  if (atom->mass) {
    msave.assign(atom->mass, atom->mass + ntypes + 1);
    msetsave.assign(atom->mass_setflag, atom->mass_setflag + ntypes + 1);
  }
  std::vector<char *> coeff_argv;
  for (auto &s : coeff) coeff_argv.push_back(const_cast<char *>(s.c_str()));
  p->coeff((int) coeff_argv.size(), coeff_argv.data());
  if (atom->mass) {
    for (int t = 1; t <= ntypes; t++)
      if (msetsave[t] && (atom->mass[t] != msave[t])) {
        if (comm->me == 0)
          utils::logmesg(lmp, "{} {} pair style: keeping the mass {} of atom type {} (the "
                         "potential file's is {})\n", style_name, role, msave[t], t,
                         atom->mass[t]);
        atom->mass[t] = msave[t];
      }
    for (int t = 1; t <= ntypes; t++) atom->mass_setflag[t] = msetsave[t] | atom->mass_setflag[t];
  }

  if (comm->me == 0) {
    std::string pname = name;
    if (sflag) pname += fmt::format("/{}", (sflag == 1) ? lmp->suffix : lmp->suffix2);
    utils::logmesg(lmp, "{} {} pair style: {} ({})\n", style_name, role, pname,
                   p->kokkosable ? "KOKKOS" : "not KOKKOS, runs on the host");
  }
  return p;
}

void FixBAOABTetherBase::aux_pair_force(Pair *p, std::vector<double> &out)
{
  double **f = atom->f;
  int nlocal = atom->nlocal;
  int ntotal = nlocal + atom->nghost;
  std::vector<double> fsave(3 * ntotal);
  if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));
  force_clear_local();
  p->compute(0, 0);
  if (force->newton_pair) comm->reverse_comm();
  out.resize(3 * nlocal);
  if (nlocal) memcpy(out.data(), &f[0][0], 3 * nlocal * sizeof(double));
  if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));
}
