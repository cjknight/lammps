/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_baoab_tether.h"

#include "angle.h"
#include "atom.h"
#include "bond.h"
#include "comm.h"
#include "dihedral.h"
#include "domain.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "improper.h"
#include "integrate.h"
#include "kspace.h"
#include "math_eigen.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

namespace {

// Validate the shared FixBAOAB positional arguments (t_start t_stop damp
// seed at arg[3..6]) before FixBAOAB's own constructor runs, since it is
// invoked below with a literal narg of 7 regardless of the caller's real
// narg -- without this check, a too-short command line would make FixBAOAB
// read past the end of the caller's arg array.
int check_base_args(LAMMPS *lmp, int narg)
{
  if (narg < 7) utils::missing_cmd_args(FLERR, "fix baoab/tether", lmp->error);
  return 7;
}

// small fixed 3x3 helpers used by the stage-2 mollifier (response block S_i,
// cross-Hessian probes, back-reaction matvecs) -- kept local rather than
// pulled from a generic linear-algebra header since nothing else in this
// fix needs anything larger than 3x3.
inline void mat3_matvec(const double m[9], const double v[3], double out[3])
{
  out[0] = m[0] * v[0] + m[1] * v[1] + m[2] * v[2];
  out[1] = m[3] * v[0] + m[4] * v[1] + m[5] * v[2];
  out[2] = m[6] * v[0] + m[7] * v[1] + m[8] * v[2];
}

// out = a * b (row-major 3x3 times row-major 3x3)
inline void mat3_matmat(const double a[9], const double b[9], double out[9])
{
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 3; c++) {
      double s = 0.0;
      for (int k = 0; k < 3; k++) s += a[3 * r + k] * b[3 * k + c];
      out[3 * r + c] = s;
    }
}

}    // namespace

/* ---------------------------------------------------------------------- */

FixBAOABTether::FixBAOABTether(LAMMPS *lmp, int narg, char **arg) :
    FixBAOAB(lmp, check_base_args(lmp, narg), arg), list(nullptr), x0(nullptr), c(nullptr),
    evecs(nullptr), om(nullptr), nJ(nullptr), Jtag(nullptr), Jblk(nullptr), demote_count(nullptr),
    last_demote(nullptr), demote_cool(nullptr), quench_exempt(nullptr), last_ke(nullptr)
{
  theta = -1.0;
  refresh_every = -1;
  eps = 0.0;
  eps_was_set = 0;

  mollify = 0;
  newton_iters = 8;
  newton_damp = 0.6;
  solve_tol_rel = 0.005;

  adapt = 0;
  c_acc = 0.25;
  dt_min = -1.0;    // sentinel: default 0.02*dt, resolved once dt is known below
  dt_max = -1.0;    // sentinel: default the fix-creation-time dt, resolved below
  ke_rel = 12.0;
  gamma_quench = 20.0;
  om_split = 0.0;
  om_split_set = 0;
  dt_lvl = -1;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "theta") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether theta", error);
      theta = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "refresh") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether refresh", error);
      refresh_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "eps") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether eps", error);
      eps = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      eps_was_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "mollify") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether mollify", error);
      mollify = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "newton_iters") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether newton_iters", error);
      newton_iters = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "newton_damp") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether newton_damp", error);
      newton_damp = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "solve_tol_rel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether solve_tol_rel", error);
      solve_tol_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "adapt") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether adapt", error);
      adapt = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "c_acc") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether c_acc", error);
      c_acc = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "dt_min") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether dt_min", error);
      dt_min = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "dt_max") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether dt_max", error);
      dt_max = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "ke_rel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether ke_rel", error);
      ke_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench", error);
      gamma_quench = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown fix baoab/tether keyword {}", arg[iarg]);
    }
  }

  if (theta <= 0.0) error->all(FLERR, "Fix baoab/tether theta must be > 0");
  if (refresh_every <= 0) error->all(FLERR, "Fix baoab/tether refresh must be > 0");
  if (eps_was_set && eps <= 0.0) error->all(FLERR, "Fix baoab/tether eps must be > 0");
  if (newton_iters < 1) error->all(FLERR, "Fix baoab/tether newton_iters must be >= 1");
  if (newton_damp <= 0.0 || newton_damp > 1.0)
    error->all(FLERR, "Fix baoab/tether newton_damp must be in (0,1]");
  if (solve_tol_rel <= 0.0) error->all(FLERR, "Fix baoab/tether solve_tol_rel must be > 0");

  if (adapt) {
    // dt_max defaults to the fix-creation-time timestep (the ceiling the
    // ladder never exceeds); dt_min defaults to a small fraction of it --
    // both mirror the Python reference's own dt_max=dt_big/dt_min=0.02*dt_big.
    if (dt_max < 0.0) dt_max = update->dt;
    if (dt_min < 0.0) dt_min = 0.02 * dt_max;
    if (c_acc <= 0.0) error->all(FLERR, "Fix baoab/tether c_acc must be > 0");
    if (dt_min <= 0.0) error->all(FLERR, "Fix baoab/tether dt_min must be > 0");
    if (dt_max < dt_min) error->all(FLERR, "Fix baoab/tether dt_max must be >= dt_min");
    if (ke_rel <= 0.0) error->all(FLERR, "Fix baoab/tether ke_rel must be > 0");
    if (gamma_quench <= 0.0) error->all(FLERR, "Fix baoab/tether gamma_quench must be > 0");
  }

  last_refresh_step = -1;
  need_refresh = 1;
  n_guard_newton = n_guard_flip = n_guard_energy = 0;
  n_demoted = n_kinetic_guard = 0;
  n_partial_force_calls = n_recompute_calls = 0;
  n_ghost_miss = 0;

  // stage-4: this fix owns a private per-atom array (c[]) that partial_force()
  // reads for possibly-ghost neighbors -- register it for standard forward
  // comm (pack/unpack below) so a ghost's copy is never stale/garbage under
  // real multi-rank. See initial_integrate()/final_integrate() for the call
  // site and force_moll() comments for the matching reverse-comm fix.
  comm_forward = 3;

  // stage-4: pack_exchange()/unpack_exchange() write a per-atom payload far
  // larger (and variable-size, up to the JMAX cap) than the base Fix class's
  // default maxexchange of 0 -- left unset, Comm::init_exchange() undercounts
  // this fix's contribution to bufextra and a heavily-blocked atom's
  // pack_exchange() can overrun comm's send buffer during a real cross-rank
  // migration (invisible on 1 rank, where exchange() is never exercised).
  // Static (not maxexchange_dynamic), sized from the fixed worst case: x0(3)
  // + c(3) + evecs(9) + om(3) + nJ-count(1) + JMAX*(tag(1)+block(9)) +
  // demote_count/last_demote/demote_cool/quench_exempt(4) + last_ke(1).
  maxexchange = 3 + 3 + 9 + 3 + 1 + JMAX * (1 + 9) + 4 + 1;

  // diagnostic vector (plan Sec 6.1 event forensics / honest cost report):
  // [0..2] self-evaluation guard counts (Newton non-convergence, center
  // flip, over-excited harmonic energy -- computed always when mollify=yes,
  // acted on -- actual demotion -- only when adapt=yes, see the per-atom
  // state comment in the header); [3] partial_force() calls (mollify=yes
  // Newton inner loop, halo-local); [4] recompute_forces_local() calls (one
  // full local force pass each) -- ratio of [3] to [4]*nlocal is a first
  // halo-locality sanity number ahead of MPI work; [5] total demotion
  // events (any cause, adapt=yes only); [6] demotion events specifically
  // from the kinetic over-excitation guard (subset of [5]); [7] stage-4
  // ghost-miss count (stored sparse-block neighbor drifted outside the
  // ghost cutoff before its back-reaction was applied, contribution
  // dropped -- see n_ghost_miss in the header).
  vector_flag = 1;
  size_vector = 8;
  extvector = 0;

  // per-atom persistent state (curvature-block eigenvectors/frequencies,
  // frozen/mollified center, sparse response blocks) -- exchanged/reordered
  // like any other atom-based array, but not restart-aware yet: a restart
  // re-refreshes on the first post-restart step rather than resuming
  // mid-cycle (stage-1/2 limitation).
  FixBAOABTether::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
}

/* ---------------------------------------------------------------------- */

FixBAOABTether::~FixBAOABTether()
{
  if (copymode) return;

  atom->delete_callback(id, Atom::GROW);

  memory->destroy(x0);
  memory->destroy(c);
  memory->destroy(evecs);
  memory->destroy(om);
  memory->destroy(nJ);
  memory->destroy(Jtag);
  memory->destroy(Jblk);
  memory->destroy(demote_count);
  memory->destroy(last_demote);
  memory->destroy(demote_cool);
  memory->destroy(quench_exempt);
  memory->destroy(last_ke);
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::init()
{
  FixBAOAB::init();

  // eps defaults to a small fraction of the neighbor skin (a system-
  // dependent length scale already known at this point) rather than a
  // hardcoded absolute distance; override with the "eps" keyword when the
  // system's own thermal amplitude calls for something tighter or looser.
  if (!eps_was_set) eps = 1.0e-4 * neighbor->skin;
  if (eps <= 0.0)
    error->all(FLERR, "Fix baoab/tether eps must be > 0 (neighbor->skin is 0 -- set eps explicitly)");

  if (mollify) {
    if (!force->pair) error->all(FLERR, "Fix baoab/tether mollify yes requires a pair style");
    if (!force->pair->single_enable)
      error->all(FLERR, "Fix baoab/tether mollify yes requires a pair style that supports single()");
    if (atom->map_style == Atom::MAP_NONE)
      error->all(FLERR, "Fix baoab/tether mollify yes requires atom_modify map (yes|array)");

    // private, non-occasional full neighbor list: the damped-Newton inner
    // loop (solve_center()/partial_force()) needs every one of a flagged
    // atom's own pair-style neighbors, every step -- not the half-list
    // "i<j" restriction, and not just occasionally (plan doc SS5.1).
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }

  // force an immediate curvature-block acquisition on the first
  // initial_integrate() call of this run
  need_refresh = 1;
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

// B-A-O-A-B, generalizing the "A" (drift) and "O" (thermostat) sub-steps
// for group atoms: per-mode harmonic drift about a center for locally
// stiff eigenmodes (omega*dt > theta), ordinary free drift and the
// inherited Cartesian-equivalent OU update for everything else. The B
// (kick) sub-steps and all non-group atoms are untouched -- identical to
// FixBAOAB::initial_integrate(). With "mollify yes", the center c[i] is
// the instantaneous Newton-solved clamped equilibrium (solve_center()) and
// atom->f is replaced by the mollified force field (force_moll()) before
// any per-atom math below reads it; with "mollify no", c[i] is the static
// frozen center set by the last refresh_blocks() call and the raw force is
// corrected in place by modal_force_correction(), exactly reproducing
// stage 1.

void FixBAOABTether::initial_integrate(int /*vflag*/)
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  compute_target();

  // stage-4: need_refresh can be forced to 1 on this rank alone (e.g. a
  // local atom's demote_atom() call in solve_center()/the kinetic guard --
  // adapt=yes only, and which atoms demote is inherently rank-local under
  // real domain decomposition). refresh_blocks() and, when adapt is on,
  // update_dt() both make their own collective MPI calls internally, so
  // every rank MUST enter refresh_blocks() on the same step or those
  // collectives desynchronize and hang. Allreduce the gating decision
  // itself so it is identical on every rank before anyone acts on it.
  {
    int local_refresh = (need_refresh || (update->ntimestep - last_refresh_step) >= refresh_every) ? 1 : 0;
    int global_refresh = local_refresh;
    if (comm->nprocs > 1) MPI_Allreduce(&local_refresh, &global_refresh, 1, MPI_INT, MPI_MAX, world);
    if (global_refresh) {
      refresh_blocks();
      last_refresh_step = update->ntimestep;
      need_refresh = 0;
    }
  }

  double kT = force->boltz * t_target / force->mvv2e;

  if (mollify) {
    // stage-4: refresh ghost copies of c[] from the previous step's
    // fully-converged values BEFORE any local atom's Newton solve below can
    // touch its own c[i] this step -- see pack_forward_comm() above.
    comm->forward_comm(this);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      double mi = rmass ? rmass[i] : mass[type[i]];
      solve_center(i, mi, kT, c[i]);
    }
    force_moll();
  }

  double one_minus_c1sq = 1.0 - c1 * c1;

  // strong-friction quench thermostat coefficients (adapt=yes only; harmless
  // to precompute unconditionally since kinetic_guard_and_quench() is only
  // ever called, and quench_active only ever comes back true, when adapt is
  // set -- see there). Full-step gamma_quench, same "O applies once, at full
  // dt" convention as c1/one_minus_c1sq above.
  double cq1 = exp(-gamma_quench * update->dt);
  double one_minus_cq1sq = 1.0 - cq1 * cq1;

  energy_onestep = 0.0;

  double fran_total[3] = {0.0, 0.0, 0.0};
  double mass_total = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double invmass = 1.0 / mi;

    // ---- B: half kick. mollify=no: masked modes get the harmonic-
    // restoring component of the raw force cancelled first (see
    // modal_force_correction()) so only the anharmonic residual is kicked
    // with -- the harmonic part is already integrated exactly by the
    // analytic drift below. mollify=yes: force_moll() has already replaced
    // f[] with the mollified field (masked component zeroed, J
    // back-reaction applied), so no further correction is needed here.
    double fcorr[3] = {0.0, 0.0, 0.0};
    if (!mollify) modal_force_correction(i, mi, fcorr);
    double dtfm = dtf * invmass;
    v[i][0] += dtfm * (f[i][0] + fcorr[0]);
    v[i][1] += dtfm * (f[i][1] + fcorr[1]);
    v[i][2] += dtfm * (f[i][2] + fcorr[2]);

    bool msk[3];
    for (int m = 0; m < 3; m++) msk[m] = mode_is_stiff(i, m);

    // project displacement-from-center and velocity into the frozen
    // eigenbasis; unmasked modes carry this round-trip through free
    // flight, so it is a no-op for them up to floating-point roundoff.
    // x[i] and c[i] are the SAME physical atom but can land in different
    // periodic images after domain->pbc() re-wraps x[i] at a reneighbor
    // step (c[i] is only ever set from x0[i]'s image at refresh time) --
    // minimum-image the difference or it can be off by a box length.
    double dxr[3] = {x[i][0] - c[i][0], x[i][1] - c[i][1], x[i][2] - c[i][2]};
    domain->minimum_image(FLERR, dxr);
    double q0[3] = {0.0, 0.0, 0.0}, p0[3] = {0.0, 0.0, 0.0};
    for (int m = 0; m < 3; m++)
      for (int k = 0; k < 3; k++) {
        double wmk = evecs[i][3 * m + k];
        q0[m] += wmk * dxr[k];
        p0[m] += wmk * v[i][k];
      }

    // ---- A: first half drift ----
    double q1[3], p1[3];
    for (int m = 0; m < 3; m++) {
      if (msk[m]) {
        double w = om[i][m];
        double cs = cos(w * dtby2), sn = sin(w * dtby2);
        q1[m] = cs * q0[m] + (sn / w) * p0[m];
        p1[m] = -w * sn * q0[m] + cs * p0[m];
      } else {
        q1[m] = q0[m] + dtby2 * p0[m];
        p1[m] = p0[m];
      }
    }

    // ---- kinetic over-excitation guard + quench (D.11.3, adapt=yes only):
    // reads the SAME post-first-half-drift p1[] the O sub-step below is
    // about to consume, exactly where lj_longstep_prototype.py's run()
    // inserts it between its first drift and its O sub-step. May demote
    // this atom (mutating msk[] in place, for the rest of THIS step as well
    // as persistently via demote_cool[]) and/or mark it for the strong-
    // friction quench kick applied in the O step below instead of the
    // ordinary analytic-mode OU.
    bool quench_active = false;
    if (adapt) kinetic_guard_and_quench(i, mi, kT, msk, p1, quench_active);

    // ---- O: exact OU update. Masked (stiff) modes get the ordinary
    // analytic-mode c1/c2 kick, same as FixBAOAB; a quench-active atom
    // (cooling down from a just-fired demotion, adapt=yes only) instead
    // gets the strong-friction cq1/cq2 kick on every component -- applying
    // it in this same eigenbasis is exactly equivalent to Cartesian
    // velocity space since the transform is orthogonal, so no separate
    // Cartesian pass is needed. Everything else (soft, non-cooling) is
    // unchanged, matching Stage 1/2. ----
    double c2 = sqrt(kT * invmass * one_minus_c1sq);
    double cq2 = quench_active ? sqrt(kT * invmass * one_minus_cq1sq) : 0.0;
    double r[3] = {random->gaussian(), random->gaussian(), random->gaussian()};
    double p2[3];
    double ke_before = 0.0, ke_after = 0.0;
    for (int m = 0; m < 3; m++) ke_before += p1[m] * p1[m];
    for (int m = 0; m < 3; m++) {
      if (msk[m]) p2[m] = c1 * p1[m] + c2 * r[m];
      else if (quench_active) p2[m] = cq1 * p1[m] + cq2 * r[m];
      else p2[m] = p1[m];
    }
    for (int m = 0; m < 3; m++) ke_after += p2[m] * p2[m];
    energy_onestep += 0.5 * force->mvv2e * mi * (ke_before - ke_after);

    if (zeroflag) {
      for (int k = 0; k < 3; k++) {
        double kick = 0.0;
        for (int m = 0; m < 3; m++) {
          if (msk[m]) kick += evecs[i][3 * m + k] * c2 * r[m];
          else if (quench_active) kick += evecs[i][3 * m + k] * cq2 * r[m];
        }
        fran_total[k] += mi * kick;
      }
      mass_total += mi;
    }

    // ---- A: second half drift ----
    double q3[3], p3[3];
    for (int m = 0; m < 3; m++) {
      if (msk[m]) {
        double w = om[i][m];
        double cs = cos(w * dtby2), sn = sin(w * dtby2);
        q3[m] = cs * q1[m] + (sn / w) * p2[m];
        p3[m] = -w * sn * q1[m] + cs * p2[m];
      } else {
        q3[m] = q1[m] + dtby2 * p2[m];
        p3[m] = p2[m];
      }
    }

    // back to Cartesian coordinates
    double dxnew[3] = {0.0, 0.0, 0.0}, vnew[3] = {0.0, 0.0, 0.0};
    for (int m = 0; m < 3; m++)
      for (int k = 0; k < 3; k++) {
        double wmk = evecs[i][3 * m + k];
        dxnew[k] += wmk * q3[m];
        vnew[k] += wmk * p3[m];
      }

    x[i][0] = c[i][0] + dxnew[0];
    x[i][1] = c[i][1] + dxnew[1];
    x[i][2] = c[i][2] + dxnew[2];
    v[i][0] = vnew[0];
    v[i][1] = vnew[1];
    v[i][2] = vnew[2];
  }

  // zero net random momentum across all MPI ranks (same optional
  // correction as FixBAOAB; the position back-correction below is exact
  // for free-drifting atoms/modes and only a small-order approximation
  // for masked/stiff modes, same as its treatment in FixBAOAB is exact
  // only for the plain Cartesian drift it assumes -- zeroflag defaults off)
  if (zeroflag) {
    double buf[4] = {fran_total[0], fran_total[1], fran_total[2], mass_total};
    double bufall[4];
    MPI_Allreduce(buf, bufall, 4, MPI_DOUBLE, MPI_SUM, world);

    if (bufall[3] > 0.0) {
      double inv_mtot = 1.0 / bufall[3];
      double vcorr[3] = {bufall[0] * inv_mtot, bufall[1] * inv_mtot, bufall[2] * inv_mtot};

      for (int i = 0; i < nlocal; i++) {
        if (!(mask[i] & groupbit)) continue;
        v[i][0] -= vcorr[0];
        v[i][1] -= vcorr[1];
        v[i][2] -= vcorr[2];
        x[i][0] -= dtby2 * vcorr[0];
        x[i][1] -= dtby2 * vcorr[1];
        x[i][2] -= dtby2 * vcorr[2];
      }
    }
  }

  energy += energy_onestep;
}

/* ----------------------------------------------------------------------
   closing half kick from the new forces. Overridden (rather than
   inherited from FixBAOAB) because it needs the same treatment as the
   opening half kick in initial_integrate(): mollify=no re-solves nothing
   (the center/curvature is unchanged across this step's substeps) and
   just re-applies modal_force_correction(); mollify=yes re-solves every
   flagged atom's center against the NEW cage (positions changed since
   initial_integrate()) and re-mollifies the freshly-computed f[] before
   this kick consumes it.
------------------------------------------------------------------------- */

void FixBAOABTether::final_integrate()
{
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  if (mollify) {
    double kT = force->boltz * t_target / force->mvv2e;
    // stage-4: see the matching comm->forward_comm(this) note in
    // initial_integrate() -- same reasoning applies here.
    comm->forward_comm(this);
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      double mi = rmass ? rmass[i] : mass[type[i]];
      solve_center(i, mi, kT, c[i]);
    }
    force_moll();
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double dtfm = dtf / mi;

    double fcorr[3] = {0.0, 0.0, 0.0};
    if (!mollify) modal_force_correction(i, mi, fcorr);
    v[i][0] += dtfm * (f[i][0] + fcorr[0]);
    v[i][1] += dtfm * (f[i][1] + fcorr[1]);
    v[i][2] += dtfm * (f[i][2] + fcorr[2]);
  }

}

/* ----------------------------------------------------------------------
   shared demotion bookkeeping (adapt=yes only): bumps the chronic-failure
   counter, records the step for forgiveness, and exiles this atom from
   analytic/Verlet treatment (mode_is_stiff() returns false for it, this
   step and every step until demote_cool) for backoff_steps steps. Callers
   (the self-evaluation guard in solve_center() and the kinetic guard
   below) compute backoff_steps themselves -- the Newton/flip/energy guard
   uses exponential backoff capped at 2000 steps, the kinetic guard uses a
   flat 3*refresh_every with no backoff -- matching the reference exactly.
------------------------------------------------------------------------- */

void FixBAOABTether::demote_atom(int i, bigint backoff_steps)
{
  bigint ntimestep = update->ntimestep;
  demote_count[i]++;
  last_demote[i] = ntimestep;
  demote_cool[i] = ntimestep + backoff_steps;
  n_demoted++;
  need_refresh = 1;
}

/* ----------------------------------------------------------------------
   kinetic over-excitation guard + quench (D.11.3, adapt=yes only): called
   from initial_integrate() between the first drift half-step and the O
   sub-step, exactly where lj_longstep_prototype.py's run() inserts the
   same check. p1[] is this atom's post-first-half-drift modal velocity
   (already computed by the caller); msk[] is this atom's current per-mode
   stiff mask, mutated in place on a trip so the O sub-step and second
   drift half-step immediately below both see the demotion.

   ke_max is the largest per-mode kinetic energy among this atom's
   currently-masked modes only (an atom with no masked modes -- soft, or
   already cooling down -- has ke_max 0 and never trips). "hot" compares it
   to a fixed multiple of kT; "sudden" additionally requires a large jump
   since the last step's value, distinguishing a genuine impulsive event
   (which should be quenched hard right away, hence quench_exempt) from a
   slow gradual heating (left to the ordinary, gentler quench treatment).

   Separately (regardless of whether this call's own guard trips), reports
   via quench_active whether this atom should receive the strong-friction
   quench kick in the O step: it must currently be exiled (demote_cool in
   the future), not exempt, and not demoted on THIS exact step -- an atom
   demoted this step already got its energy dealt with by the demotion
   itself (mask override) and should not also take a quench kick the same
   step it happened, mirroring the reference's exclusion of its currently-
   eligible atom set from the same-step quench pass.
------------------------------------------------------------------------- */

void FixBAOABTether::kinetic_guard_and_quench(int i, double mi, double kT, bool *msk,
                                               const double *p1, bool &quench_active)
{
  bigint ntimestep = update->ntimestep;

  double ke_max = 0.0;
  for (int m = 0; m < 3; m++) {
    if (!msk[m]) continue;
    double ke = 0.5 * force->mvv2e * mi * p1[m] * p1[m];
    if (ke > ke_max) ke_max = ke;
  }

  bool hot = ke_max > ke_rel * kT;
  bool sudden = (ke_max - last_ke[i]) > 0.5 * ke_rel * kT;
  last_ke[i] = ke_max;

  if (hot) {
    for (int m = 0; m < 3; m++) msk[m] = false;
    demote_atom(i, 3 * (bigint) refresh_every);
    n_kinetic_guard++;
    if (sudden) quench_exempt[i] = demote_cool[i];
  }

  quench_active = (demote_cool[i] > ntimestep) && (quench_exempt[i] <= ntimestep) &&
                  (last_demote[i] != ntimestep);
}

/* ----------------------------------------------------------------------
   Cartesian correction added to the raw force before any half kick on a
   group atom's masked (locally stiff) modes: cancels that mode's
   harmonic restoring-force component, m*omega^2*(displacement from the
   center), so the impulse carries only the anharmonic residual -- the
   harmonic part is already integrated exactly by the analytic drift.
   mollify=no only (mollify=yes gets the equivalent correction, plus the
   nonlinear J back-reaction, from force_moll() instead). Mirrors
   lj_longstep_prototype.py's FrozenHessianIntegrator._force_moll()
   mollify=False branch (g[s] += from_modal(om_eff^2 * dlt)), converted
   from that mass-weighted form to a plain Cartesian force by the factor
   of mi. Without this, a stiff mode's huge raw restoring force is
   double-counted between the kick and the drift and the kick blows up.
------------------------------------------------------------------------- */

void FixBAOABTether::modal_force_correction(int i, double mi, double *fcorr) const
{
  double **x = atom->x;
  // same periodic-image caveat as initial_integrate()'s dxr -- see there.
  double dxr[3] = {x[i][0] - c[i][0], x[i][1] - c[i][1], x[i][2] - c[i][2]};
  domain->minimum_image(FLERR, dxr);

  fcorr[0] = fcorr[1] = fcorr[2] = 0.0;
  for (int m = 0; m < 3; m++) {
    if (!mode_is_stiff(i, m)) continue;
    double q0m = 0.0;
    for (int k = 0; k < 3; k++) q0m += evecs[i][3 * m + k] * dxr[k];
    double coeff = mi * om[i][m] * om[i][m] * q0m;
    for (int k = 0; k < 3; k++) fcorr[k] += evecs[i][3 * m + k] * coeff;
  }
}

/* ----------------------------------------------------------------------
   finite-difference curvature-block acquisition (group atoms only):
   for each atom, 3 central-difference force probes (+-eps along x/y/z)
   give a 3x3 on-diagonal block; symmetrize, mass-weight, diagonalize via
   jacobi3, and reset the frozen center to the atom's current position.

   Stage 2 additionally reads, at the same 6 probes (no extra force
   calls), the force *change* on every one of atom i's current pair-style
   neighbors, builds the cross-Hessian columns Hc_j = dF_j/dx_i, and folds
   those into the sparse per-neighbor response blocks Jblk (see the header
   comment and force_moll() for how these are used -- the D.10.4/D.10.6
   linear-response back-reaction, restricted to atom i's own neighbor list
   since the true Hessian row has no support beyond it).

   Mirrors Min::energy_force()'s force-recompute sequence but skips
   neighbor->decide()/rebuild/exchange/borders: eps is small relative to
   the neighbor skin, so the existing neighbor list stays valid for the
   perturbed probe positions (documented stage-1 assumption, unchanged).
------------------------------------------------------------------------- */

void FixBAOABTether::refresh_blocks()
{
  double **x = atom->x;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // stage-4: both FD-probe loops below now go entirely through the local,
  // no-comm partial_force() kernel -- they read/write only local scratch
  // (xt[]/fplus[]/fminus[]/fj_plus[]/fj_minus[]), never atom->x or
  // atom->f, so there is nothing to stash and restore here any more (the
  // old whole-system recompute_forces_local()-based probe clobbered
  // atom->f and needed a save/restore dance; that whole mechanism is gone
  // along with it -- see the loops below and partial_force()'s own
  // updated doc comment for why this also fixes the multi-rank crash a
  // per-atom-count-dependent number of forward/reverse_comm() calls used
  // to cause).

  // one-time bootstrap (adapt=yes only, before om_split has ever been set):
  // mode_is_stiff()'s per-mode mask below needs om_split, but om_split
  // (D.11.1's spectral-gap split) is itself derived from the full group's
  // eigenfrequency census -- a chicken-and-egg problem only on the very
  // first refresh, since afterward om_split is held fixed (see the header)
  // and every subsequent refresh's mask decision only needs THIS atom's own
  // just-recomputed om[i], not the atom loop order. Resolved by running a
  // throwaway FD-probe pass first, touching only om[]/evecs[] (no Hc/Jblk,
  // no group-membership eligibility use yet), to seed the one-shot split;
  // the main loop below then repeats the same probes (needed regardless,
  // to also build Hc/Jblk) with om_split now available. This doubles the
  // force-evaluation cost of exactly one refresh call, once per run.
  if (adapt && !om_split_set) {
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;

      double mi = rmass ? rmass[i] : mass[type[i]];
      double xi[3] = {x[i][0], x[i][1], x[i][2]};
      double H[3][3];

      for (int b = 0; b < 3; b++) {
        double fplus[3], fminus[3];
        double xt[3] = {xi[0], xi[1], xi[2]};

        xt[b] = xi[b] + eps;
        partial_force(i, xt, fplus, nullptr, true);

        xt[b] = xi[b] - eps;
        partial_force(i, xt, fminus, nullptr, true);

        for (int a = 0; a < 3; a++) H[a][b] = -(fplus[a] - fminus[a]) / (2.0 * eps);
      }

      double Hs[3][3];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) Hs[a][b] = 0.5 * (H[a][b] + H[b][a]) / mi;

      double eval[3], evec[3][3];
      MathEigen::jacobi3(Hs, eval, evec);
      for (int m = 0; m < 3; m++) {
        om[i][m] = sqrt(MAX(eval[m], 0.0));
        for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = evec[a][m];
      }
    }
    compute_om_split();
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    double H[3][3];

    int jnum = 0;
    int *jlist = nullptr;
    if (mollify && list) {
      jlist = list->firstneigh[i];
      jnum = list->numneigh[i];
      if (jnum > JMAX)
        error->one(FLERR, "Fix baoab/tether: atom exceeds JMAX ({}) neighbors for the "
                           "sparse response block -- raise JMAX and recompile", JMAX);
    }
    // neighbor force-response columns: Hc[k][a][b] = dF_{jlist[k],a}/dx_{i,b}
    std::vector<double> Hc;
    if (jnum > 0) Hc.assign((size_t) jnum * 9, 0.0);

    std::vector<double> fj_plus, fj_minus;
    if (jnum > 0) {
      fj_plus.resize((size_t) jnum * 3);
      fj_minus.resize((size_t) jnum * 3);
    }

    for (int b = 0; b < 3; b++) {
      double fplus[3], fminus[3];
      double xt[3] = {xi[0], xi[1], xi[2]};

      xt[b] = xi[b] + eps;
      partial_force(i, xt, fplus, jnum > 0 ? fj_plus.data() : nullptr, true);

      xt[b] = xi[b] - eps;
      partial_force(i, xt, fminus, jnum > 0 ? fj_minus.data() : nullptr, true);

      for (int a = 0; a < 3; a++) H[a][b] = -(fplus[a] - fminus[a]) / (2.0 * eps);

      for (int k = 0; k < jnum; k++)
        for (int a = 0; a < 3; a++)
          Hc[(size_t) k * 9 + 3 * a + b] =
              (fj_plus[(size_t) k * 3 + a] - fj_minus[(size_t) k * 3 + a]) / (2.0 * eps);
    }

    double Hs[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) Hs[a][b] = 0.5 * (H[a][b] + H[b][a]) / mi;

    double eval[3], evec[3][3];
    MathEigen::jacobi3(Hs, eval, evec);

    for (int m = 0; m < 3; m++) {
      om[i][m] = sqrt(MAX(eval[m], 0.0));
      for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = evec[a][m];
    }
    x0[i][0] = xi[0];
    x0[i][1] = xi[1];
    x0[i][2] = xi[2];
    c[i][0] = xi[0];
    c[i][1] = xi[1];
    c[i][2] = xi[2];

    // S_i = (1/mi) * W * diag(msk/om^2) * W^T -- the masked-mode Cartesian
    // compliance block (D.10.4); zero contribution from soft/unmasked
    // modes and from modes with (numerically) zero curvature.
    double Si[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    for (int m = 0; m < 3; m++) {
      if (!mode_is_stiff(i, m)) continue;
      if (om[i][m] <= 0.0) continue;
      double inv = 1.0 / (mi * om[i][m] * om[i][m]);
      for (int a = 0; a < 3; a++)
        for (int bb = 0; bb < 3; bb++)
          Si[3 * a + bb] += evecs[i][3 * m + a] * evecs[i][3 * m + bb] * inv;
    }

    // Exclude neighbors that are themselves flagged (stiff/rattler) atoms:
    // the adiabatic response must depend on slow (cage) coordinates only.
    // Letting one rattler's clamped equilibrium respond to another
    // rattler's own fast oscillation couples two fast modes through their
    // response blocks -- a positive-feedback channel that parametrically
    // pumps neighboring rattler pairs (matches
    // lj_longstep_prototype.py's refresh(), "self.J[:, :, s, :] = 0.0" and
    // its accompanying comment; this is the documented fix for exactly the
    // slow light-atom heating this stage-2 port was showing).
    int nk = 0;
    for (int k = 0; k < jnum; k++) {
      int j = jlist[k] & NEIGHMASK;
      if (mask[j] & groupbit) continue;
      Jtag[i][nk] = atom->tag[j];
      const double *Hck = &Hc[(size_t) k * 9];
      // Jblk_k = Hc_k . S_i, so that the back-reaction is a plain matvec
      // at force_moll() time: f[j] += Jblk_k * g[i] (see force_moll()).
      mat3_matmat(Hck, Si, Jblk[i][nk]);
      nk++;
    }
    nJ[i] = nk;
  }

  // forgiveness (D.11.1): an atom that has gone a while without tripping a
  // guard gets one chronic-failure strike removed, so a rare, long-past
  // demotion does not permanently inflate the exponential-backoff exponent
  // solve_center()'s guard action uses. Harmless no-op when adapt=no:
  // demote_count never leaves 0 for any atom in that mode.
  bigint ntimestep = update->ntimestep;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (demote_count[i] > 0 && (ntimestep - last_demote[i]) > 400) {
      demote_count[i]--;
      last_demote[i] = ntimestep;
    }
  }

  if (adapt) update_dt();
}

/* ----------------------------------------------------------------------
   one-shot spectral-gap detection (D.11.1, adapt=yes only): collect every
   flagged atom's 3 eigenfrequencies on this rank, sort them, and split the
   stiff/soft population at the largest multiplicative gap in the upper
   half of the sorted list -- direct port of the Python reference's own
   one-time om_split bootstrap in refresh(). Held fixed after this single
   call (see the header) so it cannot drift with instantaneous thermal
   fluctuations. Stage-4: under multi-rank, every rank's local frequency
   list is merged via MPI_Allgatherv before the gap search below runs, so
   every rank picks the identical om_split -- see the merge block inline.
------------------------------------------------------------------------- */

void FixBAOABTether::compute_om_split()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  std::vector<double> freqs;
  freqs.reserve((size_t) nlocal * 3);
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    for (int m = 0; m < 3; m++)
      if (om[i][m] > 0.0) freqs.push_back(om[i][m]);
  }

  // stage-4: this is a one-shot (om_split_set latches after this call)
  // spectral-gap detection over the WHOLE group's frequency spectrum
  // (D.11.1) -- under multi-rank each rank only sees its own flagged
  // atoms, so merge every rank's local list via Allgatherv before sorting,
  // and every rank then runs the identical deterministic serial gap search
  // below on the same merged array, giving every rank the same om_split.
  if (comm->nprocs > 1) {
    int n_local = (int) freqs.size();
    std::vector<int> recvcounts(comm->nprocs), displs(comm->nprocs);
    MPI_Allgather(&n_local, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, world);
    displs[0] = 0;
    for (int p = 1; p < comm->nprocs; p++) displs[p] = displs[p - 1] + recvcounts[p - 1];
    int n_total = displs[comm->nprocs - 1] + recvcounts[comm->nprocs - 1];
    std::vector<double> freqs_all(n_total);
    MPI_Allgatherv(freqs.data(), n_local, MPI_DOUBLE, freqs_all.data(), recvcounts.data(),
                   displs.data(), MPI_DOUBLE, world);
    freqs.swap(freqs_all);
  }

  // degenerate group (too few modes for a gap to mean anything) -- fall
  // back to the fixed theta/dt threshold so mode_is_stiff() still behaves
  // sanely instead of comparing every frequency against om_split == 0.
  if (freqs.size() < 2) {
    om_split = theta / update->dt;
    om_split_set = 1;
    return;
  }

  std::sort(freqs.begin(), freqs.end());
  size_t half = freqs.size() / 2;

  double best_gap = -1.0;
  size_t best_k = half;
  for (size_t k = half; k + 1 < freqs.size(); k++) {
    if (freqs[k] <= 0.0) continue;
    double gap = freqs[k + 1] / freqs[k];
    if (gap > best_gap) {
      best_gap = gap;
      best_k = k;
    }
  }

  om_split = (best_gap > 0.0) ? sqrt(freqs[best_k] * freqs[best_k + 1]) : (theta / update->dt);
  om_split_set = 1;
}

/* ----------------------------------------------------------------------
   adaptive timestep selection (D.11.1, adapt=yes only), called from
   refresh_blocks() once the per-mode mask/om_split machinery is current
   for this refresh: recompute the quantized-ladder rung dt_max/2^dt_lvl
   from the fastest UNRESOLVED (eligible-but-not-currently-analytic) mode
   on this rank, clamp against the resonance ceiling set by the fastest
   ANALYTIC mode, then clip to [dt_min, dt_max]. The ladder only shrinks
   immediately; growing back up requires a full extra 1.25x safety margin
   and happens one rung at a time, so a transient stiffening never gets
   "rounded away" and a return to softness is not chased eagerly enough to
   thrash -- direct port of the reference's own hysteresis. If the
   resulting dt differs from update->dt, applies it via the exact
   fix_dt_reset.cpp notification sequence. Stage-4: om_unres/om_analytic_max
   below are Allreduce'd (MPI_MAX) across ranks before dt_c is computed, so
   every rank derives the identical dt_new and takes the identical
   update->dt branch -- without this, different ranks could each compute
   a different local maximum and try to set different values into the
   single shared update->dt scalar.
------------------------------------------------------------------------- */

void FixBAOABTether::update_dt()
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double om_unres = 0.0;
  double om_analytic_max = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    for (int m = 0; m < 3; m++) {
      if (mode_is_stiff(i, m))
        om_analytic_max = MAX(om_analytic_max, om[i][m]);
      else
        om_unres = MAX(om_unres, om[i][m]);
    }
  }

  if (comm->nprocs > 1) {
    double local[2] = {om_unres, om_analytic_max};
    double global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, world);
    om_unres = global[0];
    om_analytic_max = global[1];
  }

  // no unresolved mode on ANY rank at all (every flagged mode is either
  // analytic or zero) -- fall back to a floor that keeps dt_c well inside
  // [dt_min, dt_max] rather than dividing by zero.
  if (om_unres <= 0.0) om_unres = c_acc / dt_max;

  double dt_c = c_acc / om_unres;
  if (om_analytic_max > 0.0) dt_c = MIN(dt_c, 2.2 / om_analytic_max);
  dt_c = MAX(dt_min, MIN(dt_max, dt_c));

  if (dt_lvl < 0) {
    // first-ever call: initialize the rung directly from dt_c, no hysteresis.
    dt_lvl = MAX(0, (int) lround(log(dt_max / dt_c) / log(2.0)));
  } else {
    double rung_dt = dt_max / pow(2.0, dt_lvl);
    if (dt_c < 0.85 * rung_dt) {
      dt_lvl = MAX(dt_lvl + 1, (int) ceil(log(dt_max / dt_c) / log(2.0)));
    } else if (dt_c > 1.25 * rung_dt && dt_lvl > 0) {
      dt_lvl--;
    }
  }

  double dt_new = MAX(dt_min, MIN(dt_max, dt_max / pow(2.0, dt_lvl)));

  if (dt_new != update->dt) {
    update->update_time();
    update->dt = dt_new;
    update->dt_default = 0;
    if (utils::strmatch(update->integrate_style, "^respa")) update->integrate->reset_dt();
    if (force->pair) force->pair->reset_dt();
    for (const auto &ifix : modify->get_fix_list()) ifix->reset_dt();
    output->reset_dt();
  }
}

/* ----------------------------------------------------------------------
   damped-Newton solve for atom i's instantaneous clamped equilibrium,
   given its CURRENT cage (all other atoms held fixed at their present
   positions). Direct port of _solve_centers() in lj_longstep_prototype.py,
   restricted per-atom (no cross-atom J-based warm start -- see the plan's
   scope decision #1): only the masked (locally stiff) modal components of
   the position are solved for; unmasked components track the atom's real
   current position exactly (no clamping needed there). Warm-starts from
   the atom's previous converged center c[i] (== x0[i] immediately after a
   refresh).
------------------------------------------------------------------------- */

void FixBAOABTether::solve_center(int i, double mi, double kT, double *cnew)
{
  double **x = atom->x;

  bool msk[3];
  double om_top = 0.0;
  for (int m = 0; m < 3; m++) {
    msk[m] = mode_is_stiff(i, m);
    om_top = MAX(om_top, om[i][m]);
  }

  // x[i]/c[i] can be in a different periodic image than x0[i] (set once at
  // the last refresh) after an intervening domain->pbc() wrap -- minimum-
  // image both differences before projecting into the eigenbasis below, or
  // a stray box-length offset gets rotated into every Cartesian component
  // of the reconstructed center (see initial_integrate()'s dxr for the
  // same issue on the analytic-drift side).
  double dx_now[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};
  double dx_warm[3] = {c[i][0] - x0[i][0], c[i][1] - x0[i][1], c[i][2] - x0[i][2]};
  domain->minimum_image(FLERR, dx_now);
  domain->minimum_image(FLERR, dx_warm);
  // nearest image of x0[i] to the atom's CURRENT position -- reconstruct
  // trial/cnew from this instead of raw x0[i], so they land in x[i]'s
  // image and stay consistent with partial_force()'s neighbor-list ghosts
  // (which are built relative to the current image, not x0[i]'s).
  double x0_eff[3] = {x[i][0] - dx_now[0], x[i][1] - dx_now[1], x[i][2] - dx_now[2]};
  double q_now[3] = {0, 0, 0}, q[3] = {0, 0, 0}, q_warm[3] = {0, 0, 0};
  for (int m = 0; m < 3; m++)
    for (int k = 0; k < 3; k++) {
      double wmk = evecs[i][3 * m + k];
      q_now[m] += wmk * dx_now[k];
      q_warm[m] += wmk * dx_warm[k];
    }
  for (int m = 0; m < 3; m++) q[m] = msk[m] ? q_warm[m] : q_now[m];

  bool any_masked = msk[0] || msk[1] || msk[2];
  double a_th = (om_top > 0.0) ? sqrt(kT) / om_top : 0.0;
  double tol = solve_tol_rel * a_th;
  double damp = newton_damp;
  double prev_res = -1.0;
  double res = 0.0;

  if (any_masked) {
    for (int iter = 0; iter < newton_iters; iter++) {
      double trial[3];
      for (int k = 0; k < 3; k++) {
        trial[k] = x0_eff[k];
        for (int m = 0; m < 3; m++) trial[k] += evecs[i][3 * m + k] * q[m];
      }

      double fout[3];
      partial_force(i, trial, fout);
      n_partial_force_calls++;

      double r[3] = {0, 0, 0};
      for (int m = 0; m < 3; m++)
        if (msk[m])
          for (int k = 0; k < 3; k++) r[m] += evecs[i][3 * m + k] * fout[k];

      res = 0.0;
      for (int m = 0; m < 3; m++)
        if (msk[m]) res += r[m] * r[m];
      res = sqrt(res);

      if (prev_res >= 0.0 && res > prev_res) damp = MAX(0.05, damp * 0.5);
      prev_res = res;

      for (int m = 0; m < 3; m++) {
        if (!msk[m]) continue;
        double om_safe = MAX(om[i][m], 1.0e-8);
        double delta = damp * r[m] / (mi * om_safe * om_safe);
        double cap = 8.0 * sqrt(kT) / om_safe;
        if (delta > cap) delta = cap;
        if (delta < -cap) delta = -cap;
        q[m] += delta;
      }

      if (res < tol && iter >= 7) break;
    }
  }

  for (int k = 0; k < 3; k++) {
    cnew[k] = x0_eff[k];
    for (int m = 0; m < 3; m++) cnew[k] += evecs[i][3 * m + k] * q[m];
  }

  // self-evaluation guard signals (D.11.4): counted always when
  // mollify=yes (diagnostic, matches Stage 2 exactly); ACTED ON (an actual
  // demotion) only when adapt=yes, matching the reference's own
  // "& self.adapt" gate -- D.11.2's own safety argument is that a demoted
  // stiff mode meets Verlet at omega*dt ~ 2.2 in fixed-dt mode and
  // destabilizes instantly, so this guard's action is unsafe without the
  // adaptive controller. newton_tol_rel/flip_rel are the reference's own
  // hardcoded guard-scale constants (lj_longstep_prototype.py
  // self.newton_tol_rel=0.25, self.flip_rel=14.0) -- DISTINCT from
  // solve_tol_rel (self.solve_tol_rel=0.005), which only sets the Newton
  // EXIT criterion. Reusing solve_tol_rel for the guard threshold here
  // previously under-scaled the newton/flip guards by ~50x/28x relative
  // to the reference.
  if (any_masked) {
    const double newton_tol_rel = 0.25;
    const double flip_rel = 14.0;
    bool g_n = res > 20.0 * newton_tol_rel * a_th;
    if (g_n) n_guard_newton++;

    double flip = 0.0;
    for (int m = 0; m < 3; m++)
      if (msk[m]) flip += (q[m] - q_warm[m]) * (q[m] - q_warm[m]);
    flip = sqrt(flip);
    bool g_f = (a_th > 0.0) && (flip > flip_rel * a_th);
    if (g_f) n_guard_flip++;

    double eharm = 0.0;
    for (int m = 0; m < 3; m++)
      if (msk[m]) eharm += 0.5 * mi * om[i][m] * om[i][m] * q[m] * q[m];
    bool g_e = eharm > 25.0 * kT;
    if (g_e) n_guard_energy++;

    // demotion action (adapt=yes only): exponential backoff capped at 2000
    // steps, direct port of the reference's
    // "3*K*2**min(demote_count-1,6)" -- demote_count[i] here is still the
    // PRE-increment value, so MIN(demote_count[i], 6) is exactly
    // MIN(demote_count-1, 6) after demote_atom()'s increment below.
    if (adapt && (g_n || g_f || g_e)) {
      int expo = MIN(demote_count[i], 6);
      bigint mult = ((bigint) 1) << expo;
      bigint backoff = MIN((bigint) refresh_every * 3 * mult, (bigint) 2000);
      demote_atom(i, backoff);
      for (int k = 0; k < 3; k++) cnew[k] = x[i][k];
    }
  }
}

/* ----------------------------------------------------------------------
   local, no-global-force-call partial force on atom i if it were placed
   at xtrial, from its own (real, unmoved) pair-style neighbors only --
   the Newton inner-loop kernel plan doc SS5.1 specifies for classical
   pairs. Requires the private full neighbor list requested in init().
   Valid as long as |xtrial - x[i]| stays well inside the neighbor skin
   (the Newton step cap in solve_center() bounds this to a few thermal
   amplitudes, the same small-perturbation assumption refresh_blocks()
   already makes for the FD probe eps). Deliberately does NOT call
   domain->minimum_image(): j's coordinates come straight from the
   neighbor list, i.e. already the specific (possibly non-minimum, when
   cutoff exceeds half the box length) periodic image the list resolved
   for atom i's real position -- exactly the convention every pair
   style's own compute() loop uses for delx/dely/delz. Re-folding through
   minimum_image() here would collapse two distinct neighbor-list entries
   for the same periodic image pair onto the same vector whenever cutoff
   > L/2, corrupting the force sum.

   stage-4: optional fneigh output (nullptr for the Newton-solve callers,
   unaffected) returns, per neighbor jj in the SAME order as
   list->firstneigh[i], the reaction force that neighbor receives from the
   i-j pair at xtrial (i.e. -delx*fpair, by Newton's third law for a
   pairwise-additive interaction). This is what lets refresh_blocks()'s
   finite-difference Hessian probe (both the atom-i block H and the
   neighbor cross-derivative blocks Hc) run entirely off this one local,
   no-comm kernel instead of a whole-system pair->compute() + forward/
   reverse_comm() per probe -- the latter is a swap-based, call-count-
   synchronized routine, and under real multi-rank the number of local
   flagged atoms (hence the number of probe calls) generally differs
   between ranks, which desynchronizes those swaps. A local kernel has no
   such requirement: every rank calls it however many times it needs to,
   independently.

   stage-4: live_neighbor_positions (default false, preserving the Newton-
   solve callers' existing behavior below) overrides the c[j]-for-flagged-
   neighbors substitution to always use the neighbor's real, current x[j]
   instead. refresh_blocks()'s FD-Hessian probe MUST pass true here: it is
   estimating the actual instantaneous local curvature of the real system
   (the quantity that sets the integrator's stiff/soft classification and
   timescale), not solving an implicit equation that needs adjacent
   rattlers decoupled from each other's fast phase. Using c[j] there was a
   real bug (not just an inconsistency): whenever a probed atom's neighbor
   list happens to include another flagged atom whose clamped center sits
   at a different distance than its live position, the FD probe samples
   the pair potential's curvature at the WRONG separation -- for a steep
   repulsive potential like LJ this can inflate a single atom's estimated
   frequency by nearly 2x (see the stage-4 regression investigation:
   atom om jumped from ~149 to ~284 solely from this substitution),
   corrupting the one-shot om_split spectral-gap split for the whole run.
------------------------------------------------------------------------- */

bool FixBAOABTether::partial_force(int i, const double *xtrial, double *fout, double *fneigh,
                                    bool live_neighbor_positions) const
{
  fout[0] = fout[1] = fout[2] = 0.0;
  if (!list) return false;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int itype = type[i];
  double **cutsq = force->pair->cutsq;

  int *jlist = list->firstneigh[i];
  int jnum = list->numneigh[i];

  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj] & NEIGHMASK;
    int jtype = type[j];

    // A neighbor that is itself a flagged (rattler) atom must contribute
    // via its own current CENTER, not its instantaneous real position --
    // otherwise atom i's clamped equilibrium responds directly to atom j's
    // fast real-time oscillation phase, a resonant parametric-coupling
    // channel between adjacent rattlers. This mirrors
    // lj_longstep_prototype.py's _solve_centers(), which places ALL
    // flagged atoms at their (jointly converging) trial centers before
    // every force evaluation, never at their real xw positions; it is the
    // analogous fix for the Newton solve itself, complementing (not
    // duplicating) the rattler-rattler Jblk exclusion in refresh_blocks(),
    // which only concerns the back-reaction operator.
    const double *xj = (!live_neighbor_positions && (mask[j] & groupbit)) ? c[j] : x[j];

    double delx = xtrial[0] - xj[0];
    double dely = xtrial[1] - xj[1];
    double delz = xtrial[2] - xj[2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq >= cutsq[itype][jtype]) {
      if (fneigh) fneigh[3 * jj] = fneigh[3 * jj + 1] = fneigh[3 * jj + 2] = 0.0;
      continue;
    }

    double fpair = 0.0;
    force->pair->single(i, j, itype, jtype, rsq, 1.0, 1.0, fpair);
    fout[0] += delx * fpair;
    fout[1] += dely * fpair;
    fout[2] += delz * fpair;
    if (fneigh) {
      fneigh[3 * jj + 0] = -delx * fpair;
      fneigh[3 * jj + 1] = -dely * fpair;
      fneigh[3 * jj + 2] = -delz * fpair;
    }
  }
  return true;
}

/* ----------------------------------------------------------------------
   Stage-2 mollified force field (replaces modal_force_correction() when
   mollify=yes): clamp every flagged atom to its just-solved center c[i],
   recompute the REAL force field once for the whole local domain at that
   clamped configuration (this already gives the physically correct force
   on the cage from the clamped stiff atoms -- no shortcut needed there),
   then per flagged atom zero its own masked-modal force component (the
   analytic drift already integrates that part exactly, same reasoning as
   modal_force_correction()) and scatter the D.10.6 chain-rule
   back-reaction onto its neighbor list via the sparse Jblk computed in
   refresh_blocks(). Restores every clamped atom's real position before
   returning -- the clamped configuration is a scratch evaluation, not the
   integrator's actual state.
------------------------------------------------------------------------- */

void FixBAOABTether::force_moll()
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  std::vector<int> flagged;
  flagged.reserve(nlocal);
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) flagged.push_back(i);

  std::vector<double> xsave(3 * flagged.size());
  for (size_t k = 0; k < flagged.size(); k++) {
    int i = flagged[k];
    xsave[3 * k + 0] = x[i][0];
    xsave[3 * k + 1] = x[i][1];
    xsave[3 * k + 2] = x[i][2];
    x[i][0] = c[i][0];
    x[i][1] = c[i][1];
    x[i][2] = c[i][2];
  }

  comm->forward_comm();

  recompute_forces_local();

  // stage-4: recompute_forces_local()'s own comm->reverse_comm() above
  // (inside its final line) already sent home every ghost's raw
  // pair-force contribution -- but AtomVec::unpack_reverse() only adds
  // into the OWNER's f[], it never clears the GHOST's own f[] slot
  // afterward. Zero the ghost region now so the back-reaction loop below
  // writes into a clean slate; the second comm->reverse_comm() after that
  // loop then sends home exactly the NEW back-reaction contribution, with
  // no double-counting of the already-reconciled raw pair force.
  if (atom->nghost) memset(&f[atom->nlocal][0], 0, 3 * sizeof(double) * atom->nghost);

  for (size_t k = 0; k < flagged.size(); k++) {
    int i = flagged[k];

    double gi[3] = {f[i][0], f[i][1], f[i][2]};

    // back-reaction onto atom i's stored neighbors, using the FULL
    // residual g[i] -- Jblk already carries the masked-mode projector
    // (via S_i in refresh_blocks()), so unmasked components contribute
    // exactly zero here.
    for (int kk = 0; kk < nJ[i]; kk++) {
      int j = atom->map(Jtag[i][kk]);
      if (j < 0) {
        // stage-4: neighbor drifted outside the current ghost cutoff
        // since the last refresh -- fails safe (drop this contribution)
        // rather than crash; see n_ghost_miss in the header.
        n_ghost_miss++;
        continue;
      }
      double contrib[3];
      mat3_matvec(Jblk[i][kk], gi, contrib);
      f[j][0] += contrib[0];
      f[j][1] += contrib[1];
      f[j][2] += contrib[2];
    }

    // zero atom i's own masked-modal force component (mirrors
    // modal_force_correction()'s role: that part is handled exactly by
    // the analytic drift, not by the impulsive kick).
    for (int m = 0; m < 3; m++) {
      if (!mode_is_stiff(i, m)) continue;
      double gm = 0.0;
      for (int a = 0; a < 3; a++) gm += evecs[i][3 * m + a] * gi[a];
      for (int a = 0; a < 3; a++) f[i][a] -= evecs[i][3 * m + a] * gm;
    }
  }

  // stage-4: send the new back-reaction contributions written into
  // possibly-ghost f[] slots above home to their owning rank.
  // Unconditional -- unlike recompute_forces_local()'s own
  // "if (force->newton) comm->reverse_comm()", this loop writes directly
  // into f[j] itself regardless of the pair style's newton setting, so it
  // always needs the trip home.
  comm->reverse_comm();

  for (size_t k = 0; k < flagged.size(); k++) {
    int i = flagged[k];
    x[i][0] = xsave[3 * k + 0];
    x[i][1] = xsave[3 * k + 1];
    x[i][2] = xsave[3 * k + 2];
  }
  comm->forward_comm();
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::force_clear_local()
{
  size_t nbytes = sizeof(double) * (atom->nlocal + atom->nghost);
  if (nbytes) memset(&atom->f[0][0], 0, 3 * nbytes);
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::recompute_forces_local()
{
  n_recompute_calls++;
  force_clear_local();

  if (force->pair) force->pair->compute(0, 0);
  if (force->bond) force->bond->compute(0, 0);
  if (force->angle) force->angle->compute(0, 0);
  if (force->dihedral) force->dihedral->compute(0, 0);
  if (force->improper) force->improper->compute(0, 0);
  if (force->kspace) force->kspace->compute(0, 0);

  if (modify->n_pre_reverse) modify->pre_reverse(0, 0);
  if (force->newton) comm->reverse_comm();
}

/* ---------------------------------------------------------------------- */

double FixBAOABTether::memory_usage()
{
  double bytes = (double) atom->nmax * (3 + 3 + 9 + 3 + 1) * sizeof(double);
  bytes += (double) atom->nmax * sizeof(int);
  bytes += (double) atom->nmax * JMAX * sizeof(tagint);
  bytes += (double) atom->nmax * JMAX * 9 * sizeof(double);
  bytes += (double) atom->nmax * sizeof(int);
  bytes += (double) atom->nmax * 3 * sizeof(bigint);
  return bytes;
}

/* ----------------------------------------------------------------------
   diagnostic vector, this MPI rank's local counts only: [0..2]
   self-evaluation guard counts, [3] partial_force() calls, [4]
   recompute_forces_local() calls, [5] total demotion events (any cause,
   adapt=yes only), [6] demotion events from the kinetic over-excitation
   guard specifically (subset of [5]), [7] stage-4 ghost-miss count (a
   stored sparse-block neighbor could not be resolved via atom->map(),
   see n_ghost_miss in the header -- always 0 on a single rank with a
   generous cutoff/skin).
------------------------------------------------------------------------- */

double FixBAOABTether::compute_vector(int n)
{
  switch (n) {
    case 0: return (double) n_guard_newton;
    case 1: return (double) n_guard_flip;
    case 2: return (double) n_guard_energy;
    case 3: return (double) n_partial_force_calls;
    case 4: return (double) n_recompute_calls;
    case 5: return (double) n_demoted;
    case 6: return (double) n_kinetic_guard;
    case 7: return (double) n_ghost_miss;
    default: return 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::grow_arrays(int nmax)
{
  memory->grow(x0, nmax, 3, "baoab/tether:x0");
  memory->grow(c, nmax, 3, "baoab/tether:c");
  memory->grow(evecs, nmax, 9, "baoab/tether:evecs");
  memory->grow(om, nmax, 3, "baoab/tether:om");
  memory->grow(nJ, nmax, "baoab/tether:nJ");
  memory->grow(Jtag, nmax, JMAX, "baoab/tether:Jtag");
  memory->grow(Jblk, nmax, JMAX, 9, "baoab/tether:Jblk");

  memory->grow(demote_count, nmax, "baoab/tether:demote_count");
  memory->grow(last_demote, nmax, "baoab/tether:last_demote");
  memory->grow(demote_cool, nmax, "baoab/tether:demote_cool");
  memory->grow(quench_exempt, nmax, "baoab/tether:quench_exempt");
  memory->grow(last_ke, nmax, "baoab/tether:last_ke");

  // zero-init on every grow (not just the newly-added tail): correctness-
  // critical since mode_is_stiff() reads demote_cool unconditionally for
  // every group atom. This resets any in-flight cooldown/backoff state for
  // existing atoms on the rare event of a mid-run atom-count growth (no fix
  // in this project's workloads inserts/removes atoms, so in practice this
  // only ever fires once, at construction, when it's a pure no-op).
  for (int i = 0; i < nmax; i++) {
    demote_count[i] = 0;
    last_demote[i] = 0;
    demote_cool[i] = 0;
    quench_exempt[i] = 0;
    last_ke[i] = 0.0;
  }
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int k = 0; k < 3; k++) x0[j][k] = x0[i][k];
  for (int k = 0; k < 3; k++) c[j][k] = c[i][k];
  for (int k = 0; k < 9; k++) evecs[j][k] = evecs[i][k];
  for (int k = 0; k < 3; k++) om[j][k] = om[i][k];

  nJ[j] = nJ[i];
  for (int k = 0; k < nJ[i]; k++) {
    Jtag[j][k] = Jtag[i][k];
    for (int a = 0; a < 9; a++) Jblk[j][k][a] = Jblk[i][k][a];
  }

  demote_count[j] = demote_count[i];
  last_demote[j] = last_demote[i];
  demote_cool[j] = demote_cool[i];
  quench_exempt[j] = quench_exempt[i];
  last_ke[j] = last_ke[i];
}

/* ---------------------------------------------------------------------- */

int FixBAOABTether::pack_exchange(int i, double *buf)
{
  int m = 0;
  for (int k = 0; k < 3; k++) buf[m++] = x0[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = c[i][k];
  for (int k = 0; k < 9; k++) buf[m++] = evecs[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = om[i][k];

  buf[m++] = ubuf(nJ[i]).d;
  for (int k = 0; k < nJ[i]; k++) {
    buf[m++] = ubuf(Jtag[i][k]).d;
    for (int a = 0; a < 9; a++) buf[m++] = Jblk[i][k][a];
  }

  buf[m++] = ubuf(demote_count[i]).d;
  buf[m++] = ubuf(last_demote[i]).d;
  buf[m++] = ubuf(demote_cool[i]).d;
  buf[m++] = ubuf(quench_exempt[i]).d;
  buf[m++] = last_ke[i];
  return m;
}

/* ---------------------------------------------------------------------- */

int FixBAOABTether::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  for (int k = 0; k < 3; k++) x0[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) c[nlocal][k] = buf[m++];
  for (int k = 0; k < 9; k++) evecs[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) om[nlocal][k] = buf[m++];

  nJ[nlocal] = (int) ubuf(buf[m++]).i;
  for (int k = 0; k < nJ[nlocal]; k++) {
    Jtag[nlocal][k] = (tagint) ubuf(buf[m++]).i;
    for (int a = 0; a < 9; a++) Jblk[nlocal][k][a] = buf[m++];
  }

  demote_count[nlocal] = (int) ubuf(buf[m++]).i;
  last_demote[nlocal] = (bigint) ubuf(buf[m++]).i;
  demote_cool[nlocal] = (bigint) ubuf(buf[m++]).i;
  quench_exempt[nlocal] = (bigint) ubuf(buf[m++]).i;
  last_ke[nlocal] = buf[m++];
  return m;
}

/* ----------------------------------------------------------------------
   stage-4: forward comm of this fix's private c[] array, so a ghost
   neighbor's c[j] read by partial_force() (line ~1199) reflects the
   owning rank's actual last-converged center instead of stale/garbage
   ghost-buffer memory. c[] holds raw Cartesian positions in the same
   frame as atom->x, so it needs the identical periodic-image shift
   AtomVec::pack_comm() applies to x[] when pbc_flag is set (triclinic
   tilt included) -- otherwise a ghost that crosses a periodic boundary
   would have a c[] in the wrong image relative to its own (correctly
   shifted) x[], corrupting the minimum-image reconstruction in
   solve_center(). Called once per step, before the local solve_center()
   loop -- see initial_integrate()/final_integrate() -- so a ghost's c[]
   reflects the previous step's fully-converged value, matching how a
   not-yet-reached local neighbor behaves in the same loop (deliberate
   Gauss-Seidel-consistent choice, see the plan).
------------------------------------------------------------------------- */

int FixBAOABTether::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int m = 0;
  if (pbc_flag == 0) {
    for (int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = c[j][0];
      buf[m++] = c[j][1];
      buf[m++] = c[j][2];
    }
  } else {
    double dx, dy, dz;
    if (domain->triclinic == 0) {
      dx = pbc[0] * domain->xprd;
      dy = pbc[1] * domain->yprd;
      dz = pbc[2] * domain->zprd;
    } else {
      dx = pbc[0] * domain->xprd + pbc[5] * domain->xy + pbc[4] * domain->xz;
      dy = pbc[1] * domain->yprd + pbc[3] * domain->yz;
      dz = pbc[2] * domain->zprd;
    }
    for (int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = c[j][0] + dx;
      buf[m++] = c[j][1] + dy;
      buf[m++] = c[j][2] + dz;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    c[i][0] = buf[m++];
    c[i][1] = buf[m++];
    c[i][2] = buf[m++];
  }
}
