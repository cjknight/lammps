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
    evecs(nullptr), om(nullptr), Hacc(nullptr), n_hess_samples(nullptr), nJ(nullptr),
    Jtag(nullptr), Jblk(nullptr), demote_count(nullptr), last_demote(nullptr),
    demote_cool(nullptr), quench_exempt(nullptr), last_ke(nullptr)
{
  theta = -1.0;
  refresh_every = -1;
  eps = 0.0;
  eps_was_set = 0;
  heff_samples = 0;

  mollify = 0;
  newton_iters = 8;
  newton_damp = 0.6;
  solve_tol_rel = 0.005;
  use_local_partial_force = 0;

  adapt = 0;
  c_acc = 0.25;
  dt_min = -1.0;    // sentinel: default 0.02*dt, resolved once dt is known below
  dt_max = -1.0;    // sentinel: default the fix-creation-time dt, resolved below
  ke_rel = 12.0;
  gamma_quench = 20.0;
  refresh_skin = 0.0;
  drift_response = 0;
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
    } else if (strcmp(arg[iarg], "heff_samples") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether heff_samples", error);
      heff_samples = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
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
    } else if (strcmp(arg[iarg], "refresh_skin") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether refresh_skin", error);
      refresh_skin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "drift_response") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether drift_response", error);
      if (strcmp(arg[iarg + 1], "retether") == 0)
        drift_response = 0;
      else if (strcmp(arg[iarg + 1], "preshrink") == 0)
        drift_response = 1;
      else
        error->all(FLERR, iarg + 1, "Fix baoab/tether drift_response must be retether or preshrink");
      iarg += 2;
    } else if (strcmp(arg[iarg], "event_log") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether event_log", error);
      event_log_filename = arg[iarg + 1];
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown fix baoab/tether keyword {}", arg[iarg]);
    }
  }

  if (theta <= 0.0) error->all(FLERR, "Fix baoab/tether theta must be > 0");
  if (refresh_every <= 0) error->all(FLERR, "Fix baoab/tether refresh must be > 0");
  if (eps_was_set && eps <= 0.0) error->all(FLERR, "Fix baoab/tether eps must be > 0");
  if (heff_samples < 0) error->all(FLERR, "Fix baoab/tether heff_samples must be >= 0");
  if (newton_iters < 1) error->all(FLERR, "Fix baoab/tether newton_iters must be >= 1");
  if (newton_damp <= 0.0 || newton_damp > 1.0)
    error->all(FLERR, "Fix baoab/tether newton_damp must be in (0,1]");
  if (solve_tol_rel <= 0.0) error->all(FLERR, "Fix baoab/tether solve_tol_rel must be > 0");
  if (refresh_skin < 0.0) error->all(FLERR, "Fix baoab/tether refresh_skin must be >= 0");

  // permanent per-atom/per-event diagnostic log (plan Sec 6.1): opened once
  // here, for the fix's whole lifetime, matching fix_print's single-fopen-
  // in-constructor lifecycle -- this fix has no multi-run-command re-open
  // logic to preserve. Each rank opens its OWN file so this stays correct
  // under future MPI domain decomposition with no gather/collective added
  // to any per-step code path.
  if (!event_log_filename.empty()) {
    std::string fname = event_log_filename;
    if (comm->nprocs > 1) fname += fmt::format(".{}", comm->me);
    event_log_fp = fopen(fname.c_str(), "w");
    if (!event_log_fp)
      error->one(FLERR, "Fix baoab/tether: cannot open event_log file {}: {}", fname,
                 utils::getsyserror());
    fputs(fmt::format("# fix baoab/tether event log, rank {}\n"
                       "# event step tag ke ke_cap drift skin om0 om1 om2 dt note\n",
                       comm->me)
              .c_str(),
          event_log_fp);
    fflush(event_log_fp);
  }

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
  need_reneighbor = 0;
  dt_floor_until = -1;
  n_guard_newton = n_guard_flip = n_guard_energy = 0;
  n_demoted = n_kinetic_guard = 0;
  n_partial_force_calls = n_recompute_calls = 0;
  n_ghost_miss = 0;

  // request-a-real-reneighbor plumbing (Fix base class members): see
  // need_reneighbor's doc comment in the header. next_reneighbor starts at
  // a sentinel that can never match a real (>= 0) update->ntimestep, so
  // Neighbor::decide() only forces a reneighbor once we explicitly ask.
  force_reneighbor = 1;
  next_reneighbor = -1;

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
  // + c(3) + evecs(9) + om(3) + Hacc(9) + n_hess_samples(1) + nJ-count(1) +
  // JMAX*(tag(1)+block(9)) + demote_count/last_demote/demote_cool/
  // quench_exempt(4) + last_ke(1).
  maxexchange = 3 + 3 + 9 + 3 + 9 + 1 + 1 + JMAX * (1 + 9) + 4 + 1;

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
  memory->destroy(Hacc);
  memory->destroy(n_hess_samples);
  memory->destroy(nJ);
  memory->destroy(Jtag);
  memory->destroy(Jblk);
  memory->destroy(demote_count);
  memory->destroy(last_demote);
  memory->destroy(demote_cool);
  memory->destroy(quench_exempt);
  memory->destroy(last_ke);

  // event_log_fp is a SafeFilePtr member: its own destructor closes the
  // file (if opened) automatically, no explicit fclose() needed here.
}

/* ---------------------------------------------------------------------- */

// Attempt-8 diagnostic logging (plan Sec 6.1's required per-step/per-event
// log): no-op when event_log_fp is unset, via SafeFilePtr's implicit
// nullptr conversion -- every call site below is unconditionally reached,
// zero overhead when the "event_log" keyword is not given. Fields not
// meaningful for a given event use the -1.0 sentinel (ke/ke_cap/drift/
// skin/om/dt are all physically non-negative when actually logged).
// Always fflush(): the entire point of this log is to have forensic data
// survive up to the moment of a crash like the one it was built to
// diagnose, so buffered-but-lost output on a fatal error is not
// acceptable; events are rare relative to per-step compute cost (only on
// trips/refreshes/demotes, not every atom every step), so the extra
// fflush() cost is not a concern.
void FixBAOABTether::log_event(const char *event, tagint tag, double ke, double ke_cap,
                                double drift, double skin, const double *om, double dt,
                                const std::string &note)
{
  if (!event_log_fp) return;
  fputs(fmt::format("{} step={} tag={} ke={:.6g} ke_cap={:.6g} drift={:.6g} skin={:.6g} "
                     "om0={:.6g} om1={:.6g} om2={:.6g} dt={:.6g} note={}\n",
                     event, update->ntimestep, tag, ke, ke_cap, drift, skin,
                     om ? om[0] : -1.0, om ? om[1] : -1.0, om ? om[2] : -1.0, dt, note)
            .c_str(),
        event_log_fp);
  fflush(event_log_fp);
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
    use_local_partial_force = force->pair->has_local_partial_force;
    if (!use_local_partial_force && !force->pair->single_enable)
      error->all(FLERR, "Fix baoab/tether mollify yes requires a pair style that supports "
                         "either single() or local_partial_force()");
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

int FixBAOABTether::setmask()
{
  int mask = FixBAOAB::setmask();
  mask |= END_OF_STEP;
  return mask;
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
    bool sub_stepped = false;
    if (adapt) kinetic_guard_and_quench(i, mi, kT, msk, q1, p1, quench_active, sub_stepped);

    // drift_hot (refresh_skin > 0 only): kinetic_guard_and_quench() has
    // already advanced atom->x[i]/atom->v[i] all the way to the end of
    // this step itself, via local sub-stepping at a Verlet-safe dt_sub
    // (see sub_step_free_atom()) -- the O sub-step and second-half drift
    // below, which assume the shared per-atom pipeline's q1[]/p1[] are
    // still meaningful, must not run for this atom this step.
    if (sub_stepped) continue;

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

  // sub_step_free_atom() may have set need_reneighbor (rank-local) if a
  // drift_hot atom's local excursion hit its safe-displacement cap this
  // step. Allreduce before acting -- same requirement as need_refresh
  // above: every rank must agree, since next_reneighbor feeds directly
  // into Neighbor::decide()'s (unreduced) per-rank check, and disagreeing
  // ranks would desynchronize the collective comm/neighbor rebuild it
  // triggers.
  {
    int local_reneighbor = need_reneighbor ? 1 : 0;
    int global_reneighbor = local_reneighbor;
    if (comm->nprocs > 1) MPI_Allreduce(&local_reneighbor, &global_reneighbor, 1, MPI_INT, MPI_MAX, world);
    if (global_reneighbor) next_reneighbor = update->ntimestep + 1;
    need_reneighbor = 0;
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
   synchronous dt-shrink-retry (Attempt 7 Part B; adapt=yes, refresh_skin>0
   and drift_response==preshrink only): decide, one Verlet step ahead of
   time, whether update->dt needs to drop before any flagged atom's drift
   crosses refresh_skin. x[i] here is bit-identical to x[i] at the top of
   next step's initial_integrate(), so a shrink decided here is visible to
   every fix's initial_integrate() next step regardless of fix declaration
   order in the input script -- unlike an inline pre-pass inside this fix's
   own initial_integrate() (see setmask()).

   This is deliberately NOT the same same-step response
   kinetic_guard_and_quench()'s drift_hot branch takes under the default
   drift_response==retether: Attempt 7 Part A showed that a tight enough
   refresh_skin to catch a close encounter early instead fires that local
   re-tether/sub-step response on nearly every step from ordinary thermal
   jitter alone, and each trip's own approximation error compounds into
   the very encounter it was meant to prevent (see the plan file). Acting
   here, one step ahead, never calls sub_step_free_atom() -- its only
   effect is a smaller update->dt for the ordinary shared pipeline
   everyone already uses, so it cannot suffer that same self-reinforcing
   pathology no matter how often it fires.

   Deliberately does NOT reuse update_dt()'s om_unres-driven formula: a
   freshly-probed om[i]=0,0,0 (non-positive-definite curvature, the common
   signature right at a close encounter) contributes nothing to om_unres,
   and om_unres's own zero-fallback (c_acc/dt_max) would pick the LARGEST
   allowed dt -- the opposite of a shrink (confirmed: this is exactly why
   an earlier "force need_refresh=1" experiment had zero effect). Reuses
   sub_step_free_atom()'s formula instead, which already falls back
   correctly to dt_min for this case.
------------------------------------------------------------------------- */

void FixBAOABTether::end_of_step()
{
  if (!adapt || refresh_skin <= 0.0 || drift_response != 1) return;

  bigint ntimestep = update->ntimestep;
  if (ntimestep == last_refresh_step) return;    // x0[i] was just reset

  double **x = atom->x;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double dt_local = dt_max;
  int i_trigger = -1;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double dx0[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};
    domain->minimum_image(FLERR, dx0);
    double drift2 = dx0[0] * dx0[0] + dx0[1] * dx0[1] + dx0[2] * dx0[2];
    if (drift2 <= refresh_skin * refresh_skin) continue;

    refresh_one_atom(i, x[i], true);

    double om_max_new = MAX(om[i][0], MAX(om[i][1], om[i][2]));
    double dt_target = (om_max_new > 0.0) ? MIN(dt_min, 2.2 / om_max_new) : dt_min;
    if (dt_target < dt_local) {
      dt_local = dt_target;
      i_trigger = i;
    }
  }

  double dt_global = dt_local;
  if (comm->nprocs > 1) MPI_Allreduce(&dt_local, &dt_global, 1, MPI_DOUBLE, MPI_MIN, world);

  if (dt_global >= dt_max) return;    // no candidate on any rank this step

  // same shrink-only hysteresis update_dt() uses for its own dt_lvl++
  // branch -- never relax up here; relaxation is update_dt()'s job alone,
  // and only once dt_floor_until has passed.
  double rung_dt = dt_max / pow(2.0, MAX(dt_lvl, 0));
  if (dt_global < 0.85 * rung_dt) {
    dt_lvl = MAX(dt_lvl + 1, (int) ceil(log(dt_max / dt_global) / log(2.0)));
    double dt_new = MAX(dt_min, MIN(dt_max, dt_max / pow(2.0, dt_lvl)));
    apply_new_dt(dt_new);
    dt_floor_until = ntimestep + 3 * (bigint) refresh_every;
    log_event("PRESHRINK", i_trigger >= 0 ? atom->tag[i_trigger] : -1, -1.0, -1.0, -1.0, -1.0,
              nullptr, dt_new, fmt::format("old_dt={:.6g}", rung_dt));
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

void FixBAOABTether::demote_atom(int i, bigint backoff_steps, const char *source)
{
  bigint ntimestep = update->ntimestep;
  demote_count[i]++;
  last_demote[i] = ntimestep;
  demote_cool[i] = ntimestep + backoff_steps;
  n_demoted++;
  need_refresh = 1;
  log_event("DEMOTE", atom->tag[i], -1.0, -1.0, -1.0, -1.0, nullptr, -1.0,
            fmt::format("backoff={} source={}", backoff_steps, source));
}

/* ----------------------------------------------------------------------
   kinetic over-excitation guard + quench (D.11.3, adapt=yes only): called
   from initial_integrate() between the first drift half-step and the O
   sub-step, exactly where lj_longstep_prototype.py's run() inserts the
   same check. q1[]/p1[] are this atom's post-first-half-drift modal
   position/velocity (already computed by the caller, in the CURRENT
   eigenbasis evecs[i]); msk[] is this atom's current per-mode stiff mask.
   All three are mutated in place on a trip so the O sub-step and second
   drift half-step immediately below see its effect.

   ke_max is, by default (refresh_skin == 0, legacy behavior), the largest
   per-mode kinetic energy among this atom's currently-masked modes only
   (an atom with no masked modes -- soft, or already cooling down -- has
   ke_max 0 and never trips). This is a structural blind spot: a soft mode
   has no analytic tether and its curvature estimate (om[i]/evecs[i]) is
   frozen at whatever it was at the last refresh, so nothing watches it
   between refreshes even as it drifts. Stage 5's refresh_skin > 0 opt-in
   closes that gap two ways: (1) ke_max is computed over ALL modes, not
   just masked ones (ke_hot), and (2) independently, this atom is also
   flagged if it has drifted (minimum-imaged) farther than refresh_skin
   from x0[i] (its position at the last refresh -- drift_hot) --
   empirically the earlier and more useful signal, since a genuine runaway
   drift shows up as a growing displacement many steps before the kinetic
   energy itself spikes (which, for an unmollified soft mode heading into
   a close contact, tends to happen only on the same step as the resulting
   force blowup, too late to act on).

   The two triggers get DIFFERENT responses, confirmed necessary
   empirically: a first implementation reused demote_atom()'s full
   unmask+exile for both, and that made drift_hot crash *worse* (step 4-5,
   vs. the original bug's step 21) than doing nothing at all. The reason:
   by the time ke_hot fires, the dangerous close encounter has (per its own
   raw-force spike) already happened this step, so releasing the tether for
   the rest of the step is harmless -- but drift_hot fires *before* any
   force spike, while the atom is still approaching a close contact, and
   releasing its tether right then lets it drift further in, unmollified,
   for the remainder of THIS SAME step, exploding the very next kick on the
   real MACE force. So ke_hot keeps the original demote_atom() unmask+exile
   response (danger already passed, cool the atom down and let the next
   scheduled refresh re-evaluate it); drift_hot instead does an immediate,
   single-atom, out-of-cadence curvature refresh at this atom's CURRENT
   position (refresh_one_atom(), instantaneous=true -- still useful going
   forward, since it updates c[i]/x0[i]/om[i]/evecs[i] for how this atom is
   treated starting next step) followed by LOCAL SUB-STEPPING
   (sub_step_free_atom()) of this atom's remaining dtby2 of physical time
   as plain Cartesian velocity-Verlet at a much smaller, locally
   Verlet-safe dt, using real single-atom forces
   (partial_force(..., live_neighbor_positions=true)) -- NOT by
   re-projecting the refreshed q1[]/p1[] into the new eigenbasis and
   handing the atom back to the shared per-atom pipeline (O sub-step +
   analytic second-half drift) at the full production dt. Two earlier
   attempts tried exactly that hand-back -- with and without also forcing
   need_refresh for the next step's dt ladder -- and both crashed at the
   identical step with a bit-identical thermo trace to the original
   unmask+exile design: the freshly-probed curvature at a genuine close
   encounter is essentially always ineligible for masking
   (mode_is_stiff()'s all-3-modes floor, and lj_longstep_prototype.py's
   matching "eligible" gate/comment: masking a mid-collision atom in only
   one direction "was observed to destabilize the run"), so re-projecting
   into the shared pipeline just hands the atom back to the SAME
   free-particle-at-10-fs treatment that the very first falsified design
   (full-unmask during heff_samples accumulation) already showed is
   unconditionally unstable. Sub-stepping that same "free particle, no
   thermostat" treatment (an ordinary unmasked, non-quench atom already
   gets zero O-step friction/noise -- see the "else p2[m] = p1[m];" branch
   in initial_integrate()) at a safe local dt instead of the full
   production dt is a discretization fix, not a physics change. Because
   atom->x[i] itself is not updated until the very end of
   initial_integrate()'s per-atom loop, the sub-stepping's own starting
   point is reconstructed from q1[]/p1[] (this atom's already-computed
   post-first-half-drift state) projected back to Cartesian around the OLD
   center/eigenbasis; the fresh refresh_one_atom() center is placed to
   coincide with that point exactly. The caller must skip the rest of
   initial_integrate()'s per-atom pipeline for this atom this step once
   sub_stepped comes back true, since atom->x[i]/atom->v[i] are already
   fully advanced.

   "sudden" (a large jump in ke_max since the last step) and the strong-
   friction quench_exempt path stay tied to ke_hot only -- drift_hot is a
   preemptive, still-tethered correction, not an exile, so it does not
   itself request a quench kick.

   Separately (regardless of whether this call's own guard trips), reports
   via quench_active whether this atom should receive the strong-friction
   quench kick in the O step: it must currently be exiled (demote_cool in
   the future), not exempt, and not demoted on THIS exact step -- an atom
   demoted this step already got its energy dealt with by the demotion
   itself (mask override) and should not also take a quench kick the same
   step it happened, mirroring the reference's exclusion of its currently-
   eligible atom set from the same-step quench pass.

   Attempt 9 fix (event_log-diagnosed): ke_hot re-tripping on a
   CONSECUTIVE step, while this atom is still inside an already-active
   demote_cool window from a PRIOR trip, must NOT call demote_atom()
   again. demote_atom() unconditionally resets last_demote[i] =
   ntimestep, and quench_active's own exclusion above
   (last_demote[i] != ntimestep) then reads as "just demoted, skip the
   quench" -- correct for the atom's first trip into a cooldown episode,
   but if ke_hot keeps tripping every single step (empirically the
   common case for a genuinely hot light atom under MACE, per the
   event_log diagnostic: median/min gap between consecutive DEMOTEs was
   1 step for the atoms that triggered this bug), re-demoting every step
   keeps last_demote[i] pinned at ntimestep forever, so quench_active is
   permanently false and the atom's only real cooling mechanism (the
   O-step quench kick) never actually engages -- it is left unmasked
   (no analytic tether) AND unthermostatted (no quench friction) for as
   long as it stays hot, with nothing acting to bring it back down. Only
   call demote_atom() -- which logs a new DEMOTE, resets last_demote[i],
   and extends demote_cool[i] -- when this is a genuinely NEW episode
   (demote_cool[i] <= ntimestep, i.e. not already exiled from a prior
   trip); a re-trip while still inside an existing cooldown window keeps
   msk[] forced false (redundant but harmless -- it is already false)
   and otherwise leaves demote_cool[i]/last_demote[i] untouched, so
   quench_active can engage on this and subsequent hot-but-not-freshly-
   demoted steps exactly as originally intended.
------------------------------------------------------------------------- */

void FixBAOABTether::kinetic_guard_and_quench(int i, double mi, double kT, bool *msk,
                                               double *q1, double *p1, bool &quench_active,
                                               bool &sub_stepped)
{
  bigint ntimestep = update->ntimestep;
  sub_stepped = false;

  double ke_max = 0.0;
  for (int m = 0; m < 3; m++) {
    if (!msk[m] && refresh_skin <= 0.0) continue;
    double ke = 0.5 * force->mvv2e * mi * p1[m] * p1[m];
    if (ke > ke_max) ke_max = ke;
  }

  // kT here is boltz*t_target/mvv2e (native mass-velocity^2 units, matching
  // p1[]/invmass elsewhere in this fix), NOT an energy in eV. ke_max above
  // is already multiplied by mvv2e, i.e. in eV -- both caps below need the
  // same mvv2e factor to compare like with like (mirrors sub_step_free_atom()'s
  // ke_cap).
  double ke_cap = ke_rel * kT * force->mvv2e;
  bool ke_hot = ke_max > ke_cap;

  bool drift_hot = false;
  double drift_mag = -1.0;
  if (!ke_hot && refresh_skin > 0.0) {
    double **x = atom->x;
    double dx0[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};
    domain->minimum_image(FLERR, dx0);
    double drift2 = dx0[0] * dx0[0] + dx0[1] * dx0[1] + dx0[2] * dx0[2];
    if (drift2 > refresh_skin * refresh_skin) {
      drift_hot = true;
      drift_mag = sqrt(drift2);
    }
  }

  bool sudden = (ke_max - last_ke[i]) > 0.5 * ke_cap;
  last_ke[i] = ke_max;

  if (ke_hot) {
    for (int m = 0; m < 3; m++) msk[m] = false;
    log_event("GUARD_KE", atom->tag[i], ke_max, ke_cap, -1.0, -1.0, nullptr, -1.0,
              sudden ? "sudden" : "");
    n_kinetic_guard++;
    // Only start a NEW demotion episode (and its cooldown/last_demote
    // reset) if this atom isn't already inside one -- see the Attempt 9
    // doc comment above. A re-trip during an existing episode leaves
    // demote_cool[i]/last_demote[i] alone so quench_active can engage.
    if (demote_cool[i] <= ntimestep) demote_atom(i, 3 * (bigint) refresh_every, "kinetic_guard");
    if (sudden) quench_exempt[i] = demote_cool[i];
  } else if (drift_hot) {
    log_event("GUARD_DRIFT", atom->tag[i], -1.0, -1.0, drift_mag, refresh_skin, nullptr, -1.0,
              drift_response == 0 ? "retether" : "preshrink");
    if (drift_response == 0) {
      // retether (default): Do NOT re-project v_mid into the new eigenbasis
      // and hand this atom back to the shared pipeline (O sub-step +
      // second-half drift) below -- both falsified predecessors of this
      // design did exactly that, and both crashed identically to the
      // original unmask+exile bug (see the doc comment above). Instead,
      // locally sub-step this atom's own remaining dtby2 of physical time
      // as plain Cartesian velocity-Verlet, writing directly into
      // atom->x[i]/atom->v[i], and tell the caller to skip the rest of
      // this step's shared pipeline for this atom.
      double dx_mid[3] = {0.0, 0.0, 0.0}, v_mid[3] = {0.0, 0.0, 0.0};
      for (int m = 0; m < 3; m++)
        for (int k = 0; k < 3; k++) {
          dx_mid[k] += evecs[i][3 * m + k] * q1[m];
          v_mid[k] += evecs[i][3 * m + k] * p1[m];
        }
      double x_mid[3] = {c[i][0] + dx_mid[0], c[i][1] + dx_mid[1], c[i][2] + dx_mid[2]};

      refresh_one_atom(i, x_mid, true);
      n_kinetic_guard++;

      sub_step_free_atom(i, mi, x_mid, v_mid, dtby2, kT);
      sub_stepped = true;

      // Force the next initial_integrate() to run a full refresh_blocks()
      // (same rank-local-trigger pattern as demote_atom() above; the
      // MPI_Allreduce(MAX) at that call site keeps all ranks entering it
      // together), so update_dt() re-evaluates om_unres with this atom's
      // newly-probed om[i] included and the dt ladder can drop before the
      // next close encounter -- independent of, and in addition to, the
      // same-step protection sub_step_free_atom() just gave this one atom.
      need_refresh = 1;
    }
    // else: preshrink -- take NO action at all here, not even the
    // refresh_one_atom() bookkeeping call an earlier revision of this
    // branch made unconditionally. refresh_one_atom(instantaneous=true)
    // overwrites c[i]/x0[i]/om[i]/evecs[i] IN PLACE (see its tail), but
    // q1[]/p1[] are this atom's modal coordinates in the OLD eigenbasis
    // relative to the OLD center -- they do not automatically stay valid
    // just because nobody reprojects them. Calling refresh_one_atom() here
    // without also reprojecting q1[]/p1[] (which is exactly the sub-
    // stepping-free "re-tether in place" design already falsified above)
    // leaves the rest of THIS step's shared pipeline reconstructing
    // Cartesian state from stale modal coefficients against a brand-new
    // center/basis -- confirmed empirically to produce "Non-numeric atom
    // coords" within 2 steps, a worse failure than any prior attempt.
    // Under preshrink this atom must finish the CURRENT step completely
    // unperturbed -- q1[]/p1[]/msk[]/c[i]/x0[i]/evecs[i] all exactly as
    // they were before this call, as if drift_hot had never fired.
    // end_of_step() (below) independently re-checks the same |x[i]-x0[i]|
    // drift AFTER the full step completes, using the atom's genuinely
    // final position and its OWN refresh_one_atom() call, and shrinks
    // update->dt for the NEXT step if still warranted -- that is the only
    // place preshrink is allowed to act.
  }

  quench_active = (demote_cool[i] > ntimestep) && (quench_exempt[i] <= ntimestep) &&
                  (last_demote[i] != ntimestep);
}

/* ----------------------------------------------------------------------
   drift_hot response (Stage 5, adapt=yes && refresh_skin>0 only): integrate
   this ONE atom's remaining dtby2 of physical time as plain Cartesian
   velocity-Verlet, sub-cycled at a locally Verlet-safe dt_sub, using real
   single-atom forces from partial_force() (the same halo-local real-force
   kernel already used by every FD curvature probe elsewhere in this fix).
   No thermostat is applied during this window -- see the doc comment on
   kinetic_guard_and_quench() for why that is not a new approximation (an
   ordinary unmasked, non-quench atom already gets zero O-step friction/
   noise for the remainder of the shared pipeline). Writes the result
   directly into atom->x[i]/atom->v[i]; the caller must not run the rest of
   initial_integrate()'s per-atom pipeline for this atom this step.

   dt_sub falls back to dt_min (the floor of the existing adaptive dt
   ladder, already trusted to be Verlet-stable) whenever the curvature
   refresh_one_atom() just probed is non-positive-definite in every
   direction (om_max_new == 0), which is the common case right at a close
   encounter -- see the falsified re-tether-in-place debug trace. n_sub is
   rounded up and dt_sub reduced so the remaining_dt window divides evenly,
   with no leftover fractional sub-step.

   Neighbor-skin safety cap (fixes the crash Attempt 4 -- pure sub-stepping,
   no cap -- was empirically falsified with): partial_force()'s own doc
   comment states it is "valid as long as |xtrial - x[i]| stays well inside
   the neighbor skin," because it sums only over list->firstneigh[i]/
   numneigh[i], the SAME neighbor topology fixed once at the top of this
   whole (production-dt) step -- not rebuilt per sub-step. A genuinely
   violent drift_hot excursion can carry this atom multiple Angstrom in a
   handful of sub-steps (observed: 2.6 A over 25 sub-steps, tag 112, step 4
   of /tmp/sub4_skin), enough to reach a real neighbor this cached list
   never included, which the local force sum is then structurally blind to
   -- the atom free-falls with no repulsive wall from it, and
   final_integrate()'s closing kick (using the real, but by then equally
   stale-listed, global force) detonates it and its true neighbors.
   Tracking cumulative displacement from x_mid and stopping the loop once
   it would exceed half the neighbor skin -- the exact same trust radius
   Neighbor::check_distance() already uses to decide when ordinary,
   non-sub-stepped dynamics needs a rebuild -- keeps every partial_force()
   call inside its documented validity region. The atom is left with
   whatever fraction of remaining_dt it reached (a small, one-time, local
   truncation of this one atom's clock this one step) and need_reneighbor
   is set so the true, fresh environment is picked up cleanly starting the
   very next step (see need_reneighbor's doc comment in the header and its
   Allreduce site at the end of initial_integrate()).

   Kinetic-energy sanity cap (the displacement cap alone is NOT sufficient --
   empirically falsified, see /tmp/sub4_skin/debug3_new.txt): a missing
   neighbor's repulsive wall does not need multiple Angstrom of travel to
   matter -- x_mid itself can already be within a fraction of an Angstrom of
   an unlisted neighbor's repulsive core (the om_max_new == 0 case above IS
   this situation: a non-positive-definite probe at x_mid is itself a
   transition-state/close-contact signature). The observed trace shows
   ke_max-equivalent kinetic energy for tag 112 already exceeding ke_rel*kT
   after the FIRST sub-step, and blowing past it by orders of magnitude
   (~20 eV vs. a ~0.3 eV cap) by sub-step 2, while cumulative displacement
   was still under 0.1 A -- an order of magnitude below the displacement
   cap's trip point. So this atom's own kinetic energy, checked every
   sub-step against the SAME ke_rel*kT over-excitation threshold
   kinetic_guard_and_quench() already uses to decide "is this atom too hot"
   (no new, arbitrary constant introduced), is the fast-tripping signal that
   actually catches the runaway before it compounds; the displacement cap
   above remains valuable in its own right (it is what request a real
   reneighboring), but is a slower, secondary safety net for genuine
   multi-Angstrom excursions, not the primary defense against this failure
   mode.
------------------------------------------------------------------------- */

void FixBAOABTether::sub_step_free_atom(int i, double mi, const double *x_mid,
                                         const double *v_mid, double remaining_dt, double kT)
{
  double om_max_new = MAX(om[i][0], MAX(om[i][1], om[i][2]));
  double dt_sub = (om_max_new > 0.0) ? MIN(dt_min, 2.2 / om_max_new) : dt_min;
  int n_sub = MAX(1, (int) ceil(remaining_dt / dt_sub));
  dt_sub = remaining_dt / n_sub;

  double safe_dispsq = 0.25 * neighbor->skin * neighbor->skin;    // (0.5*skin)^2
  // kT here is boltz*t_target/mvv2e (native mass-velocity^2 units, matching
  // p1[]/invmass elsewhere in this fix -- see kinetic_guard_and_quench()'s
  // O-step noise draw), NOT an energy in eV. ke below is computed the same
  // way ke_max is in kinetic_guard_and_quench() -- already multiplied by
  // mvv2e, i.e. in eV -- so the cap needs the same mvv2e factor to compare
  // like with like.
  double ke_cap = ke_rel * kT * force->mvv2e;

  double ftm2v = force->ftm2v;
  double invmass = 1.0 / mi;
  double xs[3] = {x_mid[0], x_mid[1], x_mid[2]};
  double vs[3] = {v_mid[0], v_mid[1], v_mid[2]};

  double fcur[3];
  partial_force(i, xs, fcur, nullptr, true);
  n_partial_force_calls++;

  int s;
  for (s = 0; s < n_sub; s++) {
    for (int k = 0; k < 3; k++) vs[k] += 0.5 * dt_sub * ftm2v * invmass * fcur[k];
    for (int k = 0; k < 3; k++) xs[k] += dt_sub * vs[k];

    double fnext[3];
    partial_force(i, xs, fnext, nullptr, true);
    n_partial_force_calls++;

    for (int k = 0; k < 3; k++) vs[k] += 0.5 * dt_sub * ftm2v * invmass * fnext[k];
    for (int k = 0; k < 3; k++) fcur[k] = fnext[k];

    double dispsq = (xs[0] - x_mid[0]) * (xs[0] - x_mid[0]) + (xs[1] - x_mid[1]) * (xs[1] - x_mid[1]) +
                    (xs[2] - x_mid[2]) * (xs[2] - x_mid[2]);
    double ke = 0.5 * force->mvv2e * mi * (vs[0] * vs[0] + vs[1] * vs[1] + vs[2] * vs[2]);
    bool disp_hot = dispsq > safe_dispsq;
    bool ke_hot_sub = ke > ke_cap;
    if (disp_hot || ke_hot_sub) {
      need_reneighbor = 1;
      s++;
      break;
    }
  }

  atom->x[i][0] = xs[0];
  atom->x[i][1] = xs[1];
  atom->x[i][2] = xs[2];
  atom->v[i][0] = vs[0];
  atom->v[i][1] = vs[1];
  atom->v[i][2] = vs[2];
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

  // one-time bootstrap (adapt=yes, before om_split has ever been set):
  // mode_is_stiff()'s per-mode mask below needs om_split, but om_split
  // (D.11.1's spectral-gap split) is itself derived from the full group's
  // eigenfrequency census -- a chicken-and-egg problem only on the very
  // first refresh, since afterward om_split is held fixed (see the header)
  // and every subsequent refresh's mask decision only needs THIS atom's own
  // just-recomputed om[i], not the atom loop order. Resolved by running a
  // throwaway FD-probe pass first, touching only om[]/evecs[] (no Hc/Jblk,
  // no group-membership eligibility use yet), to seed the one-shot split;
  // the main loop below then repeats the same probes (needed regardless, to
  // also build Hc/Jblk) with om_split now available. This doubles the
  // force-evaluation cost of exactly one refresh call, once per run. With
  // heff_samples > 0 this throwaway estimate is also, harmlessly, this
  // atom's "first sample" instantaneous om -- the main loop's own
  // first-sample branch immediately re-derives the identical value through
  // the Hacc accumulator right after, so state stays consistent.
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
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    refresh_one_atom(i, xi, false);
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
   per-atom curvature-block refresh (group atom i only): the FD-probe body
   factored out of refresh_blocks()'s scheduled loop so it can also be
   called out-of-cadence, mid-step, for a single atom (Stage 5's
   refresh_skin excursion-guard "re-tether in place" response -- see
   kinetic_guard_and_quench()). Purely local: partial_force() makes no
   comm calls and neither does anything below, so -- unlike refresh_blocks()
   itself, which wraps this loop with the one-time om_split bootstrap and
   the collective update_dt() call -- this function is safe to invoke from
   a single rank on a single atom without desynchronizing the others.

   xi is the Cartesian base point to probe around (the scheduled caller
   passes the atom's current atom->x[i]; the excursion guard instead passes
   this atom's post-first-half-drift mid-step position, since atom->x[i]
   itself is not updated until the end of initial_integrate()'s per-atom
   loop -- see there).

   instantaneous=true forces an immediate jacobi3 diagonalization of THIS
   probe's raw Hessian into the working om[i]/evecs[i], regardless of the
   heff_samples averaging window's state, and leaves Hacc[i]/
   n_hess_samples[i] untouched -- an emergency snapshot, not a scheduled
   sample. instantaneous=false (the scheduled path) is the original
   heff_samples-aware logic, unchanged: heff_samples==0 always
   diagonalizes; heff_samples>0 only re-diagonalizes at the first and last
   sample of the averaging window (see the inline comment below).
------------------------------------------------------------------------- */

void FixBAOABTether::refresh_one_atom(int i, const double *xi, bool instantaneous)
{
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;

  double mi = rmass ? rmass[i] : mass[type[i]];
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

  // thermally-averaged effective curvature (CLAUDE.md convention #3):
  // heff_samples==0 keeps the legacy instantaneous-every-refresh write
  // below unchanged. heff_samples>0 accumulates the raw matrix every
  // refresh cycle (direct port of set_effective_light_blocks(), which
  // freezes the raw Hessian block, not the diagonalized frequencies --
  // averaging the matrix is the SCHA-consistent operation across
  // snapshots with different local eigenbases from anharmonic wobble),
  // but only RE-DIAGONALIZES the working om[i]/evecs[i] -- the values
  // mode_is_stiff() actually masks against -- at the FIRST sample (an
  // ordinary instantaneous estimate, exactly like refresh #1 always was)
  // and once more when the window closes (the full N-sample average,
  // frozen forever after). Samples in between accumulate silently
  // without ever touching om[i]/evecs[i]. This keeps the atom
  // masked/tethered continuously through the whole window using a STABLE
  // working estimate (no per-refresh thrash -> no coherent unmask flip),
  // rather than running genuinely free dynamics on a stiff anharmonic
  // mode at the production timestep, which is unconditionally unstable
  // (confirmed empirically: forcing full-unmask during accumulation blew
  // up worse, by step 4, than the original per-refresh-thrash bug this
  // mechanism exists to fix). An instantaneous=true emergency re-tether
  // always takes this same first branch, bypassing the averaging window
  // entirely -- it needs the current local curvature right now, not a
  // stale mid-window estimate -- without ever touching Hacc/
  // n_hess_samples (that scheduled bookkeeping is left exactly as the next
  // ordinary refresh_blocks() call would find it).
  if (instantaneous || heff_samples == 0) {
    double eval[3], evec[3][3];
    MathEigen::jacobi3(Hs, eval, evec);
    for (int m = 0; m < 3; m++) {
      om[i][m] = sqrt(MAX(eval[m], 0.0));
      for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = evec[a][m];
    }
  } else if (n_hess_samples[i] < heff_samples) {
    bool first_sample = (n_hess_samples[i] == 0);
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) Hacc[i][3 * a + b] += Hs[a][b];
    n_hess_samples[i]++;
    bool window_closed = (n_hess_samples[i] == heff_samples);

    if (first_sample || window_closed) {
      double Havg[3][3];
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) Havg[a][b] = Hacc[i][3 * a + b] / n_hess_samples[i];

      double eval[3], evec[3][3];
      MathEigen::jacobi3(Havg, eval, evec);
      for (int m = 0; m < 3; m++) {
        om[i][m] = sqrt(MAX(eval[m], 0.0));
        for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = evec[a][m];
      }
    }
    // else: mid-window sample (2..N-1) -- accumulate only, om[i]/
    // evecs[i] (and hence the mask decision) untouched.
  }
  // else: already frozen (n_hess_samples[i] == heff_samples) -- om[i]/
  // evecs[i] keep their frozen values, no further write.

  log_event("REFRESH", atom->tag[i], -1.0, -1.0, -1.0, -1.0, om[i], -1.0,
            instantaneous ? "forced" : "scheduled");

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
    } else if (dt_c > 1.25 * rung_dt && dt_lvl > 0 && update->ntimestep >= dt_floor_until) {
      // dt_floor_until (Attempt 7 Part B, preshrink only): skip relax-up
      // while an end_of_step() emergency shrink's short backoff is still
      // active -- om_unres above is blind to which atom justified that
      // shrink, so without this guard a scheduled refresh right after it
      // could raise dt_lvl back down before the atom has actually cleared
      // the encounter.
      dt_lvl--;
    }
  }

  double dt_new = MAX(dt_min, MIN(dt_max, dt_max / pow(2.0, dt_lvl)));
  apply_new_dt(dt_new);
}

/* ----------------------------------------------------------------------
   single shared place that mutates update->dt (Attempt 7 Part B): factored
   out of update_dt() so end_of_step()'s emergency shrink reuses the exact
   same notification sequence (fix_dt_reset.cpp's own tail) instead of a
   second, drifting copy of it.
------------------------------------------------------------------------- */

void FixBAOABTether::apply_new_dt(double dt_new)
{
  if (dt_new == update->dt) return;
  update->update_time();
  update->dt = dt_new;
  update->dt_default = 0;
  if (utils::strmatch(update->integrate_style, "^respa")) update->integrate->reset_dt();
  if (force->pair) force->pair->reset_dt();
  for (const auto &ifix : modify->get_fix_list()) ifix->reset_dt();
  output->reset_dt();
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
      demote_atom(i, backoff, "solve_center");
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

  if (use_local_partial_force) {
    // Mirror the clamped-center substitution used by the single()-based
    // fallback below (real x[j] -> c[j] for a flagged neighbor, when
    // !live_neighbor_positions): local_partial_force() applies this itself,
    // internally, to every subgraph atom it reads (not just jlist), via the
    // clamp_groupbit/clamp_c pair. c[] is kept forward-comm'd/PBC-shifted to
    // match x[]'s image (see pack_forward_comm()/unpack_forward_comm()), so
    // it is safe to hand to the pair style for ghost indices too.
    int clamp_groupbit = live_neighbor_positions ? 0 : groupbit;
    double *const *clamp_c = live_neighbor_positions ? nullptr : (double *const *) c;
    return force->pair->local_partial_force(i, xtrial, jnum, jlist, fout, fneigh,
                                             clamp_groupbit, clamp_c);
  }

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
  memory->grow(Hacc, nmax, 9, "baoab/tether:Hacc");
  memory->grow(n_hess_samples, nmax, "baoab/tether:n_hess_samples");
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
  //
  // nJ must be zeroed here too: it is otherwise only ever written inside
  // refresh_blocks() (once a block has actually been computed for that
  // atom), so right after construction it holds whatever garbage malloc
  // returned. copy_arrays()/pack_exchange()/unpack_exchange() all loop
  // "for k < nJ[i]" over Jtag/Jblk -- and Atom::sort() can call
  // copy_arrays() from inside Verlet::setup(), before refresh_blocks() has
  // run for any atom. A garbage nJ[i] there is a real (and previously
  // undetected) out-of-bounds read/write, not a hypothetical one.
  for (int i = 0; i < nmax; i++) {
    demote_count[i] = 0;
    last_demote[i] = 0;
    demote_cool[i] = 0;
    quench_exempt[i] = 0;
    last_ke[i] = 0.0;
    nJ[i] = 0;
    for (int a = 0; a < 9; a++) Hacc[i][a] = 0.0;
    n_hess_samples[i] = 0;
  }
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int k = 0; k < 3; k++) x0[j][k] = x0[i][k];
  for (int k = 0; k < 3; k++) c[j][k] = c[i][k];
  for (int k = 0; k < 9; k++) evecs[j][k] = evecs[i][k];
  for (int k = 0; k < 3; k++) om[j][k] = om[i][k];
  for (int k = 0; k < 9; k++) Hacc[j][k] = Hacc[i][k];
  n_hess_samples[j] = n_hess_samples[i];

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
  for (int k = 0; k < 9; k++) buf[m++] = Hacc[i][k];
  buf[m++] = ubuf(n_hess_samples[i]).d;

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
  for (int k = 0; k < 9; k++) Hacc[nlocal][k] = buf[m++];
  n_hess_samples[nlocal] = (int) ubuf(buf[m++]).i;

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
