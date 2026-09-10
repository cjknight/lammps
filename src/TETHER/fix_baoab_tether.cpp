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
#include "error.h"
#include "force.h"
#include "improper.h"
#include "kspace.h"
#include "math_eigen.h"
#include "memory.h"
#include "modify.h"
#include "neighbor.h"
#include "pair.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>
#include <cstring>

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

}    // namespace

/* ---------------------------------------------------------------------- */

FixBAOABTether::FixBAOABTether(LAMMPS *lmp, int narg, char **arg) :
    FixBAOAB(lmp, check_base_args(lmp, narg), arg), x0(nullptr), evecs(nullptr), om(nullptr)
{
  theta = -1.0;
  refresh_every = -1;
  eps = 0.0;
  eps_was_set = 0;

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
    } else {
      error->all(FLERR, iarg, "Unknown fix baoab/tether keyword {}", arg[iarg]);
    }
  }

  if (theta <= 0.0) error->all(FLERR, "Fix baoab/tether theta must be > 0");
  if (refresh_every <= 0) error->all(FLERR, "Fix baoab/tether refresh must be > 0");
  if (eps_was_set && eps <= 0.0) error->all(FLERR, "Fix baoab/tether eps must be > 0");

  last_refresh_step = -1;
  need_refresh = 1;

  // per-atom persistent state (curvature-block eigenvectors/frequencies,
  // frozen center) -- exchanged/reordered like any other atom-based array,
  // but not restart-aware yet: a restart re-refreshes on the first
  // post-restart step rather than resuming mid-cycle (stage-1 limitation).
  FixBAOABTether::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
}

/* ---------------------------------------------------------------------- */

FixBAOABTether::~FixBAOABTether()
{
  if (copymode) return;

  atom->delete_callback(id, Atom::GROW);

  memory->destroy(x0);
  memory->destroy(evecs);
  memory->destroy(om);
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

  // force an immediate curvature-block acquisition on the first
  // initial_integrate() call of this run
  need_refresh = 1;
}

/* ---------------------------------------------------------------------- */

// B-A-O-A-B, generalizing the "A" (drift) and "O" (thermostat) sub-steps
// for group atoms: per-mode harmonic drift about a frozen center for
// locally stiff eigenmodes (omega*dt > theta), ordinary free drift and the
// inherited Cartesian-equivalent OU update for everything else. The B
// (kick) sub-steps and all non-group atoms are untouched -- identical to
// FixBAOAB::initial_integrate().

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

  if (need_refresh || (update->ntimestep - last_refresh_step) >= refresh_every) {
    refresh_blocks();
    last_refresh_step = update->ntimestep;
    need_refresh = 0;
  }

  double dt = update->dt;
  double kT = force->boltz * t_target / force->mvv2e;
  double one_minus_c1sq = 1.0 - c1 * c1;

  energy_onestep = 0.0;

  double fran_total[3] = {0.0, 0.0, 0.0};
  double mass_total = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double invmass = 1.0 / mi;

    // ---- B: half kick, masked modes get the harmonic-restoring
    // component of the raw force cancelled first (see
    // modal_force_correction()) so only the anharmonic residual is
    // kicked with -- the harmonic part is already integrated exactly by
    // the analytic drift below. Matches _force_moll()'s mollify=False
    // branch in lj_longstep_prototype.py; without this, a stiff mode's
    // huge raw restoring force is double-counted and the kick blows up.
    double fcorr[3];
    modal_force_correction(i, mi, fcorr);
    double dtfm = dtf * invmass;
    v[i][0] += dtfm * (f[i][0] + fcorr[0]);
    v[i][1] += dtfm * (f[i][1] + fcorr[1]);
    v[i][2] += dtfm * (f[i][2] + fcorr[2]);

    bool msk[3];
    for (int m = 0; m < 3; m++) msk[m] = (om[i][m] * dt) > theta;

    // project displacement-from-center and velocity into the frozen
    // eigenbasis; unmasked modes carry this round-trip through free
    // flight, so it is a no-op for them up to floating-point roundoff
    double dxr[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};
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
    // ---- O: exact OU update, masked modes only, same c1/c2 as FixBAOAB ----
    double c2 = sqrt(kT * invmass * one_minus_c1sq);
    double r[3] = {random->gaussian(), random->gaussian(), random->gaussian()};
    double p2[3];
    double ke_before = 0.0, ke_after = 0.0;
    for (int m = 0; m < 3; m++) ke_before += p1[m] * p1[m];
    for (int m = 0; m < 3; m++) p2[m] = msk[m] ? (c1 * p1[m] + c2 * r[m]) : p1[m];
    for (int m = 0; m < 3; m++) ke_after += p2[m] * p2[m];
    energy_onestep += 0.5 * force->mvv2e * mi * (ke_before - ke_after);

    if (zeroflag) {
      for (int k = 0; k < 3; k++) {
        double kick = 0.0;
        for (int m = 0; m < 3; m++)
          if (msk[m]) kick += evecs[i][3 * m + k] * c2 * r[m];
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

    x[i][0] = x0[i][0] + dxnew[0];
    x[i][1] = x0[i][1] + dxnew[1];
    x[i][2] = x0[i][2] + dxnew[2];
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
   inherited from FixBAOAB) because it needs the same harmonic-restoring
   -force cancellation as the opening half kick in initial_integrate():
   the frozen center/curvature is unchanged across this step's substeps,
   so the same correction applies here for group atoms.
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

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double dtfm = dtf / mi;

    double fcorr[3];
    modal_force_correction(i, mi, fcorr);
    v[i][0] += dtfm * (f[i][0] + fcorr[0]);
    v[i][1] += dtfm * (f[i][1] + fcorr[1]);
    v[i][2] += dtfm * (f[i][2] + fcorr[2]);
  }
}

/* ----------------------------------------------------------------------
   Cartesian correction added to the raw force before any half kick on a
   group atom's masked (locally stiff) modes: cancels that mode's
   harmonic restoring-force component, m*omega^2*(displacement from the
   frozen center), so the impulse carries only the anharmonic residual --
   the harmonic part is already integrated exactly by the analytic drift.
   Mirrors lj_longstep_prototype.py's FrozenHessianIntegrator._force_moll()
   mollify=False branch (g[s] += from_modal(om_eff^2 * dlt)), converted
   from that mass-weighted form to a plain Cartesian force by the factor
   of mi. Without this, a stiff mode's huge raw restoring force is
   double-counted between the kick and the drift and the kick blows up.
------------------------------------------------------------------------- */

void FixBAOABTether::modal_force_correction(int i, double mi, double *fcorr) const
{
  double **x = atom->x;
  double dt = update->dt;
  double dxr[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};

  fcorr[0] = fcorr[1] = fcorr[2] = 0.0;
  for (int m = 0; m < 3; m++) {
    if (om[i][m] * dt <= theta) continue;
    double q0m = 0.0;
    for (int k = 0; k < 3; k++) q0m += evecs[i][3 * m + k] * dxr[k];
    double coeff = mi * om[i][m] * om[i][m] * q0m;
    for (int k = 0; k < 3; k++) fcorr[k] += evecs[i][3 * m + k] * coeff;
  }
}

/* ----------------------------------------------------------------------
   finite-difference curvature-block acquisition (group atoms only):
   for each atom, 3 central-difference force probes (+-eps along x/y/z)
   give a 3x3 block; symmetrize, mass-weight, diagonalize via jacobi3,
   and reset the frozen center to the atom's current position.

   Mirrors Min::energy_force()'s force-recompute sequence but skips
   neighbor->decide()/rebuild/exchange/borders: eps is small relative to
   the neighbor skin, so the existing neighbor list stays valid for the
   perturbed probe positions (documented stage-1 assumption).
------------------------------------------------------------------------- */

void FixBAOABTether::refresh_blocks()
{
  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // the probes below repeatedly clobber atom->f; stash the production
  // forces so the B-step immediately following this call sees the same
  // f[] it would have without any curvature refresh happening at all
  double **f_saved;
  memory->create(f_saved, MAX(atom->nmax, 1), 3, "baoab/tether:f_saved");
  for (int i = 0; i < nlocal; i++) {
    f_saved[i][0] = f[i][0];
    f_saved[i][1] = f[i][1];
    f_saved[i][2] = f[i][2];
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    double H[3][3];

    for (int b = 0; b < 3; b++) {
      double fplus[3], fminus[3];

      x[i][b] = xi[b] + eps;
      comm->forward_comm();
      recompute_forces_local();
      fplus[0] = f[i][0];
      fplus[1] = f[i][1];
      fplus[2] = f[i][2];

      x[i][b] = xi[b] - eps;
      comm->forward_comm();
      recompute_forces_local();
      fminus[0] = f[i][0];
      fminus[1] = f[i][1];
      fminus[2] = f[i][2];

      x[i][b] = xi[b];

      for (int a = 0; a < 3; a++) H[a][b] = -(fplus[a] - fminus[a]) / (2.0 * eps);
    }
    comm->forward_comm();    // restore atom i's ghost images to xi

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
  }

  for (int i = 0; i < nlocal; i++) {
    f[i][0] = f_saved[i][0];
    f[i][1] = f_saved[i][1];
    f[i][2] = f_saved[i][2];
  }
  memory->destroy(f_saved);
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
  return (double) atom->nmax * 15 * sizeof(double);
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::grow_arrays(int nmax)
{
  memory->grow(x0, nmax, 3, "baoab/tether:x0");
  memory->grow(evecs, nmax, 9, "baoab/tether:evecs");
  memory->grow(om, nmax, 3, "baoab/tether:om");
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::copy_arrays(int i, int j, int /*delflag*/)
{
  for (int k = 0; k < 3; k++) x0[j][k] = x0[i][k];
  for (int k = 0; k < 9; k++) evecs[j][k] = evecs[i][k];
  for (int k = 0; k < 3; k++) om[j][k] = om[i][k];
}

/* ---------------------------------------------------------------------- */

int FixBAOABTether::pack_exchange(int i, double *buf)
{
  int m = 0;
  for (int k = 0; k < 3; k++) buf[m++] = x0[i][k];
  for (int k = 0; k < 9; k++) buf[m++] = evecs[i][k];
  for (int k = 0; k < 3; k++) buf[m++] = om[i][k];
  return m;
}

/* ---------------------------------------------------------------------- */

int FixBAOABTether::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  for (int k = 0; k < 3; k++) x0[nlocal][k] = buf[m++];
  for (int k = 0; k < 9; k++) evecs[nlocal][k] = buf[m++];
  for (int k = 0; k < 3; k++) om[nlocal][k] = buf[m++];
  return m;
}
