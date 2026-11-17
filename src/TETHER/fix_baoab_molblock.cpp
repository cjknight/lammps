/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_baoab_molblock.h"

#include "accelerator_kokkos.h"
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
#include "math_eigen_impl.h"
#include "math_const.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "output.h"
#include "pair.h"
#include "platform.h"
#include "random_mars.h"
#include "update.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixBAOABMolblock::FixBAOABMolblock(LAMMPS *lmp, int narg, char **arg) :
    FixBAOABTetherBase(lmp, narg, arg, "Fix baoab/molblock"), mb_state(nullptr), mb_anc(nullptr)
{
  theta = -1.0;
  refresh_every = -1;
  heff_samples = 0;
  gamma_soft = 0.0;
  gamma_soft_was_set = 0;
  mb_verbose = 0;
  mb_inner_user = 0;
  mb_nin = 1;
  mb_wmax = 0.0;
  mb_last_refresh = -1;
  mb_step = 0;
  mb_nicmax = mb_ss = mb_bpa = mb_apa = mb_commflag = mb_probe_size = mb_nmax = 0;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "theta") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock theta", error);
      theta = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "refresh") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock refresh", error);
      refresh_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "heff_samples") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock heff_samples", error);
      heff_samples = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (heff_samples < 0)
        error->all(FLERR, iarg + 1, "Fix baoab/molblock heff_samples must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_soft") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock gamma_soft", error);
      gamma_soft = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (gamma_soft < 0.0) error->all(FLERR, iarg + 1, "Fix baoab/molblock gamma_soft must be >= 0");
      gamma_soft_was_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "eps") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock eps", error);
      eps = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (eps <= 0.0) error->all(FLERR, iarg + 1, "Fix baoab/molblock eps must be > 0");
      eps_was_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "inner") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock inner", error);
      mb_inner_user = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (mb_inner_user < 0) error->all(FLERR, iarg + 1, "Fix baoab/molblock inner must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "verbose") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/molblock verbose", error);
      mb_verbose = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown fix baoab/molblock keyword {}", arg[iarg]);
    }
  }
  if (theta <= 0.0) error->all(FLERR, "Fix baoab/molblock theta must be given and > 0");
  if (refresh_every <= 0) error->all(FLERR, "Fix baoab/molblock refresh must be given and > 0");
  if (atom->molecular != Atom::MOLECULAR)
    error->all(FLERR, "Fix baoab/molblock requires a molecular atom style with bonds");

  // molblock per-atom molecule state, sized for the largest set of bonds and angles any one atom
  // takes part in, counting each angle twice (a near-linear angle becomes two bend coordinates; a
  // molecule with more is rejected at its first refresh)
  {
    int nicmax = 0;
    for (int i = 0; i < atom->nlocal; i++) {
      int n = atom->num_bond ? atom->num_bond[i] : 0;
      if (atom->num_angle) n += 2 * atom->num_angle[i];
      nicmax = MAX(nicmax, n);
    }
    MPI_Allreduce(&nicmax, &mb_nicmax, 1, MPI_INT, MPI_MAX, world);
    mb_nicmax = MAX(1, MIN(mb_nicmax, MBN));
    mb_ss = 1 + 8 + 2 * MBA + 8 * mb_nicmax + 6 * mb_nicmax * mb_nicmax;
    mb_bpa = MAX(1, atom->bond_per_atom);
    mb_apa = atom->num_angle ? MAX(1, atom->angle_per_atom) : 0;
    const int MBMAX = 4;
    mb_probe_size = 9 + MBMAX * 10 + 1 + 1 + mb_bpa + 1 + 3 * mb_apa;
    comm_forward = MAX(comm_forward, MAX(6, mb_probe_size));
    comm_border = mb_ss;
    maxexchange += 1 + mb_ss;
  }

  FixBAOABMolblock::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::BORDER);
}

/* ---------------------------------------------------------------------- */

FixBAOABMolblock::~FixBAOABMolblock()
{
  if (copymode) return;
  atom->delete_callback(id, Atom::GROW);
  atom->delete_callback(id, Atom::BORDER);
  memory->destroy(mb_state);
  memory->destroy(mb_anc);
}

/* ---------------------------------------------------------------------- */

void FixBAOABMolblock::init()
{
  FixBAOAB::init();

  // the frictions must act on a timescale longer than one step (damp and gamma_soft are in the
  // run's time units, so a value copied between unit systems is off by orders of magnitude);
  // gamma_soft defaults to 1/damp
  if (!gamma_soft_was_set) gamma_soft = gamma;
  double dt = update->dt;
  if (gamma * dt > 1.0)
    error->all(FLERR, Error::NOLASTLINE, "Fix baoab/molblock damp {} is shorter than one timestep "
               "({}): the thermostat would re-randomize velocities every step. damp is in the "
               "run's time units; a typical choice is about 100 times the timestep, here {}",
               1.0 / gamma, dt, 100.0 * dt);
  if (gamma_soft * dt > 1.0)
    error->all(FLERR, Error::NOLASTLINE, "Fix baoab/molblock gamma_soft {} (1/time) exceeds 1 per "
               "timestep ({}): the thermostat would re-randomize velocities every step",
               gamma_soft, dt);

  // probe displacement: a small fraction of the neighbor skin unless given
  if (!eps_was_set) eps = 1.0e-4 * neighbor->skin;
  if (eps <= 0.0)
    error->all(FLERR, Error::NOLASTLINE, "Fix baoab/molblock eps must be > 0 (the neighbor skin "
               "is 0: set eps explicitly)");

  init_probe_kernels();
  // the list is read only by the curvature probes at a refresh, so it is occasional and built
  // on demand (rebuilding it with every pair list cost about as much as the pair list itself)
  neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
}

/* ---------------------------------------------------------------------- */

void FixBAOABMolblock::initial_integrate(int /*vflag*/)
{
  compute_target();
  mb_initial_integrate();
}

/* ---------------------------------------------------------------------- */

void FixBAOABMolblock::final_integrate()
{
  mb_final_integrate();
}

/* ---------------------------------------------------------------------- */

double FixBAOABMolblock::memory_usage()
{
  double bytes = (double) atom->nmax * (mb_ss * sizeof(double) + sizeof(tagint));
  bytes += (double) mbmols.capacity() * sizeof(MBMol);
  bytes += (double) (mb_Hown.capacity() + mb_X.capacity()) * sizeof(double);
  return bytes;
}

/* ---------------------------------------------------------------------- */

void FixBAOABMolblock::grow_arrays(int nmax)
{
  // molecule state must survive a mid-run grow (more ghosts): clear only the new rows
  memory->grow(mb_state, nmax, mb_ss, "baoab/molblock:mb_state");
  memory->grow(mb_anc, nmax, "baoab/molblock:mb_anc");
  for (int i = mb_nmax; i < nmax; i++) {
    mb_state[i][0] = 0.0;
    mb_anc[i] = 0;
  }
  mb_nmax = MAX(mb_nmax, nmax);
}

/* ---------------------------------------------------------------------- */

void FixBAOABMolblock::copy_arrays(int i, int j, int /*delflag*/)
{
  mb_anc[j] = mb_anc[i];
  if (mb_state[i][0] != 0.0)
    for (int k = 0; k < mb_ss; k++) mb_state[j][k] = mb_state[i][k];
  else
    mb_state[j][0] = 0.0;
}

/* ---------------------------------------------------------------------- */

int FixBAOABMolblock::pack_exchange(int i, double *buf)
{
  int m = 0;
  buf[m++] = ubuf(mb_anc[i]).d;
  if (mb_state[i][0] != 0.0)
    for (int k = 0; k < mb_ss; k++) buf[m++] = mb_state[i][k];
  else
    buf[m++] = 0.0;
  return m;
}

/* ---------------------------------------------------------------------- */

int FixBAOABMolblock::unpack_exchange(int nlocal, double *buf)
{
  int m = 0;
  mb_anc[nlocal] = (tagint) ubuf(buf[m++]).i;
  mb_state[nlocal][0] = buf[m++];
  if (mb_state[nlocal][0] != 0.0)
    for (int k = 1; k < mb_ss; k++) mb_state[nlocal][k] = buf[m++];
  return m;
}

/* ----------------------------------------------------------------------
   forward comm: velocities and forces (once per outer step), or the refresh probe data
------------------------------------------------------------------------- */

int FixBAOABMolblock::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                        int * /*pbc*/)
{
  int m = 0;
  if (mb_commflag == 1) {
    for (int i = 0; i < n; i++) {
      int j = list[i];
      for (int k = 0; k < 3; k++) buf[m++] = atom->v[j][k];
      for (int k = 0; k < 3; k++) buf[m++] = atom->f[j][k];
    }
    return m;
  }
  if (mb_commflag == 2) {
    const int MBMAX = 4;
    for (int i = 0; i < n; i++) {
      int j = list[i];
      for (int k = 0; k < 9; k++) buf[m++] = mb_Hown[(size_t) j * 9 + k];
      for (int k = 0; k < MBMAX * 9; k++) buf[m++] = mb_X[(size_t) j * MBMAX * 9 + k];
      for (int k = 0; k < MBMAX; k++) buf[m++] = ubuf(mb_tag[(size_t) j * MBMAX + k]).d;
      buf[m++] = mb_n[j];
      buf[m++] = mb_nbt[j];
      for (int k = 0; k < mb_bpa; k++) buf[m++] = ubuf(mb_btop[(size_t) j * mb_bpa + k]).d;
      buf[m++] = mb_nat[j];
      for (int k = 0; k < 3 * mb_apa; k++) buf[m++] = ubuf(mb_atop[(size_t) j * 3 * mb_apa + k]).d;
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBAOABMolblock::unpack_forward_comm(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  if (mb_commflag == 1) {
    for (int i = first; i < last; i++) {
      for (int k = 0; k < 3; k++) atom->v[i][k] = buf[m++];
      for (int k = 0; k < 3; k++) atom->f[i][k] = buf[m++];
    }
    return;
  }
  if (mb_commflag == 2) {
    const int MBMAX = 4;
    for (int i = first; i < last; i++) {
      for (int k = 0; k < 9; k++) mb_Hown[(size_t) i * 9 + k] = buf[m++];
      for (int k = 0; k < MBMAX * 9; k++) mb_X[(size_t) i * MBMAX * 9 + k] = buf[m++];
      for (int k = 0; k < MBMAX; k++) mb_tag[(size_t) i * MBMAX + k] = (tagint) ubuf(buf[m++]).i;
      mb_n[i] = (int) buf[m++];
      mb_nbt[i] = (int) buf[m++];
      for (int k = 0; k < mb_bpa; k++) mb_btop[(size_t) i * mb_bpa + k] = (tagint) ubuf(buf[m++]).i;
      mb_nat[i] = (int) buf[m++];
      for (int k = 0; k < 3 * mb_apa; k++)
        mb_atop[(size_t) i * 3 * mb_apa + k] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ----------------------------------------------------------------------
   ghost copies of anchor atoms get the owner's molecule state at
   every reneighboring (state of other atoms: a single 0 flag)
------------------------------------------------------------------------- */

int FixBAOABMolblock::pack_border(int n, int *list, double *buf)
{
  int m = 0;
  for (int i = 0; i < n; i++) {
    int j = list[i];
    if (mb_state[j][0] != 0.0)
      for (int k = 0; k < mb_ss; k++) buf[m++] = mb_state[j][k];
    else
      buf[m++] = 0.0;
  }
  return m;
}

/* ---------------------------------------------------------------------- */

int FixBAOABMolblock::unpack_border(int n, int first, double *buf)
{
  int m = 0;
  int last = first + n;
  for (int i = first; i < last; i++) {
    mb_state[i][0] = buf[m++];
    if (mb_state[i][0] != 0.0)
      for (int k = 1; k < mb_ss; k++) mb_state[i][k] = buf[m++];
    mb_anc[i] = 0;
  }
  return m;
}

/* ----------------------------------------------------------------------
   molblock (auto-RESPA): impulse multiple time stepping for the stiff
   internal motion of small bonded molecules, with the fast force field
   built from the measured Hessian instead of from a force-field split.

   A molecule is an atom plus its bond/angle partners (at most MBA atoms),
   owned by its lowest-tag atom; all its atoms must be local (serial or
   whole-molecule-per-rank). At each refresh (mb_refresh):
   - assemble the molecular Hessian from the per-atom own/cross blocks
     refresh_one_atom() measures (same collection as molblock_diag);
   - if any internal mode (span of the molecule's bond/angle gradients) has
     omega*dt > theta, all its bonds and angles form the fast set s;
   - Wilson GF force constants F of s from that Hessian and the raw forces
     (mb_gf_forceconst); the fastest GF frequency sets the substep count.
   The fast potential is V_fast = 1/2 (s - s0)^T F (s - s0), with the
   reference s0 anchored once at the molecule's first Newton center and kept
   for the rest of the run. Each outer step (consistent splitting):
     B_slow(dt/2) [ B_fast(h/2) A(h) B_fast(h/2) ]^n (O over dt mid-loop) B_slow(dt/2)
   with slow force f - f_fast. When F is refreshed, the step in progress
   keeps its old F through its closing kick and the new F takes over at the
   start of the next step, so every outer step uses a single split.
   Changes nothing unless "molblock yes" is given.
------------------------------------------------------------------------- */

// local indices of molecule M's atoms (owned, else the ghost image nearest an owned member) and
// their positions relative to its COM (minimum image); false if no member is owned or one is missing
bool FixBAOABMolblock::mb_locate(const MBMol &M, int *loc, double (*xr)[3]) const
{
  double **x = atom->x;
  int nlocal = atom->nlocal;
  int iref = -1;
  for (int a = 0; a < M.n; a++) {
    loc[a] = atom->map(M.tag[a]);
    if (loc[a] < 0) return false;
    if ((loc[a] < nlocal) && (iref < 0)) iref = loc[a];
  }
  if (iref < 0) return false;
  double xa[MBA][3];
  for (int a = 0; a < M.n; a++) {
    if (loc[a] >= nlocal) loc[a] = domain->closest_image(iref, loc[a]);
    for (int k = 0; k < 3; k++) xa[a][k] = x[loc[a]][k];
  }
  mb_rel(M, xa, xr);
  return true;
}

// positions xr relative to the COM of molecule M from its atoms' absolute positions xa
void FixBAOABMolblock::mb_rel(const MBMol &M, const double (*xa)[3], double (*xr)[3]) const
{
  double com[3] = {0, 0, 0}, mt = 0.0;
  for (int a = 0; a < M.n; a++) {
    double d[3] = {xa[a][0] - xa[0][0], xa[a][1] - xa[0][1], xa[a][2] - xa[0][2]};
    domain->minimum_image(FLERR, d);
    for (int k = 0; k < 3; k++) {
      xr[a][k] = d[k];
      com[k] += M.m[a] * d[k];
    }
    mt += M.m[a];
  }
  for (int a = 0; a < M.n; a++)
    for (int k = 0; k < 3; k++) xr[a][k] -= com[k] / mt;
}

// persistent molecule state <-> flat per-atom buffer (layout fixed by mb_nicmax)
int FixBAOABMolblock::mb_pack(const MBMol &M, double *buf) const
{
  int m = 0, nc = M.nic, L = mb_nicmax;
  buf[m++] = 1.0;
  buf[m++] = M.n;
  buf[m++] = nc;
  buf[m++] = M.nm;
  buf[m++] = M.have_s0 ? 1.0 : 0.0;
  buf[m++] = M.pending ? 1.0 : 0.0;
  buf[m++] = M.nd;
  buf[m++] = M.hprev ? 1.0 : 0.0;
  buf[m++] = M.has_mom ? 1.0 : 0.0;
  for (int a = 0; a < MBA; a++) buf[m++] = (a < M.n) ? ubuf(M.tag[a]).d : 0.0;
  for (int a = 0; a < MBA; a++) buf[m++] = (a < M.n) ? M.m[a] : 0.0;
  for (int q = 0; q < L; q++) {
    bool in = q < nc;
    buf[m++] = in ? M.ic_type[q] : 0;
    for (int k = 0; k < 3; k++) buf[m++] = in ? M.ic_at[q][k] : 0;
    buf[m++] = in ? M.sig0[q] : 0.0;
    buf[m++] = in ? M.sprev[q] : 0.0;
    buf[m++] = in ? M.pprev[q] : 0.0;
    buf[m++] = (q < M.nm) ? M.w[q] : 0.0;
  }
  const double(*mats[6])[MBN] = {M.Fic, M.Fold, M.dss, M.dps, M.Sss, M.Sps};
  for (auto mat : mats)
    for (int q = 0; q < L; q++)
      for (int r = 0; r < L; r++) buf[m++] = ((q < nc) && (r < nc)) ? mat[q][r] : 0.0;
  return m;
}

void FixBAOABMolblock::mb_unpack(MBMol &M, const double *buf) const
{
  int m = 1, L = mb_nicmax;
  M.n = (int) buf[m++];
  M.nic = (int) buf[m++];
  M.nm = (int) buf[m++];
  M.have_s0 = buf[m++] != 0.0;
  M.pending = buf[m++] != 0.0;
  M.nd = (int) buf[m++];
  M.hprev = buf[m++] != 0.0;
  M.has_mom = buf[m++] != 0.0;
  for (int a = 0; a < MBA; a++) M.tag[a] = (tagint) ubuf(buf[m++]).i;
  for (int a = 0; a < MBA; a++) M.m[a] = buf[m++];
  for (int q = 0; q < L; q++) {
    M.ic_type[q] = (int) buf[m++];
    for (int k = 0; k < 3; k++) M.ic_at[q][k] = (int) buf[m++];
    M.sig0[q] = buf[m++];
    M.sprev[q] = buf[m++];
    M.pprev[q] = buf[m++];
    M.w[q] = buf[m++];
  }
  double(*mats[6])[MBN] = {M.Fic, M.Fold, M.dss, M.dps, M.Sss, M.Sps};
  for (auto mat : mats)
    for (int q = 0; q < L; q++)
      for (int r = 0; r < L; r++) mat[q][r] = buf[m++];
  M.valid = false;
}

// this rank's molecule copies: one per molecule with an owned atom, state read from its anchor
void FixBAOABMolblock::mb_gather()
{
  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  mbmols.clear();
  mb_aidx.clear();
  // a ghost anchor's state is kept on its image nearest the molecule's lowest-ID owned atom (owned
  // atoms of one molecule can sit in different periodic images, so "nearest an owned atom" alone is
  // ambiguous); mb_refresh() uses the same rule
  std::unordered_map<tagint, int> first;
  std::vector<tagint> order;
  for (int i = 0; i < nlocal; i++) {
    tagint t = mb_anc[i];
    if (t == 0) continue;
    auto it = first.find(t);
    if (it == first.end()) {
      first[t] = i;
      order.push_back(t);
    } else if (tag[i] < tag[it->second]) {
      it->second = i;
    }
  }
  for (tagint t : order) {
    int i = first[t];
    int j = atom->map(t);
    if (j >= nlocal) j = domain->closest_image(i, j);
    if ((j < 0) || (mb_state[j][0] == 0.0))
      error->one(FLERR, "Fix baoab/molblock: the molecule of atom ID {} is not fully within "
                 "this processor's ghost atoms; increase the communication cutoff (comm_modify cutoff)",
                 tag[i]);
    MBMol M;
    mb_unpack(M, mb_state[j]);
    mbmols.push_back(M);
    mb_aidx.push_back(j);
  }
}

// write each copy's state back to its anchor (owned or ghost) and the anchor tag to owned members
void FixBAOABMolblock::mb_scatter()
{
  int nlocal = atom->nlocal;
  for (size_t k = 0; k < mbmols.size(); k++) {
    const MBMol &M = mbmols[k];
    mb_pack(M, mb_state[mb_aidx[k]]);
    for (int a = 0; a < M.n; a++) {
      int j = atom->map(M.tag[a]);
      if ((j >= 0) && (j < nlocal)) mb_anc[j] = M.tag[0];
    }
  }
}

// thermostat noise keyed by (seed, atom ID, step, component): every copy of a molecule draws the
// same numbers on every rank, and results do not depend on the number of processors
static inline uint64_t mb_mix(uint64_t z)
{
  z += 0x9e3779b97f4a7c15ULL;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

double FixBAOABMolblock::mb_gauss(tagint tag, int k) const
{
  uint64_t h = mb_mix(mb_mix(mb_mix((uint64_t) seed) ^ (uint64_t) tag) ^ (uint64_t) update->ntimestep);
  uint64_t a = mb_mix(h ^ (uint64_t) (2 * k + 1)), b = mb_mix(h ^ (uint64_t) (2 * k + 2));
  const double inv53 = 1.0 / 9007199254740992.0;
  double u1 = ((double) (a >> 11) + 0.5) * inv53;
  double u2 = (double) (b >> 11) * inv53;
  return sqrt(-2.0 * log(u1)) * cos(2.0 * MathConst::MY_PI * u2);
}

// small dense solve A x = b (Gaussian elimination, partial pivoting); b overwritten by x
template <int W> static bool mb_solve(int n, double (*A)[W], double *b)
{
  for (int k = 0; k < n; k++) {
    int p = k;
    for (int i = k + 1; i < n; i++)
      if (fabs(A[i][k]) > fabs(A[p][k])) p = i;
    if (fabs(A[p][k]) < 1.0e-300) return false;
    if (p != k) {
      for (int j = 0; j < n; j++) std::swap(A[k][j], A[p][j]);
      std::swap(b[k], b[p]);
    }
    for (int i = k + 1; i < n; i++) {
      double f = A[i][k] / A[k][k];
      for (int j = k; j < n; j++) A[i][j] -= f * A[k][j];
      b[i] -= f * b[k];
    }
  }
  for (int k = n - 1; k >= 0; k--) {
    double s = b[k];
    for (int j = k + 1; j < n; j++) s -= A[k][j] * b[j];
    b[k] = s / A[k][k];
  }
  return true;
}

static int mb_lin_axis(const double (*xr)[3], int a, int c, double *e);

void FixBAOABMolblock::mb_refresh()
{
  const int MBMAX = 4;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double **x = atom->x;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;

  // probes of the owned atoms, then the probe data and bond/angle lists of all atoms to the ghosts
  int nall = nlocal + atom->nghost;
  mb_Hown.assign((size_t) nall * 9, 0.0);
  mb_X.assign((size_t) nall * MBMAX * 9, 0.0);
  mb_tag.assign((size_t) nall * MBMAX, 0);
  mb_n.assign(nall, 0);
  mb_btop.assign((size_t) nall * mb_bpa, 0);
  mb_atop.assign((size_t) nall * 3 * mb_apa, 0);
  mb_nbt.assign(nall, 0);
  mb_nat.assign(nall, 0);
  if (list) neighbor->build_one(list);    // occasional under molblock, see init()
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    double H[3][3];
    int jnum, nb;
    int *jlist;
    std::vector<double> Hc, Hcb;
    std::vector<tagint> bt;
    probe_atom(i, xi, true, true, H, jnum, jlist, Hc, bt, Hcb, nb);
    collect_partners(H, jnum, jlist, Hc, nb, bt, Hcb, &mb_Hown[(size_t) i * 9],
                     &mb_X[(size_t) i * MBMAX * 9], &mb_tag[(size_t) i * MBMAX], mb_n[i]);
  }
  for (int i = 0; i < nlocal; i++) {
    mb_nbt[i] = MIN(atom->num_bond[i], mb_bpa);
    for (int b = 0; b < mb_nbt[i]; b++) mb_btop[(size_t) i * mb_bpa + b] = atom->bond_atom[i][b];
    if (mb_apa) {
      mb_nat[i] = MIN(atom->num_angle[i], mb_apa);
      for (int b = 0; b < mb_nat[i]; b++) {
        tagint *t = &mb_atop[(size_t) i * 3 * mb_apa + 3 * b];
        t[0] = atom->angle_atom1[i][b];
        t[1] = atom->angle_atom2[i][b];
        t[2] = atom->angle_atom3[i][b];
      }
    }
  }
  mb_commflag = 2;
  comm->forward_comm(this, mb_probe_size);
  mb_commflag = 0;

  std::vector<MBMol> mbold;
  mbold.swap(mbmols);
  mb_aidx.clear();
  for (int i = 0; i < nall; i++) {
    mb_state[i][0] = 0.0;
    mb_anc[i] = 0;
  }
  double wmax_ref = 0.0;
  mb_member.assign(nlocal, 0);
  int nmasked = 0, nskip = 0, nmol = 0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int np = mb_n[i];
    if (np <= 0) continue;
    // the molecule is the anchor (lowest ID) and its partners, in the anchor's own partner order so
    // that every rank's copy lists atoms and coordinates identically; built once per rank, from the
    // anchor if it is owned, else from the lowest-ID owned member
    tagint tanc = atom->tag[i];
    for (int p = 0; p < np; p++) tanc = MIN(tanc, mb_tag[(size_t) i * MBMAX + p]);
    int ja = atom->map(tanc);
    if (ja < 0) {
      if (tanc == atom->tag[i]) nskip++;
      continue;
    }
    if ((ja < nlocal) && (i != ja)) continue;
    if (ja >= nlocal) ja = domain->closest_image(i, ja);
    if (mb_n[ja] != np) {
      if (tanc == atom->tag[i]) nskip++;
      continue;
    }
    MBMol M;
    M.n = np + 1;
    M.tag[0] = tanc;
    for (int p = 0; p < np; p++) M.tag[p + 1] = mb_tag[(size_t) ja * MBMAX + p];
    int n = M.n, N = 3 * n;
    int loc[MBMAX + 1];
    bool ok = true;
    int ifirst = -1;
    for (int a = 0; a < n; a++) {
      loc[a] = atom->map(M.tag[a]);
      if (loc[a] < 0) {
        ok = false;
        continue;
      }
      if (loc[a] < nlocal) {
        if ((ifirst < 0) || (atom->tag[loc[a]] < atom->tag[ifirst])) ifirst = loc[a];
      } else {
        loc[a] = domain->closest_image(i, loc[a]);
      }
      if ((mb_n[loc[a]] != np) || !(mask[loc[a]] & groupbit)) ok = false;
    }
    if (ja < nlocal) {
      if (i != ja) continue;
    } else if (i != ifirst) {
      continue;
    }
    if (!ok) {
      if (ja < nlocal) nskip++;
      continue;
    }
    for (int a = 0; a < n; a++) M.m[a] = rmass ? rmass[loc[a]] : mass[type[loc[a]]];

    // raw Hessian, column b from atom b's own probes (see molblock_diagnostic())
    double Hm[MBN][MBN];
    for (int A = 0; A < N; A++)
      for (int B = 0; B < N; B++) Hm[A][B] = 0.0;
    for (int b = 0; b < n; b++) {
      int lb = loc[b];
      for (int r = 0; r < 3; r++)
        for (int s = 0; s < 3; s++) Hm[3 * b + r][3 * b + s] = mb_Hown[(size_t) lb * 9 + 3 * r + s];
      for (int p = 0; p < mb_n[lb]; p++) {
        tagint ta = mb_tag[(size_t) lb * MBMAX + p];
        int a = -1;
        for (int q = 0; q < n; q++)
          if (M.tag[q] == ta) a = q;
        if ((a < 0) || (a == b)) continue;
        const double *X = &mb_X[((size_t) lb * MBMAX + p) * 9];
        for (int r = 0; r < 3; r++)
          for (int s = 0; s < 3; s++) Hm[3 * a + r][3 * b + s] = -X[3 * r + s];
      }
    }

    // reference geometry and mass-weighted Hessian
    if (!mb_locate(M, loc, M.xref)) continue;
    for (int A = 0; A < N; A++)
      for (int B = 0; B < N; B++)
        M.Dref[A][B] = 0.5 * (Hm[A][B] + Hm[B][A]) / sqrt(M.m[A / 3] * M.m[B / 3]);

    // bonds and angles with all atoms inside this molecule (newton_bond off: every atom stores its
    // own; lists of ghost atoms come with the probe data)
    auto idx = [&](tagint t) {
      for (int q = 0; q < n; q++)
        if (M.tag[q] == t) return q;
      return -1;
    };
    M.nic = 0;
    for (int a = 0; a < n; a++) {
      int la = loc[a];
      for (int b = 0; b < mb_nbt[la]; b++) {
        int j = idx(mb_btop[(size_t) la * mb_bpa + b]);
        if ((j < 0) || (j <= a) || (M.nic >= MBN)) continue;
        M.ic_type[M.nic] = 0;
        M.ic_at[M.nic][0] = a;
        M.ic_at[M.nic][1] = j;
        M.nic++;
      }
      for (int b = 0; b < mb_nat[la]; b++) {
        const tagint *t = &mb_atop[(size_t) la * 3 * mb_apa + 3 * b];
        {
          int i1 = idx(t[0]), i2 = idx(t[1]), i3 = idx(t[2]);
          if ((i1 < 0) || (i2 < 0) || (i3 < 0) || (M.nic >= MBN)) continue;
          bool dup = false;
          for (int q = 0; q < M.nic; q++)
            if ((M.ic_type[q] == 1) && (M.ic_at[q][1] == i2) &&
                (((M.ic_at[q][0] == i1) && (M.ic_at[q][2] == i3)) ||
                 ((M.ic_at[q][0] == i3) && (M.ic_at[q][2] == i1))))
              dup = true;
          if (dup) continue;
          M.ic_type[M.nic] = 1;
          M.ic_at[M.nic][0] = i1;
          M.ic_at[M.nic][1] = i2;
          M.ic_at[M.nic][2] = i3;
          M.nic++;
        }
      }
    }
    // angles above 150 degrees at the refresh geometry become linear bends (two coordinates)
    for (int q = 0; q < M.nic; q++) {
      if (M.ic_type[q] != 1) continue;
      int a = M.ic_at[q][0], b = M.ic_at[q][1], cc = M.ic_at[q][2];
      double e1[3], e2[3], r1 = 0.0, r2 = 0.0, d = 0.0;
      for (int p = 0; p < 3; p++) {
        e1[p] = M.xref[a][p] - M.xref[b][p];
        e2[p] = M.xref[cc][p] - M.xref[b][p];
        r1 += e1[p] * e1[p];
        r2 += e2[p] * e2[p];
        d += e1[p] * e2[p];
      }
      if (d / sqrt(r1 * r2) > cos(150.0 * MathConst::MY_PI / 180.0)) continue;
      if (M.nic >= MBN) error->one(FLERR, "Fix baoab/molblock: molecule too large");
      for (int r = M.nic; r > q + 1; r--) {
        M.ic_type[r] = M.ic_type[r - 1];
        for (int k = 0; k < 3; k++) M.ic_at[r][k] = M.ic_at[r - 1][k];
      }
      double e[3];
      int ax = mb_lin_axis(M.xref, a, cc, e);
      M.ic_type[q] = 2 + 2 * ax;
      M.ic_type[q + 1] = 3 + 2 * ax;
      for (int k = 0; k < 3; k++) M.ic_at[q + 1][k] = M.ic_at[q][k];
      M.nic++;
      q++;
    }
    if (M.nic > mb_nicmax)
      error->one(FLERR, "Fix baoab/molblock: the molecule with atom ID {} has more bonds and "
                 "angles ({}) than any one of its atoms takes part in; this molecule shape is not "
                 "supported", M.tag[0], M.nic);

    // fast set: every bond and angle of a molecule with any stiff internal mode, else none
    if (mb_stiff_modes(M, M.xref) == 0) M.nic = 0;
    double fr[MBA][3];
    for (int a = 0; a < n; a++)
      for (int p = 0; p < 3; p++) fr[a][p] = atom->f[loc[a]][p];
    mb_gf_forceconst(M, M.xref, fr);
    mb_gf_freqs(M, M.xref);
    for (int k = 0; k < M.nm; k++) wmax_ref = MAX(wmax_ref, M.w[k]);
    if (mb_verbose && (comm->me == 0) && mbmols.empty() && (M.nic > 0) && (ja < nlocal)) {
      std::string s = fmt::format("MOLBLOCK IC force constants (molecule of atom {}):", M.tag[0]);
      for (int q = 0; q < M.nic; q++) s += fmt::format(" {:.2f}", M.Fic[q][q]);
      s += " | off-diag:";
      for (int q = 0; q < M.nic; q++)
        for (int r = q + 1; r < M.nic; r++) s += fmt::format(" {:.2f}", M.Fic[q][r]);
      utils::logmesg(lmp, s + "\n");
    }

    // same molecule as before the refresh: keep its reference s0 (re-anchoring it would change
    // the fast/slow split mid-run); the step in progress keeps its old force constants through
    // its closing kick, the new ones take over at the start of the next step
    for (auto &O : mbold) {
      if ((O.tag[0] != M.tag[0]) || (O.nic != M.nic)) continue;
      bool same = true;
      for (int q = 0; q < M.nic; q++)
        if ((MIN(O.ic_type[q], 2) != MIN(M.ic_type[q], 2)) || (O.ic_at[q][0] != M.ic_at[q][0]) ||
            (O.ic_at[q][1] != M.ic_at[q][1]))
          same = false;
      if (!same) break;
      M.have_s0 = O.have_s0;
      for (int q = 0; q < M.nic; q++) M.sig0[q] = O.sig0[q];
      M.hprev = O.hprev;
      for (int q = 0; q < M.nic; q++) {
        M.sprev[q] = O.sprev[q];
        M.pprev[q] = O.pprev[q];
      }
      if (O.have_s0) {
        M.pending = true;
        for (int q = 0; q < M.nic; q++)
          for (int r = 0; r < M.nic; r++) M.Fold[q][r] = O.pending ? O.Fold[q][r] : O.Fic[q][r];
      }
      break;
    }
    if (ja < nlocal) {
      nmol++;
      nmasked += M.nm;
    }
    for (int a = 0; a < n; a++)
      if (loc[a] < nlocal) mb_member[loc[a]] = 1;
    mbmols.push_back(M);
    mb_aidx.push_back(ja);
  }
  mb_last_refresh = mb_step;
  MPI_Allreduce(&wmax_ref, &mb_wmax, 1, MPI_DOUBLE, MPI_MAX, world);
  mb_nin = mb_inner_user;
  if (mb_nin == 0) {
    mb_nin = (int) ceil(mb_wmax * update->dt / 0.3);
    mb_nin = MAX(2, mb_nin + (mb_nin % 2));
  }
  int cnt[3] = {nmol, nmasked, nskip}, all[3];
  MPI_Allreduce(cnt, all, 3, MPI_INT, MPI_SUM, world);
  if (mb_verbose && (comm->me == 0))
    utils::logmesg(lmp, "MOLBLOCK refresh step {}: {} molecules, {} fast modes, {} skipped; fastest "
                   "w*dt = {:.3f}, {} inner substeps\n", mb_step, all[0], all[1], all[2],
                   mb_wmax * update->dt, mb_nin);
  if (all[2] > 0)
    error->all(FLERR, "Fix baoab/molblock: {} groups of bonded atoms could not be treated as "
               "molecules (every atom of a molecule must be linked to every other one by a bond or "
               "an angle, at most 5 atoms)", all[2]);
}



/* linear bend of a near-linear triple (a,b,c): a single angle cannot describe the bend of a linear
   molecule (two perpendicular directions; no defined direction of change at 180 degrees). It is
   replaced by the two components of the displacement of the middle atom b from the line a-c,
   d = P (x_b - x_a), P = 1 - e e^T, e = (x_c - x_a)/|x_c - x_a|, along two unit vectors u1, u2
   perpendicular to e (u1 from the lab axis stored in the coordinate type, u2 = e x u1). The bend
   block of the force constants is kept isotropic (k times the unit matrix) and uncoupled from the
   other coordinates, and its reference is d = 0 (both exact by symmetry for a linear equilibrium),
   so the fast energy 1/2 k |d|^2 and its force do not depend on the choice of u1, u2. */
static void mb_lin_frame(const double *e, int ax, double u[2][3])
{
  double nn = 0.0;
  for (int p = 0; p < 3; p++) {
    u[0][p] = ((p == ax) ? 1.0 : 0.0) - e[ax] * e[p];
    nn += u[0][p] * u[0][p];
  }
  nn = sqrt(nn);
  for (int p = 0; p < 3; p++) u[0][p] /= nn;
  u[1][0] = e[1] * u[0][2] - e[2] * u[0][1];
  u[1][1] = e[2] * u[0][0] - e[0] * u[0][2];
  u[1][2] = e[0] * u[0][1] - e[1] * u[0][0];
}

// unit vector from atom a to atom c and the lab axis least aligned with it
static int mb_lin_axis(const double (*xr)[3], int a, int c, double *e)
{
  double nn = 0.0;
  for (int p = 0; p < 3; p++) {
    e[p] = xr[c][p] - xr[a][p];
    nn += e[p] * e[p];
  }
  nn = sqrt(nn);
  int ax = 0;
  for (int p = 0; p < 3; p++) {
    e[p] /= nn;
    if (fabs(e[p]) < fabs(e[ax])) ax = p;
  }
  return ax;
}

// bend block of linear bends made isotropic and uncoupled (see mb_lin_frame) in matrix F
void FixBAOABMolblock::mb_lin_sym(MBMol &M, double (*F)[MBN]) const
{
  for (int q = 0; q + 1 < M.nic; q++) {
    if ((M.ic_type[q] < 2) || ((M.ic_type[q] - 2) % 2 != 0)) continue;
    int q2 = q + 1;
    double k = 0.5 * (F[q][q] + F[q2][q2]);
    for (int r = 0; r < M.nic; r++) F[q][r] = F[r][q] = F[q2][r] = F[r][q2] = 0.0;
    F[q][q] = F[q2][q2] = k;
  }
}

// re-pick the frame axis of linear bends whose line has turned toward it (keeps u1 well defined);
// the next step-to-step difference is skipped, its two ends would be in different frames
void FixBAOABMolblock::mb_lin_reframe(MBMol &M, const double (*xr)[3]) const
{
  for (int q = 0; q + 1 < M.nic; q++) {
    if ((M.ic_type[q] < 2) || ((M.ic_type[q] - 2) % 2 != 0)) continue;
    double e[3];
    int ax = mb_lin_axis(xr, M.ic_at[q][0], M.ic_at[q][2], e);
    int cur = (M.ic_type[q] - 2) / 2;
    if (fabs(e[cur]) < 0.8) continue;
    M.ic_type[q] = 2 + 2 * ax;
    M.ic_type[q + 1] = 3 + 2 * ax;
    M.hprev = false;
  }
}

// internal coordinate values and Cartesian gradients (rows) of molecule M at geometry xr
void FixBAOABMolblock::mb_ic_eval(const MBMol &M, const double (*xr)[3], double *sig,
                                double (*Bc)[MBN]) const
{
  int N = 3 * M.n;
  for (int q = 0; q < M.nic; q++) {
    for (int A = 0; A < N; A++) Bc[q][A] = 0.0;
    if (M.ic_type[q] >= 2) {
      // linear bend component: d_i = u_i . P (x_b - x_a); gradient rows u_i on b, -t u_i on c,
      // -(1 - t) u_i on a, t = e . (x_b - x_a) / |x_c - x_a| (frame held fixed, see mb_lin_frame)
      int a = M.ic_at[q][0], b = M.ic_at[q][1], cc = M.ic_at[q][2];
      int comp = (M.ic_type[q] - 2) % 2, ax = (M.ic_type[q] - 2) / 2;
      double e[3], u[2][3], r[3], L = 0.0, er = 0.0;
      mb_lin_axis(xr, a, cc, e);
      mb_lin_frame(e, ax, u);
      for (int p = 0; p < 3; p++) {
        r[p] = xr[b][p] - xr[a][p];
        double lp = xr[cc][p] - xr[a][p];
        L += lp * lp;
        er += e[p] * r[p];
      }
      L = sqrt(L);
      double t = er / L, s = 0.0;
      for (int p = 0; p < 3; p++) s += u[comp][p] * (r[p] - er * e[p]);
      sig[q] = s;
      for (int p = 0; p < 3; p++) {
        Bc[q][3 * b + p] = u[comp][p];
        Bc[q][3 * cc + p] = -t * u[comp][p];
        Bc[q][3 * a + p] = -(1.0 - t) * u[comp][p];
      }
    } else if (M.ic_type[q] == 0) {
      int a = M.ic_at[q][0], b = M.ic_at[q][1];
      double d[3] = {xr[a][0] - xr[b][0], xr[a][1] - xr[b][1], xr[a][2] - xr[b][2]};
      double r = sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
      sig[q] = r;
      for (int p = 0; p < 3; p++) {
        Bc[q][3 * a + p] = d[p] / r;
        Bc[q][3 * b + p] = -d[p] / r;
      }
    } else {
      int a = M.ic_at[q][0], b = M.ic_at[q][1], cc = M.ic_at[q][2];
      double e1[3], e2[3], r1 = 0.0, r2 = 0.0;
      for (int p = 0; p < 3; p++) {
        e1[p] = xr[a][p] - xr[b][p];
        e2[p] = xr[cc][p] - xr[b][p];
        r1 += e1[p] * e1[p];
        r2 += e2[p] * e2[p];
      }
      r1 = sqrt(r1);
      r2 = sqrt(r2);
      double cth = 0.0;
      for (int p = 0; p < 3; p++) {
        e1[p] /= r1;
        e2[p] /= r2;
        cth += e1[p] * e2[p];
      }
      cth = MAX(-1.0, MIN(1.0, cth));
      sig[q] = acos(cth);
      double sth = sqrt(MAX(1.0 - cth * cth, 1.0e-12));
      for (int p = 0; p < 3; p++) {
        double ga = (cth * e1[p] - e2[p]) / (r1 * sth);
        double gc = (cth * e2[p] - e1[p]) / (r2 * sth);
        Bc[q][3 * a + p] = ga;
        Bc[q][3 * cc + p] = gc;
        Bc[q][3 * b + p] = -ga - gc;
      }
    }
  }
}


// at refresh: internal-coordinate force constants from the measured Cartesian
// Hessian H at geometry xr with raw forces fr (Wilson): B^T F B = H + sum_q phi_q d2sigma_q,
// phi = G^-1 B M^-1 f; F = G^-1 B M^-1 (H + sum phi d2sigma) M^-1 B^T G^-1
void FixBAOABMolblock::mb_gf_forceconst(MBMol &M, const double (*xr)[3], const double (*fr)[3])
{
  int n = M.n, N = 3 * n, nc = M.nic;
  if (nc == 0) return;
  double sg[MBN], Bc[MBN][MBN], G[MBN][MBN], phi[MBN];
  mb_ic_eval(M, xr, sg, Bc);
  for (int q = 0; q < nc; q++) {
    double s = 0.0;
    for (int a = 0; a < n; a++)
      for (int p = 0; p < 3; p++) s += Bc[q][3 * a + p] * fr[a][p] / M.m[a];
    phi[q] = s;
    for (int r = 0; r < nc; r++) {
      double t = 0.0;
      for (int C = 0; C < N; C++) t += Bc[q][C] * Bc[r][C] / M.m[C / 3];
      G[q][r] = t;
    }
  }
  double Gs[MBN][MBN];
  for (int q = 0; q < nc; q++)
    for (int r = 0; r < nc; r++) Gs[q][r] = G[q][r];
  mb_solve(nc, Gs, phi);
  // Hc = H + sum_q phi_q d2sigma_q  (d2sigma by central differences of the analytic gradient)
  double Hc[MBN][MBN];
  for (int A = 0; A < N; A++)
    for (int B = 0; B < N; B++) Hc[A][B] = M.Dref[A][B] * sqrt(M.m[A / 3] * M.m[B / 3]);
  const double h = 1.0e-5;
  for (int C = 0; C < N; C++) {
    double xp[MBA][3], xm[MBA][3], sp[MBN], sm2[MBN], Bp[MBN][MBN], Bm[MBN][MBN];
    for (int a = 0; a < n; a++)
      for (int p = 0; p < 3; p++) xp[a][p] = xm[a][p] = xr[a][p];
    xp[C / 3][C % 3] += h;
    xm[C / 3][C % 3] -= h;
    mb_ic_eval(M, xp, sp, Bp);
    mb_ic_eval(M, xm, sm2, Bm);
    for (int A = 0; A < N; A++) {
      double s = 0.0;
      for (int q = 0; q < nc; q++) s += phi[q] * (Bp[q][A] - Bm[q][A]) / (2.0 * h);
      Hc[A][C] += s;
    }
  }
  // W = G^-1 B M^-1  (nc x N);  F = W Hc W^T
  double W[MBN][MBN];
  for (int A = 0; A < N; A++) {
    double col[MBN];
    for (int q = 0; q < nc; q++) col[q] = Bc[q][A] / M.m[A / 3];
    for (int q = 0; q < nc; q++)
      for (int r = 0; r < nc; r++) Gs[q][r] = G[q][r];
    mb_solve(nc, Gs, col);
    for (int q = 0; q < nc; q++) W[q][A] = col[q];
  }
  for (int q = 0; q < nc; q++)
    for (int r = 0; r < nc; r++) {
      double s = 0.0;
      for (int A = 0; A < N; A++)
        for (int B = 0; B < N; B++) s += W[q][A] * 0.5 * (Hc[A][B] + Hc[B][A]) * W[r][B];
      M.Fic[q][r] = s;
    }
  for (int q = 0; q < nc; q++)
    for (int r = q + 1; r < nc; r++) M.Fic[q][r] = M.Fic[r][q] = 0.5 * (M.Fic[q][r] + M.Fic[r][q]);
  mb_lin_sym(M, M.Fic);
}

// at refresh: GF normal-mode frequencies of the fast internal coordinates with force constants
// Fic: eigenvalues of A = G^1/2 F G^1/2, G = B M^-1 B^T (sets M.nm = M.nic, M.w)
void FixBAOABMolblock::mb_gf_freqs(MBMol &M, const double (*xr)[3])
{
  int n = M.n, N = 3 * n, nc = M.nic;
  double conv = sqrt(force->ftm2v);
  M.nm = 0;
  if (nc == 0) return;
  double sg[MBN], Bc[MBN][MBN];
  mb_ic_eval(M, xr, sg, Bc);
  double **G, **V, *ev;
  memory->create(G, nc, nc, "baoab/tether:gfG");
  memory->create(V, nc, nc, "baoab/tether:gfV");
  memory->create(ev, nc, "baoab/tether:gfev");
  for (int q = 0; q < nc; q++)
    for (int r = 0; r < nc; r++) {
      double t = 0.0;
      for (int C = 0; C < N; C++) t += Bc[q][C] * Bc[r][C] / M.m[C / 3];
      G[q][r] = t;
    }
  MathEigen::Jacobi<double, double *, double **, double const *const *> ecalc(nc);
  ecalc.Diagonalize(G, ev, V);
  // eigenvalues come sorted in decreasing order
  if (ev[nc - 1] < 1.0e-10 * ev[0])
    error->one(FLERR, "Fix baoab/molblock: the bonds and angles of the molecule with atom "
               "ID {} are not independent (e.g. a linear or ring molecule)", M.tag[0]);
  double Gh[MBN][MBN];
  for (int q = 0; q < nc; q++)
    for (int r = 0; r < nc; r++) {
      double a = 0.0;
      for (int e = 0; e < nc; e++) {
        double le = MAX(ev[e], 1.0e-300);
        a += V[e][q] * sqrt(le) * V[e][r];
      }
      Gh[q][r] = a;
    }
  for (int q = 0; q < nc; q++)
    for (int r = 0; r < nc; r++) {
      double s = 0.0;
      for (int a = 0; a < nc; a++)
        for (int b = 0; b < nc; b++) s += Gh[q][a] * M.Fic[a][b] * Gh[b][r];
      G[q][r] = s;
    }
  for (int q = 0; q < nc; q++)
    for (int r = q + 1; r < nc; r++) G[q][r] = G[r][q] = 0.5 * (G[q][r] + G[r][q]);
  ecalc.Diagonalize(G, ev, V);
  for (int k = 0; k < nc; k++) M.w[k] = sqrt(MAX(ev[k], 1.0e-6 * fabs(ev[0]) + 1.0e-300)) * conv;
  M.nm = nc;
  memory->destroy(G);
  memory->destroy(V);
  memory->destroy(ev);
}


// fast (harmonic internal-coordinate) force on the atoms of molecule M at geometry xr
void FixBAOABMolblock::mb_fast_force(const MBMol &M, const double (*xr)[3], double (*ff)[3],
                                   bool old) const
{
  int n = M.n, nc = M.nic;
  for (int a = 0; a < n; a++) ff[a][0] = ff[a][1] = ff[a][2] = 0.0;
  if ((nc == 0) || !M.have_s0) return;
  double sg[MBN], Bc[MBN][MBN], g[MBN];
  mb_ic_eval(M, xr, sg, Bc);
  for (int q = 0; q < nc; q++) {
    double s = 0.0;
    for (int r = 0; r < nc; r++) s += (old ? M.Fold[q][r] : M.Fic[q][r]) * (sg[r] - M.sig0[r]);
    g[q] = s;
  }
  for (int a = 0; a < n; a++)
    for (int p = 0; p < 3; p++) {
      double s = 0.0;
      for (int q = 0; q < nc; q++) s += Bc[q][3 * a + p] * g[q];
      ff[a][p] = -s;
    }
}

// internal coordinates s of molecule M at geometry xr (local indices loc) and the raw force on
// them, phi = G^-1 B M^-1 f with G = B M^-1 B^T (atom->f must hold the raw force)
void FixBAOABMolblock::mb_ic_force(const MBMol &M, const double (*xr)[3], const int *loc, double *sg,
                                 double *phi) const
{
  double **f = atom->f;
  int N = 3 * M.n, nc = M.nic;
  double Bc[MBN][MBN], G[MBN][MBN];
  mb_ic_eval(M, xr, sg, Bc);
  for (int q = 0; q < nc; q++) {
    double s = 0.0;
    for (int a = 0; a < M.n; a++)
      for (int p = 0; p < 3; p++) s += Bc[q][3 * a + p] * f[loc[a]][p] / M.m[a];
    phi[q] = s;
    for (int r = 0; r < nc; r++) {
      double t = 0.0;
      for (int C = 0; C < N; C++) t += Bc[q][C] * Bc[r][C] / M.m[C / 3];
      G[q][r] = t;
    }
  }
  mb_solve(nc, G, phi);
}

// heff_samples > 0, every refresh_every steps after the first: force constants from the step-to-step
// differences, dphi = -F ds. The block moments D_ss = sum ds ds^T and D_ps = sum dphi ds^T enter
// exponential moving averages S (weight 1/heff_samples per block), and F = -S_ps S_ss^-1, so each
// block counts by how much the coordinates moved in it. The finite-difference F of the first refresh
// is the prior: S_ps starts at -F_fd D_ss of the first block, making the first update
// (1 - w) F_fd + w F_block. An estimate that is not positive definite is not used (the molecule
// keeps its F). The step in progress finishes with the old constants (same switch as after a
// probe refresh).
void FixBAOABMolblock::mb_reestimate()
{
  double wt = 1.0 / heff_samples;
  double wmax = 0.0;
  int nupd = 0, nrej = 0;
  int nlocal = atom->nlocal;
  for (size_t km = 0; km < mbmols.size(); km++) {
    MBMol &M = mbmols[km];
    bool counted = mb_aidx[km] < nlocal;    // count each molecule once, on its anchor's rank
    int nc = M.nic;
    if ((nc == 0) || !M.have_s0) continue;
    int loc[MBA];
    double xr[MBA][3];
    bool ok = mb_locate(M, loc, xr) && (M.nd >= nc + 1);
    double Fb[MBN][MBN];
    if (ok) {
      if (!M.has_mom) {
        for (int q = 0; q < nc; q++)
          for (int r = 0; r < nc; r++) {
            double s = 0.0;
            for (int t = 0; t < nc; t++) s += M.Fic[q][t] * M.dss[t][r];
            M.Sss[q][r] = M.dss[q][r];
            M.Sps[q][r] = -s;
          }
        M.has_mom = true;
      }
      for (int q = 0; q < nc; q++)
        for (int r = 0; r < nc; r++) {
          M.Sss[q][r] += wt * (M.dss[q][r] - M.Sss[q][r]);
          M.Sps[q][r] += wt * (M.dps[q][r] - M.Sps[q][r]);
        }
      for (int q = 0; q < nc; q++) {
        double A[MBN][MBN], x[MBN];
        for (int r = 0; r < nc; r++) {
          for (int t = 0; t < nc; t++) A[r][t] = M.Sss[r][t];
          x[r] = -M.Sps[q][r];
        }
        if (!mb_solve(nc, A, x)) ok = false;
        for (int r = 0; r < nc; r++) Fb[q][r] = x[r];
      }
    }
    if (ok) {
      // symmetrize; positive definite (Cholesky) or reject
      double L[MBN][MBN];
      for (int q = 0; q < nc; q++)
        for (int r = q; r < nc; r++) Fb[q][r] = Fb[r][q] = 0.5 * (Fb[q][r] + Fb[r][q]);
      mb_lin_sym(M, Fb);
      for (int q = 0; (q < nc) && ok; q++)
        for (int r = 0; r <= q; r++) {
          double s = Fb[q][r];
          for (int t = 0; t < r; t++) s -= L[q][t] * L[r][t];
          if (q == r) {
            if (s <= 0.0) {
              ok = false;
              break;
            }
            L[q][q] = sqrt(s);
          } else {
            L[q][r] = s / L[r][r];
          }
        }
    }
    M.nd = 0;
    if (ok) {
      if (!M.pending)
        for (int q = 0; q < nc; q++)
          for (int r = 0; r < nc; r++) M.Fold[q][r] = M.Fic[q][r];
      M.pending = true;
      for (int q = 0; q < nc; q++)
        for (int r = 0; r < nc; r++) M.Fic[q][r] = Fb[q][r];
      mb_gf_freqs(M, xr);
      if (counted) nupd++;
    } else if (counted) {
      nrej++;
    }
    for (int k = 0; k < M.nm; k++) wmax = MAX(wmax, M.w[k]);
  }
  mb_last_refresh = mb_step;
  MPI_Allreduce(&wmax, &mb_wmax, 1, MPI_DOUBLE, MPI_MAX, world);
  mb_nin = mb_inner_user;
  if (mb_nin == 0) {
    mb_nin = (int) ceil(mb_wmax * update->dt / 0.3);
    mb_nin = MAX(2, mb_nin + (mb_nin % 2));
  }
  if (mb_verbose) {
    int cnt[2] = {nupd, nrej}, all[2];
    MPI_Allreduce(cnt, all, 2, MPI_INT, MPI_SUM, world);
    double fsum[MBN] = {0.0}, fall[MBN];
    int nf = 0, nfall;
    for (size_t km = 0; km < mbmols.size(); km++) {
      const MBMol &M = mbmols[km];
      if ((mb_aidx[km] < nlocal) && (M.nic == mbmols.front().nic)) {
        nf++;
        for (int q = 0; q < M.nic; q++) fsum[q] += M.Fic[q][q];
      }
    }
    MPI_Allreduce(fsum, fall, MBN, MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(&nf, &nfall, 1, MPI_INT, MPI_SUM, world);
    if (comm->me == 0) {
      std::string s = fmt::format("MOLBLOCK re-estimate step {}: {} molecules updated, {} kept; "
                                  "fastest w*dt = {:.3f}, {} inner substeps; mean diagonal F:",
                                  mb_step, all[0], all[1], mb_wmax * update->dt, mb_nin);
      if (!mbmols.empty())
        for (int q = 0; q < mbmols.front().nic; q++) s += fmt::format(" {:.2f}", fall[q] / MAX(nfall, 1));
      utils::logmesg(lmp, s + "\n");
    }
  }
}

// atom->f holds the raw force at x on entry (owned atoms, and ghosts after the forward comm in
// mb_initial_integrate()); on exit the slow force f - f_fast for every atom of this rank's
// molecule copies, owned or ghost
void FixBAOABMolblock::mb_prepare()
{
  double **f = atom->f;
  int nlocal = atom->nlocal;
  mb_gather();
  // force constants every refresh_every steps: heff_samples = 0, finite-difference probes (the
  // instantaneous curvature); heff_samples > 0, probes only at the first refresh, afterwards a
  // thermally averaged estimate from the trajectory (no extra force evaluations)
  if (mb_last_refresh < 0) mb_refresh();
  else if ((mb_step - mb_last_refresh) >= refresh_every) {
    if (heff_samples > 0) mb_reestimate();
    else mb_refresh();
  }
  mb_member.assign(nlocal, 0);
  for (auto &M : mbmols) {
    int loc[MBA];
    double xr[MBA][3];
    if (!mb_locate(M, loc, xr))
      error->one(FLERR, "Fix baoab/molblock: the molecule of atom ID {} is not fully within "
                 "this processor's ghost atoms; increase the communication cutoff (comm_modify cutoff)",
                 M.tag[0]);
    M.valid = true;
    for (int a = 0; a < M.n; a++)
      if (loc[a] < nlocal) mb_member[loc[a]] = 1;
    int nc = M.nic;
    if (nc == 0) continue;
    mb_lin_reframe(M, xr);
    double sg[MBN], phi[MBN];
    mb_ic_force(M, xr, loc, sg, phi);
    // trajectory estimate of F: consecutive differences, dphi = -F ds (the slowly moving center
    // the molecule vibrates about cancels from the differences but not from the values)
    if (M.nd == 0)
      for (int q = 0; q < nc; q++)
        for (int r = 0; r < nc; r++) M.dss[q][r] = M.dps[q][r] = 0.0;
    if (M.hprev) {
      M.nd++;
      for (int q = 0; q < nc; q++)
        for (int r = 0; r < nc; r++) {
          double dsr = sg[r] - M.sprev[r];
          M.dss[q][r] += (sg[q] - M.sprev[q]) * dsr;
          M.dps[q][r] += (phi[q] - M.pprev[q]) * dsr;
        }
    }
    M.hprev = true;
    for (int q = 0; q < nc; q++) {
      M.sprev[q] = sg[q];
      M.pprev[q] = phi[q];
    }
    if (!M.have_s0) {
      // anchor s0 at the center: s0 = s + F^-1 phi
      double Fm[MBN][MBN], d[MBN];
      for (int q = 0; q < nc; q++) {
        d[q] = phi[q];
        for (int r = 0; r < nc; r++) Fm[q][r] = M.Fic[q][r];
      }
      mb_solve(nc, Fm, d);
      for (int q = 0; q < nc; q++) M.sig0[q] = (M.ic_type[q] >= 2) ? 0.0 : sg[q] + d[q];
      M.have_s0 = true;
    }
    double ff[MBA][3];
    mb_fast_force(M, xr, ff, M.pending);
    for (int a = 0; a < M.n; a++)
      for (int p = 0; p < 3; p++) f[loc[a]][p] -= ff[a][p];
  }
}

/* ----------------------------------------------------------------------
   molblock outer step, first half: B_slow(dt/2) [B_fast A B_fast]^n with
   O mid-loop. The only communication is the forward comm of velocities
   and forces at its start; afterwards every rank integrates its own copy
   of each molecule with an owned atom (ghost members included) and keeps
   the result for the owned atoms only. Copies agree because they start
   from the same data and draw the same noise (mb_gauss()).
------------------------------------------------------------------------- */

void FixBAOABMolblock::mb_initial_integrate()
{
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  double dt = update->dt;
  double kT = force->boltz * t_target / force->mvv2e;
  double c1s = exp(-gamma_soft * dt), s1s = sqrt(kT * (-std::expm1(-2.0 * gamma_soft * dt)));

  mb_commflag = 1;
  comm->forward_comm(this, 6);
  mb_commflag = 0;

  // positions and forces are those of the end of the last step (the first refresh counts as step 1)
  mb_step = (mb_last_refresh < 0) ? update->ntimestep : update->ntimestep - 1;
  mb_prepare();

  // molecules refreshed just now switch to their new force constants: the slow force at the same
  // x changes by f_fast(old F) - f_fast(new F) (the last step closed with the old F)
  for (auto &M : mbmols) {
    if (!M.pending) continue;
    int loc[MBA];
    double xr[MBA][3], fo[MBA][3], fn[MBA][3];
    mb_locate(M, loc, xr);
    mb_fast_force(M, xr, fo, true);
    mb_fast_force(M, xr, fn, false);
    for (int a = 0; a < M.n; a++)
      for (int p = 0; p < 3; p++) f[loc[a]][p] += fo[a][p] - fn[a][p];
    M.pending = false;
  }
  mb_scatter();

  int nin = MAX(1, mb_nin);
  double h = dt / nin, hf = 0.5 * h * force->ftm2v;
  int omid = nin / 2;

  // molecules: B_slow, then velocity Verlet substeps under the fast force, O mid-loop
  for (auto &M : mbmols) {
    int loc[MBA];
    double xr[MBA][3], xa[MBA][3], va[MBA][3], ff[MBA][3];
    mb_locate(M, loc, xr);
    for (int a = 0; a < M.n; a++)
      for (int p = 0; p < 3; p++) {
        xa[a][p] = x[loc[a]][p];
        va[a][p] = v[loc[a]][p] + dtf * f[loc[a]][p] / M.m[a];
      }
    for (int s = 0; s < nin; s++) {
      mb_rel(M, xa, xr);
      mb_fast_force(M, xr, ff);
      for (int a = 0; a < M.n; a++)
        for (int p = 0; p < 3; p++) {
          va[a][p] += hf * ff[a][p] / M.m[a];
          xa[a][p] += h * va[a][p];
          xr[a][p] += h * va[a][p];
        }
      mb_fast_force(M, xr, ff);
      for (int a = 0; a < M.n; a++)
        for (int p = 0; p < 3; p++) va[a][p] += hf * ff[a][p] / M.m[a];
      if ((s == omid - 1) || (nin == 1))
        for (int a = 0; a < M.n; a++) {
          double sv = s1s / sqrt(M.m[a]);
          for (int p = 0; p < 3; p++) va[a][p] = c1s * va[a][p] + sv * mb_gauss(M.tag[a], p);
        }
    }
    for (int a = 0; a < M.n; a++) {
      if (loc[a] >= nlocal) continue;
      for (int p = 0; p < 3; p++) {
        x[loc[a]][p] = xa[a][p];
        v[loc[a]][p] = va[a][p];
      }
    }
  }

  // everything else: B_slow, drift, O mid-loop
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit) || mb_member[i]) continue;
    double mi = rmass ? rmass[i] : mass[type[i]];
    double sv = s1s / sqrt(mi);
    for (int k = 0; k < 3; k++) v[i][k] += dtf * f[i][k] / mi;
    for (int s = 0; s < nin; s++) {
      for (int k = 0; k < 3; k++) x[i][k] += h * v[i][k];
      if ((s == omid - 1) || (nin == 1))
        for (int k = 0; k < 3; k++) v[i][k] = c1s * v[i][k] + sv * mb_gauss(tag[i], k);
    }
  }
}

// number of stiff internal modes of molecule M at its refresh geometry xr: the refresh Hessian
// (M.Dref) diagonalized within the span of the mass-weighted bond/angle gradients (orthonormalized);
// a mode is stiff if omega*dt > theta
int FixBAOABMolblock::mb_stiff_modes(const MBMol &M, const double (*xr)[3]) const
{
  int n = M.n, N = 3 * n;
  double conv = sqrt(force->ftm2v);
  double E[MBN][MBN], sg[MBN], Bc[MBN][MBN];
  int ne = 0;
  mb_ic_eval(M, xr, sg, Bc);
  for (int q = 0; q < M.nic; q++) {
    double g[MBN];
    for (int A = 0; A < N; A++) g[A] = Bc[q][A] / sqrt(M.m[A / 3]);
    for (int e = 0; e < ne; e++) {
      double dp = 0.0;
      for (int A = 0; A < N; A++) dp += E[e][A] * g[A];
      for (int A = 0; A < N; A++) g[A] -= dp * E[e][A];
    }
    double nn = 0.0;
    for (int A = 0; A < N; A++) nn += g[A] * g[A];
    if (nn < 1.0e-12) continue;
    for (int A = 0; A < N; A++) E[ne][A] = g[A] / sqrt(nn);
    ne++;
  }
  if (ne == 0) return 0;
  double **K, **V, *ev;
  memory->create(K, ne, ne, "baoab/tether:mbK");
  memory->create(V, ne, ne, "baoab/tether:mbKV");
  memory->create(ev, ne, "baoab/tether:mbKev");
  for (int e = 0; e < ne; e++)
    for (int f2 = 0; f2 < ne; f2++) {
      double s = 0.0;
      for (int A = 0; A < N; A++)
        for (int B = 0; B < N; B++) s += E[e][A] * M.Dref[A][B] * E[f2][B];
      K[e][f2] = s;
    }
  for (int e = 0; e < ne; e++)
    for (int f2 = e + 1; f2 < ne; f2++) K[e][f2] = K[f2][e] = 0.5 * (K[e][f2] + K[f2][e]);
  MathEigen::Jacobi<double, double *, double **, double const *const *> ecalc(ne);
  ecalc.Diagonalize(K, ev, V);
  int nstiff = 0;
  for (int m = 0; m < ne; m++)
    if ((ev[m] > 0.0) && (sqrt(ev[m]) * conv * update->dt > theta)) nstiff++;
  memory->destroy(K);
  memory->destroy(V);
  memory->destroy(ev);
  return nstiff;
}

// molblock outer step, second half: B_slow(dt/2) with the slow force f - f_fast at the new x
void FixBAOABMolblock::mb_final_integrate()
{
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  mb_gather();
  mb_member.assign(nlocal, 0);
  for (auto &M : mbmols) {
    int loc[MBA];
    double xr[MBA][3], ff[MBA][3];
    if (!mb_locate(M, loc, xr))
      error->one(FLERR, "Fix baoab/molblock: the molecule of atom ID {} is not fully within "
                 "this processor's ghost atoms; increase the communication cutoff (comm_modify cutoff)",
                 M.tag[0]);
    mb_fast_force(M, xr, ff);
    for (int a = 0; a < M.n; a++) {
      int i = loc[a];
      if (i >= nlocal) continue;
      mb_member[i] = 1;
      for (int p = 0; p < 3; p++) v[i][p] += dtf * (f[i][p] - ff[a][p]) / M.m[a];
    }
  }
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit) || mb_member[i]) continue;
    double mi = rmass ? rmass[i] : mass[type[i]];
    for (int k = 0; k < 3; k++) v[i][k] += dtf * f[i][k] / mi;
  }
}

