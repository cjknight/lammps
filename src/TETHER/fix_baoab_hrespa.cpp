/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_baoab_hrespa.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "kspace.h"
#include "pair.h"
#include "platform.h"
#include "random_mars.h"
#include "safe_pointers.h"
#include "text_file_reader.h"
#include "tokenizer.h"
#include "update.h"

#include <array>
#include <cmath>
#include <cstring>
#include <map>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

FixBAOABHRespa::FixBAOABHRespa(LAMMPS *lmp, int narg, char **arg) :
    FixBAOABTetherBase(lmp, narg, arg, "fix baoab/hrespa"), inner_user(0), nin(1), cut(0.0),
    wmax(0.0), eps(0.0), file_in(nullptr), file_out(nullptr), ready(false), natoms_ref(0)
{
  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "inner") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/hrespa inner", error);
      inner_user = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (inner_user < 0) error->all(FLERR, iarg + 1, "Fix baoab/hrespa inner must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "cut") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/hrespa cut", error);
      cut = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (cut <= 0.0) error->all(FLERR, iarg + 1, "Fix baoab/hrespa cut must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "eps") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/hrespa eps", error);
      eps = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      if (eps <= 0.0) error->all(FLERR, iarg + 1, "Fix baoab/hrespa eps must be > 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "read") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/hrespa read", error);
      delete[] file_in;
      file_in = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "write") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/hrespa write", error);
      delete[] file_out;
      file_out = utils::strdup(arg[iarg + 1]);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown fix baoab/hrespa keyword {}", arg[iarg]);
    }
  }
  if (igroup != 0) error->all(FLERR, 1, "Fix baoab/hrespa must be applied to group all");
}

/* ---------------------------------------------------------------------- */

FixBAOABHRespa::~FixBAOABHRespa()
{
  delete[] file_in;
  delete[] file_out;
}

/* ---------------------------------------------------------------------- */

void FixBAOABHRespa::init()
{
  FixBAOAB::init();
  // the friction must act on a timescale longer than one step (damp is in the run's time
  // units, so a value copied between unit systems is off by orders of magnitude)
  if (gamma * update->dt > 1.0)
    error->all(FLERR, Error::NOLASTLINE, "Fix baoab/hrespa damp {} is shorter than one timestep "
               "({}): the thermostat would re-randomize velocities every step. damp is in the "
               "run's time units; a typical choice is about 100 times the timestep, here {}",
               1.0 / gamma, update->dt, 100.0 * update->dt);
  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, Error::NOLASTLINE, "Fix baoab/hrespa requires atom_modify map (yes|array)");
}

/* ---------------------------------------------------------------------- */

void FixBAOABHRespa::setup(int /*vflag*/)
{
  setup_model();
}

/* ---------------------------------------------------------------------- */

double FixBAOABHRespa::memory_usage()
{
  double bytes = (double) (xref.size() + fref.size() + mass_id.size() + K.size() + fh.size()) *
      sizeof(double);
  bytes += (double) rowptr.size() * sizeof(bigint) + (double) col.size() * sizeof(tagint);
  return bytes;
}

/* ----------------------------------------------------------------------
   The harmonic inner model (see the header) is
     F_h(x) = f_ref - K (x - x_ref),  displacements by minimum image,
   with K symmetric, cut at "cut" (atom pairs farther apart at x_ref are dropped) and the
   acoustic sum rule restored (diagonal blocks from the off-diagonal ones), as in the Python
   prototype (workloads/mtp-les_HfO/mts_proto/mts_levels.py, HarmonicCartesian).
------------------------------------------------------------------------- */

void FixBAOABHRespa::setup_model()
{
  if (atom->natoms > MAXSMALLINT || atom->map_tag_max != atom->natoms)
    error->all(FLERR, "Fix baoab/hrespa requires consecutive atom IDs 1..N");
  if (!force->pair) error->all(FLERR, "Fix baoab/hrespa requires a pair style");
  if (cut <= 0.0) cut = force->pair->cutforce;
  double cutghost = comm->get_comm_cutoff();
  if (cut > cutghost)
    error->all(FLERR, "Fix baoab/hrespa cut {} exceeds the ghost-atom cutoff {}; increase it "
               "with comm_modify cutoff", cut, cutghost);

  int nlocal = atom->nlocal;
  tagint *tag = atom->tag;
  if (ready && atom->natoms != natoms_ref)
    error->all(FLERR, "Fix baoab/hrespa: the number of atoms changed since the harmonic "
               "model was built");

  // masses by atom ID (the model is replicated, so are they)
  bigint N = atom->natoms;
  {
    std::vector<double> m(N + 1, 0.0);
    for (int i = 0; i < nlocal; i++)
      m[tag[i]] = atom->rmass ? atom->rmass[i] : atom->mass[atom->type[i]];
    mass_id.assign(N + 1, 0.0);
    MPI_Allreduce(m.data(), mass_id.data(), N + 1, MPI_DOUBLE, MPI_SUM, world);
  }

  if (!ready) {
    natoms_ref = N;
    double t0 = platform::walltime();
    if (file_in) read_model(file_in);
    else build_model_fd();
    if (file_out) write_model(file_out);
    ready = true;

    // fastest harmonic frequency (power iteration on M^-1/2 K M^-1/2, identical on every rank)
    std::vector<double> z(3 * (N + 1), 0.0), y(3 * (N + 1), 0.0);
    for (bigint t = 1; t <= N; t++)
      for (int a = 0; a < 3; a++) z[3 * t + a] = ((t * 7 + a * 3) % 11) - 5.0 + 0.1 * a;
    double lam = 0.0;
    for (int it = 0; it < 300; it++) {
      double nz = 0.0;
      for (auto zi : z) nz += zi * zi;
      nz = sqrt(nz);
      for (auto &zi : z) zi /= nz;
      for (bigint i = 1; i <= N; i++) {
        double yi[3] = {0.0, 0.0, 0.0};
        for (bigint k = rowptr[i]; k < rowptr[i + 1]; k++) {
          tagint t = col[k];
          const double *Kb = &K[9 * k];
          double st = 1.0 / sqrt(mass_id[t]);
          for (int a = 0; a < 3; a++)
            for (int b = 0; b < 3; b++) yi[a] += Kb[3 * a + b] * st * z[3 * t + b];
        }
        for (int a = 0; a < 3; a++) y[3 * i + a] = yi[a] / sqrt(mass_id[i]);
      }
      lam = 0.0;
      for (size_t k = 0; k < z.size(); k++) lam += z[k] * y[k];
      z.swap(y);
    }
    wmax = sqrt(fabs(lam) * force->ftm2v);
    if (comm->me == 0)
      utils::logmesg(lmp, "  baoab/hrespa: harmonic model of {} atoms, {} 3x3 blocks ({:.1f} per atom, "
                     "cut {}), fastest frequency {:.4g} rad/time unit, built in {:.1f} s\n", N,
                     K.size() / 9, (double) K.size() / 9.0 / N, cut, wmax,
                     platform::walltime() - t0);
  }

  // inner substeps: omega_max * h <= 0.1 (velocity Verlet's frequency error ~(omega h)^2/24 ~ 4e-4)
  nin = inner_user;
  if (nin == 0) {
    nin = (int) ceil(wmax * update->dt / 0.1);
    nin = MAX(2, nin + (nin % 2));
  }
  if (comm->me == 0)
    utils::logmesg(lmp, "  baoab/hrespa: {} inner substeps per timestep (omega_max * dt = {:.3f})\n",
                   nin, wmax * update->dt);
}

// reference = the current configuration; K by central differences of whole-system forces
// (6N recomputes). The step is eps if given (keyword eps), else 0.001 Angstrom (as in the Python prototype).
void FixBAOABHRespa::build_model_fd()
{
  double **x = atom->x;
  double **f = atom->f;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  int ntotal = nlocal + atom->nghost;
  bigint N = natoms_ref;

  // reference positions and forces by atom ID (replicated)
  {
    std::vector<double> xl(3 * (N + 1), 0.0), fl(3 * (N + 1), 0.0);
    for (int i = 0; i < nlocal; i++)
      for (int a = 0; a < 3; a++) {
        xl[3 * tag[i] + a] = x[i][a];
        fl[3 * tag[i] + a] = f[i][a];
      }
    xref.assign(3 * (N + 1), 0.0);
    fref.assign(3 * (N + 1), 0.0);
    MPI_Allreduce(xl.data(), xref.data(), 3 * (N + 1), MPI_DOUBLE, MPI_SUM, world);
    MPI_Allreduce(fl.data(), fref.data(), 3 * (N + 1), MPI_DOUBLE, MPI_SUM, world);
  }

  // the probes overwrite forces, charges and the accumulated energies/virials: save them
  std::vector<double> fsave(3 * ntotal);
  if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));
  std::vector<double> qsave;
  save_charges(qsave);
  double esave[NACCUM];
  save_accumulators(esave);

  double e = (eps > 0.0) ? eps : 1.0e-3 * force->angstrom;
  double cut2 = cut * cut;
  std::vector<double> fplus(3 * nlocal), loc;    // loc: (row ID, column ID, b, -dF_row/dx_col,b)
  double t0 = platform::walltime();
  if (comm->me == 0)
    utils::logmesg(lmp, "  baoab/hrespa: building the harmonic model by finite differences ({} force "
                   "evaluations, step {})\n", 6 * N, e);
  for (bigint t = 1; t <= N; t++) {
    int it = atom->map((tagint) t);
    bool own = (it >= 0) && (it < nlocal);
    for (int b = 0; b < 3; b++) {
      double x0 = own ? x[it][b] : 0.0;
      for (int s = 0; s < 2; s++) {
        if (own) x[it][b] = x0 + (s == 0 ? e : -e);
        comm->forward_comm();
        recompute_forces_local();
        if (s == 0)
          for (int j = 0; j < nlocal; j++)
            for (int a = 0; a < 3; a++) fplus[3 * j + a] = f[j][a];
      }
      if (own) x[it][b] = x0;
      for (int j = 0; j < nlocal; j++) {
        tagint tj = tag[j];
        double d[3];
        for (int a = 0; a < 3; a++) d[a] = xref[3 * tj + a] - xref[3 * t + a];
        domain->minimum_image(FLERR, d);
        if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > cut2) continue;
        loc.push_back((double) tj);
        loc.push_back((double) t);
        loc.push_back((double) b);
        for (int a = 0; a < 3; a++) loc.push_back(-(fplus[3 * j + a] - f[j][a]) / (2.0 * e));
      }
    }
    if ((comm->me == 0) && ((t % MAX(1, N / 10)) == 0))
      utils::logmesg(lmp, "  baoab/hrespa: {}/{} atoms, {:.1f} s\n", t, N, platform::walltime() - t0);
  }
  comm->forward_comm();
  if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));
  restore_charges(qsave);
  restore_accumulators(esave);

  // gather every rank's columns
  int n = (int) loc.size(), np = comm->nprocs;
  std::vector<int> cnt(np), dsp(np, 0);
  MPI_Allgather(&n, 1, MPI_INT, cnt.data(), 1, MPI_INT, world);
  for (int p = 1; p < np; p++) dsp[p] = dsp[p - 1] + cnt[p - 1];
  std::vector<double> all(dsp[np - 1] + cnt[np - 1]);
  MPI_Allgatherv(loc.data(), n, MPI_DOUBLE, all.data(), cnt.data(), dsp.data(), MPI_DOUBLE, world);

  std::map<std::pair<tagint, tagint>, std::array<double, 9>> B;
  for (size_t k = 0; k < all.size(); k += 6) {
    auto key = std::make_pair((tagint) all[k], (tagint) all[k + 1]);
    int b = (int) all[k + 2];
    auto it = B.find(key);
    if (it == B.end()) it = B.emplace(key, std::array<double, 9>{}).first;
    for (int a = 0; a < 3; a++) it->second[3 * a + b] = all[k + 3 + a];
  }
  std::vector<tagint> rows, cols;
  std::vector<double> vals;
  for (auto &kv : B) {
    rows.push_back(kv.first.first);
    cols.push_back(kv.first.second);
    vals.insert(vals.end(), kv.second.begin(), kv.second.end());
  }
  finish_model(rows, cols, vals);
}

// symmetrize, cut, restore the acoustic sum rule, store as CSR rows by atom ID
void FixBAOABHRespa::finish_model(std::vector<tagint> &rows, std::vector<tagint> &cols,
                               std::vector<double> &vals)
{
  bigint N = natoms_ref;
  double cut2 = cut * cut;
  std::map<std::pair<tagint, tagint>, std::array<double, 9>> B;
  for (size_t k = 0; k < rows.size(); k++) {
    tagint i = rows[k], j = cols[k];
    if (i == j) continue;    // diagonal blocks come from the sum rule
    double d[3];
    for (int a = 0; a < 3; a++) d[a] = xref[3 * i + a] - xref[3 * j + a];
    domain->minimum_image(FLERR, d);
    if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2] > cut2) continue;
    std::array<double, 9> blk;
    for (int q = 0; q < 9; q++) blk[q] = vals[9 * k + q];
    B[std::make_pair(i, j)] = blk;
  }
  // K_ij = (K_ij + K_ji^T) / 2 (a block missing on one side counts as zero)
  std::map<std::pair<tagint, tagint>, std::array<double, 9>> S;
  for (auto &kv : B) {
    tagint i = kv.first.first, j = kv.first.second;
    if (S.count(kv.first)) continue;
    std::array<double, 9> kij = kv.second, kji{};
    auto it = B.find(std::make_pair(j, i));
    if (it != B.end()) kji = it->second;
    std::array<double, 9> s, st;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) {
        s[3 * a + b] = 0.5 * (kij[3 * a + b] + kji[3 * b + a]);
        st[3 * b + a] = s[3 * a + b];
      }
    S[kv.first] = s;
    S[std::make_pair(j, i)] = st;
  }
  // diagonal: -(sum_j K_ij) symmetrized
  std::vector<std::array<double, 9>> D(N + 1, std::array<double, 9>{});
  for (auto &kv : S)
    for (int q = 0; q < 9; q++) D[kv.first.first][q] -= kv.second[q];
  for (bigint i = 1; i <= N; i++) {
    std::array<double, 9> d = D[i];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) D[i][3 * a + b] = 0.5 * (d[3 * a + b] + d[3 * b + a]);
    S[std::make_pair((tagint) i, (tagint) i)] = D[i];
  }
  rowptr.assign(N + 2, 0);
  col.clear();
  K.clear();
  for (auto &kv : S) {    // ordered by (row, column)
    rowptr[kv.first.first + 1]++;
    col.push_back(kv.first.second);
    K.insert(K.end(), kv.second.begin(), kv.second.end());
  }
  for (bigint i = 1; i <= N + 1; i++) rowptr[i] += rowptr[i - 1];
}

// file format (text): comment lines start with '#'; then "N nblocks"; N lines
// "ID x y z fx fy fz" (the reference); nblocks lines "I J K_xx K_xy ... K_zz" (row atom I,
// column atom J, row-major). Blocks are re-cut, symmetrized and sum-ruled on reading.
void FixBAOABHRespa::write_model(const char *file) const
{
  if (comm->me != 0) return;
  SafeFilePtr fp = fopen(file, "w");
  if (!fp) error->one(FLERR, "Fix baoab/hrespa write: cannot open file {}: {}", file,
                      utils::getsyserror());
  bigint N = natoms_ref;
  utils::print(fp, "# fix baoab/hrespa harmonic model: F(x) = f_ref - K (x - x_ref)\n");
  utils::print(fp, "{} {}\n", N, K.size() / 9);
  for (bigint t = 1; t <= N; t++)
    utils::print(fp, "{} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g}\n", t, xref[3 * t],
               xref[3 * t + 1], xref[3 * t + 2], fref[3 * t], fref[3 * t + 1],
               fref[3 * t + 2]);
  for (bigint i = 1; i <= N; i++)
    for (bigint k = rowptr[i]; k < rowptr[i + 1]; k++) {
      utils::print(fp, "{} {}", i, col[k]);
      for (int q = 0; q < 9; q++) utils::print(fp, " {:.17g}", K[9 * k + q]);
      utils::print(fp, "\n");
    }
  fp = nullptr;
  utils::logmesg(lmp, "  baoab/hrespa: harmonic model written to {}\n", file);
}

void FixBAOABHRespa::read_model(const char *file)
{
  bigint N = natoms_ref;
  std::vector<double> xf(6 * (N + 1), 0.0);
  std::vector<tagint> rows, cols;
  std::vector<double> vals;
  bigint nb = 0;
  if (comm->me == 0) {
    try {
      TextFileReader reader(file, "hrespa harmonic model");
      auto v = reader.next_values(2);
      bigint nf = v.next_bigint();
      nb = v.next_bigint();
      if (nf != N)
        error->one(FLERR, "Fix baoab/hrespa read: file {} has {} atoms, the system {}", file,
                   nf, N);
      for (bigint t = 0; t < N; t++) {
        auto w = reader.next_values(7);
        tagint id = w.next_tagint();
        if (id < 1 || id > N) error->one(FLERR, "Fix baoab/hrespa read: bad atom ID {}", id);
        for (int q = 0; q < 6; q++) xf[6 * id + q] = w.next_double();
      }
      for (bigint k = 0; k < nb; k++) {
        auto w = reader.next_values(11);
        rows.push_back(w.next_tagint());
        cols.push_back(w.next_tagint());
        for (int q = 0; q < 9; q++) vals.push_back(w.next_double());
      }
    } catch (std::exception &ex) {
      error->one(FLERR, "Fix baoab/hrespa read: error reading {}: {}", file, ex.what());
    }
  }
  MPI_Bcast(xf.data(), 6 * (N + 1), MPI_DOUBLE, 0, world);
  MPI_Bcast(&nb, 1, MPI_LMP_BIGINT, 0, world);
  rows.resize(nb);
  cols.resize(nb);
  vals.resize(9 * nb);
  MPI_Bcast(rows.data(), nb, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(cols.data(), nb, MPI_LMP_TAGINT, 0, world);
  MPI_Bcast(vals.data(), 9 * nb, MPI_DOUBLE, 0, world);
  xref.assign(3 * (N + 1), 0.0);
  fref.assign(3 * (N + 1), 0.0);
  for (bigint t = 1; t <= N; t++)
    for (int a = 0; a < 3; a++) {
      xref[3 * t + a] = xf[6 * t + a];
      fref[3 * t + a] = xf[6 * t + 3 + a];
    }
  finish_model(rows, cols, vals);
  if (comm->me == 0) utils::logmesg(lmp, "  baoab/hrespa: harmonic model read from {}\n", file);
}

// harmonic force on every owned atom at the current positions (ghosts must be current)
void FixBAOABHRespa::harmonic_force()
{
  double **x = atom->x;
  tagint *tag = atom->tag;
  int nlocal = atom->nlocal;
  fh.resize(3 * nlocal);
  for (int i = 0; i < nlocal; i++) {
    tagint ti = tag[i];
    double fi[3] = {fref[3 * ti], fref[3 * ti + 1], fref[3 * ti + 2]};
    for (bigint k = rowptr[ti]; k < rowptr[ti + 1]; k++) {
      tagint t = col[k];
      int m = atom->map(t);
      if (m < 0)
        error->one(FLERR, "Fix baoab/hrespa: atom {} (coupled to atom {}) is not a local or "
                   "ghost atom; increase the ghost-atom cutoff (comm_modify cutoff)", t, ti);
      double d[3] = {x[m][0] - xref[3 * t], x[m][1] - xref[3 * t + 1],
                     x[m][2] - xref[3 * t + 2]};
      domain->minimum_image(FLERR, d);
      const double *Kb = &K[9 * k];
      for (int a = 0; a < 3; a++) fi[a] -= Kb[3 * a] * d[0] + Kb[3 * a + 1] * d[1] + Kb[3 * a + 2] * d[2];
    }
    for (int a = 0; a < 3; a++) fh[3 * i + a] = fi[a];
  }
}

// outer half kick with F - F_h, inner velocity-Verlet substeps under F_h (O step mid-loop),
// the closing outer half kick is in final_integrate()
void FixBAOABHRespa::initial_integrate(int /*vflag*/)
{
  compute_target();
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double dt = update->dt;
  double kT = force->boltz * t_target / force->mvv2e;
  double c1o = exp(-gamma * dt), s1o = sqrt(kT * (-std::expm1(-2.0 * gamma * dt)));

  int ns = MAX(1, nin);
  double h = dt / ns, hk = 0.5 * h * force->ftm2v;
  int omid = ns / 2;

  harmonic_force();
  for (int i = 0; i < nlocal; i++) {
    double mi = rmass ? rmass[i] : mass[type[i]];
    for (int a = 0; a < 3; a++) v[i][a] += dtf * (f[i][a] - fh[3 * i + a]) / mi;
  }
  for (int s = 0; s < ns; s++) {
    for (int i = 0; i < nlocal; i++) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      for (int a = 0; a < 3; a++) {
        v[i][a] += hk * fh[3 * i + a] / mi;
        x[i][a] += h * v[i][a];
      }
    }
    comm->forward_comm();
    harmonic_force();
    for (int i = 0; i < nlocal; i++) {
      double mi = rmass ? rmass[i] : mass[type[i]];
      for (int a = 0; a < 3; a++) v[i][a] += hk * fh[3 * i + a] / mi;
    }
    if (((s == omid - 1) || (ns == 1)) && (gamma > 0.0))
      for (int i = 0; i < nlocal; i++) {
        double mi = rmass ? rmass[i] : mass[type[i]];
        double sv = s1o / sqrt(mi);
        for (int a = 0; a < 3; a++) v[i][a] = c1o * v[i][a] + sv * random->gaussian();
      }
  }
}

void FixBAOABHRespa::final_integrate()
{
  double **v = atom->v;
  double **f = atom->f;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  harmonic_force();
  for (int i = 0; i < nlocal; i++) {
    double mi = rmass ? rmass[i] : mass[type[i]];
    for (int a = 0; a < 3; a++) v[i][a] += dtf * (f[i][a] - fh[3 * i + a]) / mi;
  }
}
