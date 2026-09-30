/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// Contributing author: Chuck Witt

#include "pair_symmetrix_mace.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"

#include <algorithm>
#include <numeric>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairSymmetrixMACE::PairSymmetrixMACE(LAMMPS *lmp)
  : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
  no_virial_fdotr_compute = 1;
  // WARNING: for mace, these variables are model-dependent, so i
  //          reset them after the model is loaded (in coeff).
  //          however, i can't make them zero here, because that
  //          confusingly yields seg faults with hybrid/overlay.
  //          so, i set them to a fairly big number here and hope.
  //          not a great solution.
  comm_forward = 1024;
  comm_reverse = 1024;
}

/* ---------------------------------------------------------------------- */

PairSymmetrixMACE::~PairSymmetrixMACE()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(cutghost);
  }
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute(int eflag, int vflag)
{
  if (mode == "no_domain_decomposition") {
    compute_no_domain_decomposition(eflag, vflag);
  } else if (mode == "mpi_message_passing") {
    compute_mpi_message_passing(eflag, vflag);
  } else if (mode == "no_mpi_message_passing") {
    compute_no_mpi_message_passing(eflag, vflag);
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairSymmetrixMACE::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; ++i)
    for (int j=i; j<atom->ntypes+1; ++j)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
  if (ghostneigh)
    memory->create(cutghost, atom->ntypes+1, atom->ntypes+1, "pair:cutghost");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairSymmetrixMACE::settings(int narg, char **arg)
{
  if (narg == 0) {
    mode = (comm->nprocs == 1) ? "no_domain_decomposition" : "mpi_message_passing";
  } else if (narg == 1) {
    mode = std::string(arg[0]);
    if (mode != "no_domain_decomposition" and mode != "mpi_message_passing" and mode != "no_mpi_message_passing")
        error->all(FLERR, "The command \'pair_style symmetrix/mace {}\' is invalid", mode);
  } else {
    error->all(FLERR, "Too many pair_style arguments for symmetrix/mace");
  }

  if (mode == "no_domain_decomposition" and comm->nprocs != 1)
    error->all(FLERR, "Cannot use no_domain_decomposition with multiple MPI processes");

  ghostneigh = (mode == "no_mpi_message_passing");
  // local_partial_force() reuses the ghost-halo/edge-list machinery that
  // only no_mpi_message_passing sets up (2*r_cut+skin comm cutoff and a
  // REQ_GHOST neighbor list valid for ghost-atom "i" rows); not offered in
  // the other two modes.
  has_local_partial_force = (mode == "no_mpi_message_passing");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairSymmetrixMACE::coeff(int narg, char **arg)
{
  if (!allocated) allocate();

  utils::logmesg(lmp, "Loading MACE model from \'{}\' ... ", arg[2]);
  mace = std::make_unique<MACE>(arg[2]);
  utils::logmesg(lmp, "success\n");

  // extract atomic numbers from pair_coeff
  mace_types = std::vector<int>();
  for (int i=3; i<narg; ++i) {
    // find atomic number for element in arg[i]
    auto iter1 = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    if (iter1 == periodic_table.end())
      error->all(FLERR, "{} does not appear in the periodic table", arg[i]);
    int atomic_number = std::distance(periodic_table.begin(), iter1) + 1;
    // find mace index corresponding to this element
    auto iter2 = std::find(mace->atomic_numbers.begin(), mace->atomic_numbers.end(), atomic_number);
    if (iter2 == mace->atomic_numbers.end())
      error->all(FLERR, "Problem matching LAMMPS types to MACE types.");
    int mace_index = std::distance(mace->atomic_numbers.begin(), iter2);
    utils::logmesg(lmp, "  mapping LAMMPS type {} ({}) to MACE type {}\n",
                   i-2, arg[i], mace_index);
    mace_types.push_back(mace_index);
  }

  // set message size
  if (mode == "mpi_message_passing") {
    comm_forward = mace->num_LM*mace->num_channels;
    comm_reverse = mace->num_LM*mace->num_channels;
  } else {
    comm_forward = 0;
    comm_reverse = 0;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairSymmetrixMACE::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

  if (ghostneigh) cutghost[i][j] = cutghost[j][i] = mace->r_cut;
  return mace->r_cut;
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairSymmetrixMACE::init_style()
{
  if (atom->map_user == atom->MAP_NONE) error->all(FLERR, "symmetrix/mace requires \'atom_modify map [yes|array|hash]\'");
  if (force->newton_pair == 0) error->all(FLERR, "symmetrix/mace requires newton pair on");

  if (mode == "no_domain_decomposition" or mode == "mpi_message_passing") {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  } else {
    // enforce the communication cutoff is more than twice the model cutoff
    const double comm_cutoff = comm->get_comm_cutoff();
    if (comm->get_comm_cutoff() < (2*mace->r_cut + neighbor->skin)){
      std::string cutoff_val = std::to_string((2.0 * mace->r_cut) + neighbor->skin);
      char *args[2];
      args[0] = (char *)"cutoff";
      args[1] = const_cast<char *>(cutoff_val.c_str());
      comm->modify_params(2, args);
      if (comm->me == 0) utils::logmesg(lmp, "symmetrix/mace is setting the communication cutoff to {}", cutoff_val);
    }
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  }
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACE::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/, int * /*pbc*/)
{
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      buf[ii*mace->num_LM*mace->num_channels+k] = H1[i*mace->num_LM*mace->num_channels+k];
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::unpack_forward_comm(int n, int first, double *buf)
{
  for (int i=0; i<n; ++i) {
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1[(first+i)*mace->num_LM*mace->num_channels+k] = buf[i*mace->num_LM*mace->num_channels+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

int PairSymmetrixMACE::pack_reverse_comm(int n, int first, double *buf)
{
  for (int i=0; i<n; ++i) {
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      buf[i*mace->num_LM*mace->num_channels+k] = H1_adj[(first+i)*mace->num_LM*mace->num_channels+k];
    }
  }
  return n*mace->num_LM*mace->num_channels;
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::unpack_reverse_comm(int n, int *list, double *buf)
{
  for (int ii=0; ii<n; ++ii) {
    const int i = list[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1_adj[i*mace->num_LM*mace->num_channels+k] += buf[ii*mace->num_LM*mace->num_channels+k];
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute_no_domain_decomposition(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // count edges
  int num_edges = 0;
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared)
        num_edges += 1;
    }
  }

  // resize vectors
  const int num_nodes = list->inum;
  node_types.resize(list->inum);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(num_edges);
  neigh_types.resize(num_edges);
  xyz.resize(3*num_edges);
  r.resize(num_edges);
  node_i.resize(num_nodes);
  neigh_j.resize(num_edges);

  // update ii_from_i
  ii_from_i.clear();
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    ii_from_i[i] = ii;
  }

  // populate neighbor list variables
  int ij = 0;
  for (int ii=0; ii<list->inum; ii++) {
    const int i = list->ilist[ii];
    node_i[ii] = i;
    node_types[ii] = mace_types[atom->type[i]-1];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    num_neigh[ii] = 0;
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        num_neigh[ii] += 1;
        const int j_local = atom->map(atom->tag[j]);
        neigh_j[ij] = j_local;
        neigh_indices[ij] = ii_from_i[j_local];
        neigh_types[ij] = mace_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }

  // ----- begin mace evaluation -----

  mace->compute_node_energies_forces(
    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r);

  // ----- end mace evaluation -----

  if (eflag_global) {
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += mace->node_energies[ii];
  }

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[node_i[ii]] = mace->node_energies[ii];
  }

  ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_i[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_j[ij];
      atom->f[i][0] -= mace->node_forces[3*ij];
      atom->f[i][1] -= mace->node_forces[3*ij+1];
      atom->f[i][2] -= mace->node_forces[3*ij+2];
      atom->f[j][0] += mace->node_forces[3*ij];
      atom->f[j][1] += mace->node_forces[3*ij+1];
      atom->f[j][2] += mace->node_forces[3*ij+2];
      ij += 1;
    }
  }

  if (vflag_global) {
    ij = 0;
    for (int ii=0; ii<num_nodes; ++ii) {
      for (int jj=0; jj<num_neigh[ii]; ++jj) {
        const double x = xyz[3*ij];
        const double y = xyz[3*ij+1];
        const double z = xyz[3*ij+2];
        const double f_x = mace->node_forces[3*ij];
        const double f_y = mace->node_forces[3*ij+1];
        const double f_z = mace->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x*f_z + z*f_x);
        virial[5] += 0.5*(y*f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace.");
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // count edges
  int num_edges = 0;
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared)
        num_edges += 1;
    }
  }

  // resize vectors
  const int num_nodes = list->inum;
  node_types.resize(num_nodes);
  num_neigh.resize(num_nodes);
  neigh_indices.resize(num_edges);
  neigh_types.resize(num_edges);
  xyz.resize(3*num_edges);
  r.resize(num_edges);
  node_i.resize(num_nodes);
  neigh_j.resize(num_edges);

  // populate neighbor list variables
  int ij = 0;
  for (int ii=0; ii<list->inum; ii++) {
    const int i = list->ilist[ii];
    node_i[ii] = i;
    node_types[ii] = mace_types[atom->type[i]-1];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    num_neigh[ii] = 0;
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        num_neigh[ii] += 1;
        neigh_j[ij] = j;
        neigh_types[ij] = mace_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }

  // ----- begin mace evaluation -----

  mace->node_energies.resize(num_nodes);
  std::fill(mace->node_energies.begin(), mace->node_energies.end(), 0.0);
  mace->node_forces.resize(xyz.size());
  std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

  if (mace->has_zbl)
    mace->zbl.compute_ZBL(
      num_nodes, node_types, num_neigh, neigh_types,
      mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

  mace->compute_Y(xyz);

  mace->compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_A0(num_nodes, node_types, num_neigh, neigh_types);
  mace->compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_nodes, node_types);
  mace->compute_H1(num_nodes);

  // sort local H1 contributions by i (rather than ii)
  H1.resize((atom->nlocal+atom->nghost)*mace->num_LM*mace->num_channels);
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    for (int k=0; k<mace->num_LM*mace->num_channels; ++k) {
      H1[i*mace->num_LM*mace->num_channels+k] = mace->H1[ii*mace->num_LM*mace->num_channels+k];
    }
  }
  comm->forward_comm(this);
  mace->H1 = H1;

  mace->compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_nodes, num_neigh, neigh_j);
  mace->compute_A1(num_nodes);
  mace->compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_nodes, node_types);
  mace->compute_H2(num_nodes, node_types);

  mace->compute_readouts(num_nodes, node_types);

  mace->reverse_H2(num_nodes, node_types, false);
  mace->reverse_M1(num_nodes, node_types);
  mace->reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
  mace->reverse_A1(num_nodes);
  mace->reverse_Phi1(num_nodes, num_neigh, neigh_j, xyz, r, false, false);

  H1_adj = mace->H1_adj;
  comm->reverse_comm(this);
  mace->H1_adj = H1_adj;

  mace->reverse_H1(num_nodes);
  mace->reverse_M0(num_nodes, node_types);
  mace->reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);

  // ----- end mace evaluation -----

  if (eflag_global) {
    for (int ii=0; ii<num_nodes; ++ii)
      eng_vdwl += mace->node_energies[ii];
  }

  if (eflag_atom) {
    for (int ii=0; ii<num_nodes; ++ii)
      eatom[node_i[ii]] = mace->node_energies[ii];
  }

  ij = 0;
  for (int ii=0; ii<num_nodes; ++ii) {
    const int i = node_i[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_j[ij];
      atom->f[i][0] -= mace->node_forces[3*ij];
      atom->f[i][1] -= mace->node_forces[3*ij+1];
      atom->f[i][2] -= mace->node_forces[3*ij+2];
      atom->f[j][0] += mace->node_forces[3*ij];
      atom->f[j][1] += mace->node_forces[3*ij+1];
      atom->f[j][2] += mace->node_forces[3*ij+2];
      ij += 1;
    }
  }

  if (vflag_global) {
    ij = 0;
    for (int ii=0; ii<num_nodes; ++ii) {
      for (int jj=0; jj<num_neigh[ii]; ++jj) {
        const double x = xyz[3*ij];
        const double y = xyz[3*ij+1];
        const double z = xyz[3*ij+2];
        const double f_x = mace->node_forces[3*ij];
        const double f_y = mace->node_forces[3*ij+1];
        const double f_z = mace->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x*f_z + z*f_x);
        virial[5] += 0.5*(y*f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace.");
}

/* ---------------------------------------------------------------------- */

void PairSymmetrixMACE::compute_no_mpi_message_passing(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  // locate ghosts within r_cut of locals
  is_local.resize(atom->nlocal+atom->nghost);
  std::fill(is_local.begin(), is_local.end(), false);
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    is_local[i] = true;
  }
  is_ghost.resize(atom->nlocal+atom->nghost);
  std::fill(is_ghost.begin(), is_ghost.end(), false);
  for (int ii=0; ii<list->inum; ++ii) {
    const int i = list->ilist[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared<r_cut_squared and not is_local[j])
        is_ghost[j] = true;
    }
  }

  // set num_local_nodes and num_ghost_nodes
  const int num_local_nodes = list->inum;
  const int num_ghost_nodes = std::reduce(is_ghost.begin(), is_ghost.end(), 0);

  // collect indices of ghosts within r_cut of locals
  ghost_indices.resize(num_ghost_nodes);
  int i = 0;
  for (int ii=0; ii<atom->nlocal+atom->nghost; ++ii)
    if (is_ghost[ii])
      ghost_indices[i++] = ii;

  // populate node_indices, node_types, and num_neigh
  node_i.resize(num_local_nodes+num_ghost_nodes);
  node_types.resize(num_local_nodes+num_ghost_nodes);
  num_neigh.resize(num_local_nodes+num_ghost_nodes);
  std::fill(num_neigh.begin(), num_neigh.end(), 0);
  ii_from_i.clear();
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ii++) {
    const int i = (ii<num_local_nodes) ? list->ilist[ii] : ghost_indices[ii-num_local_nodes];
    node_i[ii] = i;
    ii_from_i[i] = ii;
    node_types[ii] = mace_types[atom->type[i]-1];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared)
        num_neigh[ii] += 1;
    }
  }

  // count edges
  int num_local_edges = 0;
  for (int ii=0; ii<num_local_nodes; ++ii)
    num_local_edges += num_neigh[ii];
  int num_ghost_edges = 0;
  for (int ii=num_local_nodes; ii<num_local_nodes+num_ghost_nodes; ++ii)
    num_ghost_edges += num_neigh[ii];

  // populate neigh_indices, neigh_types, xyz, and r
  neigh_j.resize(num_local_edges+num_ghost_edges);
  neigh_indices.resize(num_local_edges+num_ghost_edges);
  neigh_types.resize(num_local_edges+num_ghost_edges);
  xyz.resize(3*(num_local_edges+num_ghost_edges));
  r.resize(num_local_edges+num_ghost_edges);
  int ij = 0;
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
    const int i = node_i[ii];
    const double x_i = atom->x[i][0];
    const double y_i = atom->x[i][1];
    const double z_i = atom->x[i][2];
    int* jlist = list->firstneigh[i];
    for (int jj=0; jj<list->numneigh[i]; jj++) {
      const int j = (jlist[jj] & NEIGHMASK);
      const double dx = atom->x[j][0] - x_i;
      const double dy = atom->x[j][1] - y_i;
      const double dz = atom->x[j][2] - z_i;
      const double r_squared = dx*dx + dy*dy + dz*dz;
      if (r_squared < r_cut_squared) {
        neigh_j[ij] = j;
        neigh_indices[ij] = ii_from_i[j];
        neigh_types[ij] = mace_types[atom->type[j]-1];
        xyz[3*ij] = dx;
        xyz[3*ij+1] = dy;
        xyz[3*ij+2] = dz;
        r[ij] = std::sqrt(r_squared);
        ij += 1;
      }
    }
  }


  // ----- begin mace evaluation -----

  mace->node_energies.resize(num_local_nodes);
  std::fill(mace->node_energies.begin(), mace->node_energies.end(), 0.0);
  mace->node_forces.resize(xyz.size());
  std::fill(mace->node_forces.begin(), mace->node_forces.end(), 0.0);

  if (mace->has_zbl)
    mace->zbl.compute_ZBL(
     num_local_nodes, node_types, num_neigh, neigh_types,
     mace->atomic_numbers, r, xyz, mace->node_energies, mace->node_forces);

  mace->compute_Y(xyz);

  mace->compute_R0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_A0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types);
  mace->compute_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->compute_H1(num_local_nodes+num_ghost_nodes);

  mace->compute_R1(num_local_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_Phi1(num_local_nodes, num_neigh, neigh_indices);
  mace->compute_A1(num_local_nodes);
  mace->compute_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, r);
  mace->compute_M1(num_local_nodes, node_types);
  mace->compute_H2(num_local_nodes, node_types);

  mace->compute_readouts(num_local_nodes, node_types);

  mace->reverse_H2(num_local_nodes, node_types, false);
  mace->reverse_M1(num_local_nodes, node_types);
  mace->reverse_A1_scaled(num_local_nodes, node_types, num_neigh, neigh_types, xyz, r, false);
  mace->reverse_A1(num_local_nodes);
  mace->reverse_Phi1(num_local_nodes, num_neigh, neigh_indices, xyz, r, false, false);

  mace->reverse_H1(num_local_nodes+num_ghost_nodes);
  mace->reverse_M0(num_local_nodes+num_ghost_nodes, node_types);
  mace->reverse_A0_scaled(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);
  mace->reverse_A0(num_local_nodes+num_ghost_nodes, node_types, num_neigh, neigh_types, xyz, r);

  // ----- end mace evaluation -----

  if (eflag_global)
    for (int ii=0; ii<num_local_nodes; ++ii)
      eng_vdwl += mace->node_energies[ii];

  if (eflag_atom)
    for (int ii=0; ii<num_local_nodes; ++ii)
      eatom[node_i[ii]] = mace->node_energies[ii];

  ij = 0;
  for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
    const int i = node_i[ii];
    for (int jj=0; jj<num_neigh[ii]; ++jj) {
      const int j = neigh_j[ij];
      atom->f[i][0] -= mace->node_forces[3*ij];
      atom->f[i][1] -= mace->node_forces[3*ij+1];
      atom->f[i][2] -= mace->node_forces[3*ij+2];
      atom->f[j][0] += mace->node_forces[3*ij];
      atom->f[j][1] += mace->node_forces[3*ij+1];
      atom->f[j][2] += mace->node_forces[3*ij+2];
      ij += 1;
    }
  }

  if (vflag_global) {
    ij = 0;
    for (int ii=0; ii<num_local_nodes+num_ghost_nodes; ++ii) {
      for (int jj=0; jj<num_neigh[ii]; ++jj) {
        const double x = xyz[3*ij];
        const double y = xyz[3*ij+1];
        const double z = xyz[3*ij+2];
        const double f_x = mace->node_forces[3*ij];
        const double f_y = mace->node_forces[3*ij+1];
        const double f_z = mace->node_forces[3*ij+2];
        virial[0] += x*f_x;
        virial[1] += y*f_y;
        virial[2] += z*f_z;
        virial[3] += 0.5*(x*f_y + y*f_x);
        virial[4] += 0.5*(x*f_z + z*f_x);
        virial[5] += 0.5*(y*f_z + z*f_y);
        ij += 1;
      }
    }
  }

  if (vflag_atom)
    error->all(FLERR, "Atomic virials not yet supported by pair_style symmetrix/mace.");
}

// Local re-evaluation of the force on atom i if displaced to xtrial, holding
// every other atom at its real position -- the many-body analog of single(),
// used by e.g. fix baoab/tether's mollify-yes Newton solve.
//
// MACE is a 2-layer message-passing model, so atom i's force is
// F_i = -sum_j dE_j/dx_i over every atom j whose OWN energy readout depends
// on x_i -- every j within 2*r_cut of i, since j's H2 (layer 2) aggregates
// its r_cut-neighbors' H1 (layer 1), and i can be a neighbor-of-a-neighbor
// of j. An earlier version of this function built a minimal subgraph with a
// readout for node 0 (i) only, which silently dropped every j != i
// contribution and gave wrong forces whenever another atom's receptive
// field reached back to i (confirmed by comparing against ground-truth
// atom->f[i] on the H/Pd testbed). Rather than widen that subgraph (real
// locality benefit only once r_cut << box size, not on today's testbeds;
// see the design doc for the tiered-subgraph alternative kept on file for
// when that changes), this function instead mirrors
// compute_no_mpi_message_passing()'s own node/edge construction and full
// compute/reverse chain verbatim -- readouts for every local atom, not just
// i -- with every position read routed through pos() instead of raw
// atom->x[], so the trial/clamp substitution below still applies. This
// reproduces compute_no_mpi_message_passing()'s own result exactly when
// clamp_groupbit == 0 and xtrial == atom->x[i], and costs about as much as
// one pair->compute() pass per call -- accepted for correctness given
// MACE's role here as an interim reference/teacher model, not a permanent
// production force field.
//
// Fresh local vectors are used throughout (not the class's own scratch
// members, which compute() owns), so this cannot alias a pair->compute()
// in flight. It does not touch atom->f, eng_vdwl, or virial, but it does
// overwrite mace's persistent internal scratch vectors (H1, A0, R0, ...),
// which compute() also owns.
//
// Hard invariant: only safe to call between timesteps' pair->compute()
// calls (fix_baoab_tether calls this from refresh_blocks()/solve_center(),
// never interleaved with compute()).
//
// Clamped-neighbor substitution: when clamp_groupbit != 0 and clamp_c !=
// nullptr, any atom j (j != i) with atom->mask[j] & clamp_groupbit uses
// clamp_c[j] instead of atom->x[j], matching the substitution the
// single()-based fallback already applies per-pair in
// FixBAOABTether::partial_force(). pos() is the single choke point for
// every position read in this function, so the substitution applies
// uniformly no matter how many hops separate the flagged atom from i --
// including cases the pairwise single() path could never express in the
// first place. atom->mask is valid for ghost indices (LAMMPS border comm
// carries it for every atom style); clamp_c is expected to already be
// forward-comm'd/PBC-shifted to match atom->x's image (see
// FixBAOABTether::pack_forward_comm()).
bool PairSymmetrixMACE::local_partial_force(int i, const double *xtrial, int jnum,
                                             const int *jlist, double *fout, double *fneigh,
                                             int clamp_groupbit, double *const *clamp_c)
{
  fout[0] = fout[1] = fout[2] = 0.0;
  if (fneigh)
    for (int jj = 0; jj < jnum; jj++) fneigh[3*jj] = fneigh[3*jj+1] = fneigh[3*jj+2] = 0.0;

  if (mode != "no_mpi_message_passing" || !list) return false;

  const double r_cut_squared = mace->r_cut*mace->r_cut;

  auto pos = [&](int idx, double *out) {
    if (idx == i) { out[0] = xtrial[0]; out[1] = xtrial[1]; out[2] = xtrial[2]; }
    else if (clamp_groupbit && (atom->mask[idx] & clamp_groupbit)) {
      out[0] = clamp_c[idx][0]; out[1] = clamp_c[idx][1]; out[2] = clamp_c[idx][2];
    }
    else { out[0] = atom->x[idx][0]; out[1] = atom->x[idx][1]; out[2] = atom->x[idx][2]; }
  };

  const int nall = atom->nlocal + atom->nghost;

  // locate ghosts within r_cut of locals -- mirrors
  // compute_no_mpi_message_passing()'s is_local/is_ghost discovery, with
  // every position read going through pos().
  std::vector<bool> l_is_local(nall, false);
  for (int ii = 0; ii < list->inum; ++ii) l_is_local[list->ilist[ii]] = true;

  std::vector<bool> l_is_ghost(nall, false);
  for (int ii = 0; ii < list->inum; ++ii) {
    const int li = list->ilist[ii];
    double xli[3]; pos(li, xli);
    int *jl = list->firstneigh[li];
    const int jn = list->numneigh[li];
    for (int jj = 0; jj < jn; jj++) {
      const int j = jl[jj] & NEIGHMASK;
      double xj[3]; pos(j, xj);
      const double dx = xj[0]-xli[0], dy = xj[1]-xli[1], dz = xj[2]-xli[2];
      if ((dx*dx+dy*dy+dz*dz < r_cut_squared) && !l_is_local[j]) l_is_ghost[j] = true;
    }
  }

  const int num_local_nodes = list->inum;
  int num_ghost_nodes = 0;
  for (int ii = 0; ii < nall; ++ii)
    if (l_is_ghost[ii]) num_ghost_nodes++;

  std::vector<int> l_ghost_indices(num_ghost_nodes);
  {
    int k = 0;
    for (int ii = 0; ii < nall; ++ii)
      if (l_is_ghost[ii]) l_ghost_indices[k++] = ii;
  }

  const int num_nodes = num_local_nodes + num_ghost_nodes;
  std::vector<int> l_node_i(num_nodes), l_node_types(num_nodes), l_num_neigh(num_nodes, 0);
  std::unordered_map<int,int> l_ii_from_i;
  l_ii_from_i.reserve(2*num_nodes);
  for (int ii = 0; ii < num_nodes; ii++) {
    const int ni = (ii < num_local_nodes) ? list->ilist[ii] : l_ghost_indices[ii-num_local_nodes];
    l_node_i[ii] = ni;
    l_ii_from_i[ni] = ii;
    l_node_types[ii] = mace_types[atom->type[ni]-1];
    double xn[3]; pos(ni, xn);
    int *jl = list->firstneigh[ni];
    const int jn = list->numneigh[ni];
    int cnt = 0;
    for (int jj = 0; jj < jn; jj++) {
      const int j = jl[jj] & NEIGHMASK;
      double xj[3]; pos(j, xj);
      const double dx = xj[0]-xn[0], dy = xj[1]-xn[1], dz = xj[2]-xn[2];
      if (dx*dx+dy*dy+dz*dz < r_cut_squared) cnt++;
    }
    l_num_neigh[ii] = cnt;
  }

  int num_local_edges = 0;
  for (int ii = 0; ii < num_local_nodes; ++ii) num_local_edges += l_num_neigh[ii];
  int num_ghost_edges = 0;
  for (int ii = num_local_nodes; ii < num_nodes; ++ii) num_ghost_edges += l_num_neigh[ii];

  const int num_edges = num_local_edges + num_ghost_edges;
  std::vector<int> l_neigh_j(num_edges), l_neigh_indices(num_edges), l_neigh_types(num_edges);
  std::vector<double> l_xyz(3*num_edges), l_r(num_edges);
  {
    int ij = 0;
    for (int ii = 0; ii < num_nodes; ++ii) {
      const int ni = l_node_i[ii];
      double xn[3]; pos(ni, xn);
      int *jl = list->firstneigh[ni];
      const int jn = list->numneigh[ni];
      for (int jj = 0; jj < jn; jj++) {
        const int j = jl[jj] & NEIGHMASK;
        double xj[3]; pos(j, xj);
        const double dx = xj[0]-xn[0], dy = xj[1]-xn[1], dz = xj[2]-xn[2];
        const double rsq = dx*dx+dy*dy+dz*dz;
        if (rsq >= r_cut_squared) continue;
        l_neigh_j[ij] = j;
        l_neigh_indices[ij] = l_ii_from_i[j];
        l_neigh_types[ij] = mace_types[atom->type[j]-1];
        l_xyz[3*ij] = dx; l_xyz[3*ij+1] = dy; l_xyz[3*ij+2] = dz;
        l_r[ij] = std::sqrt(rsq);
        ij += 1;
      }
    }
  }

  mace->node_energies.assign(num_local_nodes, 0.0);
  mace->node_forces.assign(l_xyz.size(), 0.0);
  if (mace->has_zbl)
    mace->zbl.compute_ZBL(num_local_nodes, l_node_types, l_num_neigh, l_neigh_types,
                           mace->atomic_numbers, l_r, l_xyz, mace->node_energies,
                           mace->node_forces);

  mace->compute_Y(l_xyz);
  mace->compute_R0(num_nodes, l_node_types, l_num_neigh, l_neigh_types, l_r);
  mace->compute_A0(num_nodes, l_node_types, l_num_neigh, l_neigh_types);
  mace->compute_A0_scaled(num_nodes, l_node_types, l_num_neigh, l_neigh_types, l_r);
  mace->compute_M0(num_nodes, l_node_types);
  mace->compute_H1(num_nodes);

  mace->compute_R1(num_local_nodes, l_node_types, l_num_neigh, l_neigh_types, l_r);
  mace->compute_Phi1(num_local_nodes, l_num_neigh, l_neigh_indices);
  mace->compute_A1(num_local_nodes);
  mace->compute_A1_scaled(num_local_nodes, l_node_types, l_num_neigh, l_neigh_types, l_r);
  mace->compute_M1(num_local_nodes, l_node_types);
  mace->compute_H2(num_local_nodes, l_node_types);
  mace->compute_readouts(num_local_nodes, l_node_types);

  mace->reverse_H2(num_local_nodes, l_node_types, false);
  mace->reverse_M1(num_local_nodes, l_node_types);
  mace->reverse_A1_scaled(num_local_nodes, l_node_types, l_num_neigh, l_neigh_types, l_xyz, l_r,
                           false);
  mace->reverse_A1(num_local_nodes);
  mace->reverse_Phi1(num_local_nodes, l_num_neigh, l_neigh_indices, l_xyz, l_r, false, false);

  mace->reverse_H1(num_nodes);
  mace->reverse_M0(num_nodes, l_node_types);
  mace->reverse_A0_scaled(num_nodes, l_node_types, l_num_neigh, l_neigh_types, l_xyz, l_r);
  mace->reverse_A0(num_nodes, l_node_types, l_num_neigh, l_neigh_types, l_xyz, l_r);

  // Accumulate into a buffer indexed by RAW atom index (mirroring how
  // compute_no_mpi_message_passing() accumulates into atom->f directly),
  // not by node index: a ghost-tier node's own neighbor ("hop-3", raw edge
  // data only) need not itself be a node, so mapping it back through
  // l_ii_from_i would either need a defensive check on every edge or
  // silently misattribute the reaction force via operator[]'s "insert a 0"
  // behavior on a missing key. Sizing by nall and indexing directly with
  // node_i[]/neigh_j[]'s raw indices sidesteps the issue entirely.
  std::vector<double> local_force(3*nall, 0.0);
  {
    int ij = 0;
    for (int ii = 0; ii < num_nodes; ++ii) {
      const int ni = l_node_i[ii];
      for (int jj = 0; jj < l_num_neigh[ii]; ++jj) {
        const int j = l_neigh_j[ij];
        local_force[3*ni]   -= mace->node_forces[3*ij];
        local_force[3*ni+1] -= mace->node_forces[3*ij+1];
        local_force[3*ni+2] -= mace->node_forces[3*ij+2];
        local_force[3*j]    += mace->node_forces[3*ij];
        local_force[3*j+1]  += mace->node_forces[3*ij+1];
        local_force[3*j+2]  += mace->node_forces[3*ij+2];
        ij += 1;
      }
    }
  }

  // compute_no_mpi_message_passing() accumulates the final force the same
  // way (raw-index, including ghost-tier nodes' own edges), and relies on
  // the outer Verlet loop's comm->reverse_comm() -- called after
  // pair->compute() returns -- to fold each ghost atom's accumulated
  // partial force back onto its true local owner. This function never
  // goes through that path, so it must fold locally: with the periodic
  // box smaller than 2*r_cut, an atom's own ghost image is routinely a
  // separate node/edge-target in this graph and can carry a real share of
  // that atom's total force.
  for (int idx = atom->nlocal; idx < nall; ++idx) {
    if (local_force[3*idx] == 0.0 && local_force[3*idx+1] == 0.0 && local_force[3*idx+2] == 0.0)
      continue;
    const int owner = atom->map(atom->tag[idx]);
    if (owner < 0) continue;
    local_force[3*owner]   += local_force[3*idx];
    local_force[3*owner+1] += local_force[3*idx+1];
    local_force[3*owner+2] += local_force[3*idx+2];
  }

  fout[0] = local_force[3*i]; fout[1] = local_force[3*i+1]; fout[2] = local_force[3*i+2];

  if (fneigh) {
    for (int jj = 0; jj < jnum; jj++) {
      const int j = jlist[jj] & NEIGHMASK;
      const int j_owner = (j < atom->nlocal) ? j : atom->map(atom->tag[j]);
      const int jr = (j_owner >= 0) ? j_owner : j;
      fneigh[3*jj]   = local_force[3*jr];
      fneigh[3*jj+1] = local_force[3*jr+1];
      fneigh[3*jj+2] = local_force[3*jr+2];
    }
  }

  return true;
}
