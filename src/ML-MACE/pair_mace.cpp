/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributors
      William C Witt (University of Cambridge)
------------------------------------------------------------------------- */

// TODO: add check for simulation units--if not metal as expected by pytorch/mace model,
// convert to metal and then convert back after force computation. 

#include "pair_mace.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "comm.h"
#include "update.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairMACE::PairMACE(LAMMPS *lmp) : Pair(lmp)
{
  no_virial_fdotr_compute = 1;
}

/* ---------------------------------------------------------------------- */

PairMACE::~PairMACE()
{
}

/* ---------------------------------------------------------------------- */

void PairMACE::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);
/*
  if (atom->nlocal != list->inum) error->all(FLERR, "ERROR: nlocal != inum.");
  if (domain_decomposition) {
    if (atom->nghost != list->gnum) error->all(FLERR, "ERROR: nghost != gnum.");
  }
*/

/*
  // Debug prints for atom counts
  if (comm->me == 0) {
    printf("DEBUG: atom counts - nlocal: %d, nghost: %d, inum: %d, gnum: %d\n", 
           atom->nlocal, atom->nghost, list->inum, list->gnum);
  }
*/

  // Check if unit conversion is needed - MACE expects metal units
  if (need_unit_conversion) {
    // Convert positions from current units to metal
    // This affects atom->x which is in distance units
    for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
      atom->x[i][0] *= distance_conv_factor;
      atom->x[i][1] *= distance_conv_factor;
      atom->x[i][2] *= distance_conv_factor;
    }
    
    // Convert box dimensions
    domain->h[0] *= distance_conv_factor;  // xprd
    domain->h[1] *= distance_conv_factor;  // yprd
    domain->h[2] *= distance_conv_factor;  // zprd
    domain->h[3] *= distance_conv_factor;  // xy
    domain->h[4] *= distance_conv_factor;  // xz
    domain->h[5] *= distance_conv_factor;  // yz
    
    // Need to update h_inv for correct periodic image calculations
    domain->h_inv[0] = 1.0/domain->h[0];
    domain->h_inv[1] = 1.0/domain->h[1];
    domain->h_inv[2] = 1.0/domain->h[2];
    domain->h_inv[3] = -domain->h[3]/(domain->h[1]*domain->h[2]);
    domain->h_inv[4] = (domain->h[3]*domain->h[5]/domain->h[1] - domain->h[4])/(domain->h[0]*domain->h[2]);
    domain->h_inv[5] = -domain->h[5]/(domain->h[0]*domain->h[1]);
  }

  // ----- positions -----
  int cand_nodes = domain_decomposition ?
                 list->inum + list->gnum :
                 list->inum;

  // 2. allocate with the *maximum* size first
  std::vector<int> atom_to_idx(atom->nmax, -1);
  std::vector<int> idx_to_atom(cand_nodes);

  // 3. copy only atoms that belong to the nnp group
  int seq = 0;
  for (int ii = 0; ii < cand_nodes; ++ii) {
    int i = list->ilist[ii];
    if (!(atom->mask[i] & groupbit_nnp)) continue;  // skip non-ML
    atom_to_idx[i] = seq;
    idx_to_atom[seq] = i;
    ++seq;
  }
  int n_nodes = seq;              // *** final ML-only count ***
  
  
  // 4. now you can create the positions tensor
  auto positions = torch::empty({n_nodes,3}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii = 0; ii < n_nodes; ++ii) {
    int i = idx_to_atom[ii];
    positions[ii][0] = atom->x[i][0];
    positions[ii][1] = atom->x[i][1];
    positions[ii][2] = atom->x[i][2];
  }

/*
  // Debug prints for atom mapping
  if (comm->me == 0) {
    printf("DEBUG: atom mapping created for %d nodes\n", n_nodes);
  }
*/

/*
  // Debug prints for list indices
  if (comm->me == 0) {
    printf("DEBUG: list indices - n_nodes: %d, first few ilist entries:", n_nodes);
    for (int i = 0; i < std::min(5, n_nodes); i++) {
      printf(" %d", list->ilist[i]);
    }
    printf("\n");
  }
*/

  // ----- cell -----
  auto cell = torch::zeros({3,3}, torch_float_dtype);
  cell[0][0] = domain->h[0];
  cell[0][1] = 0.0;
  cell[0][2] = 0.0;
  cell[1][0] = domain->h[5];
  cell[1][1] = domain->h[1];
  cell[1][2] = 0.0;
  cell[2][0] = domain->h[4];
  cell[2][1] = domain->h[3];
  cell[2][2] = domain->h[2];

  // ----- edge_index and unit_shifts -----
  // count total number of edges
  int n_edges = 0;
  std::vector<int> n_edges_vec(n_nodes, 0);
  #pragma omp parallel for reduction(+:n_edges)
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = idx_to_atom[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    for (int jj=0; jj<jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (!(atom->mask[j] & groupbit_nnp)) continue;   // skip neighbour not in nnp
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max_squared) {
        n_edges += 1;
        n_edges_vec[ii] += 1;
      }
    }
  }
  
  // make first_edge vector to help with parallelizing following loop
  std::vector<int> first_edge(n_nodes);
  first_edge[0] = 0;
  for (int ii=0; ii<n_nodes-1; ++ii) {
    first_edge[ii+1] = first_edge[ii] + n_edges_vec[ii];
  }
  // fill edge_index and unit_shifts tensors
  auto edge_index = torch::empty({2,n_edges}, torch::dtype(torch::kInt64));
  auto unit_shifts = torch::zeros({n_edges,3}, torch_float_dtype);
  auto shifts = torch::zeros({n_edges,3}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = idx_to_atom[ii];
    double xtmp = atom->x[i][0];
    double ytmp = atom->x[i][1];
    double ztmp = atom->x[i][2];
    int *jlist = list->firstneigh[i];
    int jnum = list->numneigh[i];
    int k = first_edge[ii];
    for (int jj=0; jj<jnum; ++jj) {
      int j = jlist[jj];
      j &= NEIGHMASK;
      double delx = xtmp - atom->x[j][0];
      double dely = ytmp - atom->x[j][1];
      double delz = ztmp - atom->x[j][2];
      double rsq = delx * delx + dely * dely + delz * delz;
      if (rsq < r_max_squared) {
        edge_index[0][k] = ii; // Use sequential index for source node
        if (domain_decomposition) {
          // Check if j is in our mapping, if not add it
          if (atom_to_idx[j] == -1) {
            printf("WARNING: atom %d not found in mapping\n", j);
            edge_index[1][k] = 0; // Use a default value (could be problematic)
          } else {
            edge_index[1][k] = atom_to_idx[j]; // Use sequential index for target node
          }
        } else {
          int j_local = atom->map(atom->tag[j]);
          int j_idx = atom_to_idx[j_local];
          if (j_idx == -1) {
            printf("WARNING: atom %d (local %d) not found in mapping\n", j, j_local);
            edge_index[1][k] = 0; // Use a default value (could be problematic)
          } else {
            edge_index[1][k] = j_idx; // Use sequential index for target node
          }
          double shiftx = atom->x[j][0] - atom->x[j_local][0];
          double shifty = atom->x[j][1] - atom->x[j_local][1];
          double shiftz = atom->x[j][2] - atom->x[j_local][2];
          double shiftxs = std::round(domain->h_inv[0]*shiftx + domain->h_inv[5]*shifty + domain->h_inv[4]*shiftz);
          double shiftys = std::round(domain->h_inv[1]*shifty + domain->h_inv[3]*shiftz);
          double shiftzs = std::round(domain->h_inv[2]*shiftz);
          unit_shifts[k][0] = shiftxs;
          unit_shifts[k][1] = shiftys;
          unit_shifts[k][2] = shiftzs;
          shifts[k][0] = domain->h[0]*shiftxs + domain->h[5]*shiftys + domain->h[4]*shiftzs;
          shifts[k][1] = domain->h[1]*shiftys + domain->h[3]*shiftzs;
          shifts[k][2] = domain->h[2]*shiftzs;
        }
        k++;
      }
    }
  }

/*
  // Debug prints for edge indices
  if (comm->me == 0 && n_edges > 0) {
    printf("DEBUG: edge indices - n_edges: %d, first edge: (%d, %d)\n",
           n_edges, 
           edge_index[0][0].item<int64_t>(),
           edge_index[1][0].item<int64_t>());
  }
*/

  // ----- node_attrs -----
  int n_node_feats = mace_atomic_numbers.size();
  auto node_attrs = torch::zeros({n_nodes,n_node_feats}, torch_float_dtype);
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = idx_to_atom[ii];
    node_attrs[ii][mace_type(atom->type[i])-1] = 1.0;
  }

  // ----- mask for ghost -----
  auto mask = torch::zeros(n_nodes, torch::dtype(torch::kBool));
  #pragma omp parallel for
  for (int ii=0; ii<list->inum; ++ii) {
    mask[ii] = true;
  }

  auto batch = torch::zeros({n_nodes}, torch::dtype(torch::kInt64));
  auto energy = torch::empty({1}, torch_float_dtype);
  auto forces = torch::empty({n_nodes,3}, torch_float_dtype);
  auto ptr = torch::empty({2}, torch::dtype(torch::kInt64));
  auto weight = torch::empty({1}, torch_float_dtype);
  ptr[0] = 0;
  ptr[1] = n_nodes;
  weight[0] = 1.0;

  // transfer data to device
  batch = batch.to(device);
  cell = cell.to(device);
  edge_index = edge_index.to(device);
  energy = energy.to(device);
  forces = forces.to(device);
  node_attrs = node_attrs.to(device);
  positions = positions.to(device);
  ptr = ptr.to(device);
  shifts = shifts.to(device);
  unit_shifts = unit_shifts.to(device);
  weight = weight.to(device);

  // pack the input, call the model
  c10::Dict<std::string, torch::Tensor> input;
  input.insert("batch", batch);
  input.insert("cell", cell);
  input.insert("edge_index", edge_index);
  input.insert("energy", energy);
  input.insert("forces", forces);
  input.insert("node_attrs", node_attrs);
  input.insert("positions", positions);
  input.insert("ptr", ptr);
  input.insert("shifts", shifts);
  input.insert("unit_shifts", unit_shifts);
  input.insert("weight", weight);
  auto output = model.forward({input, mask.to(device), bool(vflag_global)}).toGenericDict();

  // mace energy
  //   -> sum of site energies of local atoms
  if (eflag_global) {
    energy = output.at("total_energy_local").toTensor().cpu();
    eng_vdwl += energy.item<double>() * (need_unit_conversion ? energy_conv_factor : 1.0);
  }

  // mace forces
  //   -> derivatives of total mace energy
  forces = output.at("forces").toTensor().cpu();
  #pragma omp parallel for
  for (int ii=0; ii<n_nodes; ++ii) {
    int i = idx_to_atom[ii];
    double force_factor = need_unit_conversion ? force_conv_factor : 1.0;
    atom->f[i][0] += forces[ii][0].item<double>() * force_factor;
    atom->f[i][1] += forces[ii][1].item<double>() * force_factor;
    atom->f[i][2] += forces[ii][2].item<double>() * force_factor;
  }

/*
  // Debug print forces
  if (comm->me == 0) {
    printf("DEBUG: Forces after accumulation:\n");
    for (int ii=0; ii<n_nodes; ++ii) {
      int i = idx_to_atom[ii];
      printf("  Atom %d (local %d): fx=%g fy=%g fz=%g\n", 
             atom->tag[i], i, 
             atom->f[i][0], atom->f[i][1], atom->f[i][2]);
    }
  }
*/


  // mace site energies
  //   -> local atoms only
  if (eflag_atom) {
    auto node_energy = output.at("node_energy").toTensor().cpu();
    #pragma omp parallel for
    for (int ii=0; ii<list->inum; ++ii) {
      int i = idx_to_atom[ii];
      eatom[i] = node_energy[ii].item<double>() * (need_unit_conversion ? energy_conv_factor : 1.0);
    }
  }

  // mace virials (local atoms only)
  //   -> derivatives of sum of site energies of local atoms
  if (vflag_global) {
    auto vir = output.at("virials").toTensor().cpu();
    double virial_factor = need_unit_conversion ? energy_conv_factor : 1.0;
    virial[0] += vir[0][0][0].item<double>() * virial_factor;
    virial[1] += vir[0][1][1].item<double>() * virial_factor;
    virial[2] += vir[0][2][2].item<double>() * virial_factor;
    virial[3] += 0.5*(vir[0][1][0].item<double>() + vir[0][0][1].item<double>()) * virial_factor;
    virial[4] += 0.5*(vir[0][2][0].item<double>() + vir[0][0][2].item<double>()) * virial_factor;
    virial[5] += 0.5*(vir[0][2][1].item<double>() + vir[0][1][2].item<double>()) * virial_factor;
  }

  // mace site virials
  //   -> not available
  if (vflag_atom) {
    error->all(FLERR, "ERROR: pair_mace does not support vflag_atom.");
  }
  
  // Convert back to original units if needed
  if (need_unit_conversion) {
    // Convert positions back to original units
    for (int i = 0; i < atom->nlocal + atom->nghost; i++) {
      atom->x[i][0] /= distance_conv_factor;
      atom->x[i][1] /= distance_conv_factor;
      atom->x[i][2] /= distance_conv_factor;
    }
    
    // Convert box dimensions back
    domain->h[0] /= distance_conv_factor;
    domain->h[1] /= distance_conv_factor;
    domain->h[2] /= distance_conv_factor;
    domain->h[3] /= distance_conv_factor;
    domain->h[4] /= distance_conv_factor;
    domain->h[5] /= distance_conv_factor;
    
    // Restore h_inv
    domain->h_inv[0] = 1.0/domain->h[0];
    domain->h_inv[1] = 1.0/domain->h[1];
    domain->h_inv[2] = 1.0/domain->h[2];
    domain->h_inv[3] = -domain->h[3]/(domain->h[1]*domain->h[2]);
    domain->h_inv[4] = (domain->h[3]*domain->h[5]/domain->h[1] - domain->h[4])/(domain->h[0]*domain->h[2]);
    domain->h_inv[5] = -domain->h[5]/(domain->h[0]*domain->h[1]);
  }
}

/* ---------------------------------------------------------------------- */

void PairMACE::settings(int narg, char **arg)
{
  if (narg > 1) {
    error->all(FLERR, "Too many pair_style arguments for pair_style mace.");
  }

  if (narg == 1) {
    if (strcmp(arg[0], "no_domain_decomposition") == 0) {
      domain_decomposition = false;
      // TODO: add check against MPI rank
    } else {
      error->all(FLERR, "Unrecognized argument for pair_style mace.");
    }
  }
}

/* ---------------------------------------------------------------------- */

void PairMACE::coeff(int narg, char **arg)
{
  // TODO: remove print statements from this routine, or have a single proc print

  if (!allocated) allocate();

  if (!torch::cuda::is_available()) {
    std::cout << "CUDA unavailable, setting device type to torch::kCPU." << std::endl;
    device = c10::Device(torch::kCPU);
  } else {
    std::cout << "CUDA found, setting device type to torch::kCUDA." << std::endl;
    MPI_Comm local;
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localrank;
    MPI_Comm_rank(local, &localrank);
    device = c10::Device(torch::kCUDA,localrank);
  }

  std::cout << "Loading MACE model from \"" << arg[2] << "\" ...";
  model = torch::jit::load(arg[2], device);
  std::cout << " finished." << std::endl;

  // extract default dtype from mace model
  for (auto p: model.named_attributes()) {
      // this is a somewhat random choice of variable to check. could it be improved?
      if (p.name == "model.node_embedding.linear.weight") {
          if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<float>()) {
            torch_float_dtype = torch::kFloat32;
          } else if (p.value.toTensor().dtype() == caffe2::TypeMeta::Make<double>()) {
            torch_float_dtype = torch::kFloat64;
          }
      }
  }
  std::cout << "  - The torch_float_dtype is: " << torch_float_dtype << std::endl;

  // extract r_max from mace model
  r_max = model.attr("r_max").toTensor().item<double>();
  r_max_squared = r_max*r_max;
  std::cout << "  - The r_max is: " << r_max << "." << std::endl;
  num_interactions = model.attr("num_interactions").toTensor().item<int64_t>();
  std::cout << "  - The model has: " << num_interactions << " layers." << std::endl;

  // extract atomic numbers from mace model
  auto a_n = model.attr("atomic_numbers").toTensor();
  for (int i=0; i<a_n.size(0); ++i) {
    mace_atomic_numbers.push_back(a_n[i].item<int64_t>());
  }
  std::cout << "  - The MACE model atomic numbers are: " << mace_atomic_numbers << "." << std::endl;

  // extract atomic numbers from pair_coeff
  for (int i=3; i<narg; ++i) {
    auto iter = std::find(periodic_table.begin(), periodic_table.end(), arg[i]);
    int index = std::distance(periodic_table.begin(), iter) + 1;
    lammps_atomic_numbers.push_back(index);
  }
  std::cout << "  - The pair_coeff atomic numbers are: " << lammps_atomic_numbers << "." << std::endl;

  for (int i=1; i<=lammps_atomic_numbers.size(); ++i) {
    std::cout << "  - Mapping LAMMPS type " << i
      << " (" << periodic_table[lammps_atomic_numbers[i-1]-1]
      << ") to MACE type " << mace_type(i) << "." << std::endl;
  }

  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 1;
}

void PairMACE::init_style()
{
  if (force->newton_pair == 0) error->all(FLERR, "ERROR: Pair style mace requires newton pair on.");

  /*
    MACE requires the full neighbor list AND neighbors of ghost atoms
    it appears that:
      * without REQ_GHOST
           list->gnum == 0
           list->ilist does not include ghost atoms, but the jlists do
      * with REQ_GHOST
           list->gnum == atom->nghost
           list->ilist includes ghost atoms
  */
  if (domain_decomposition) {
    neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_GHOST);
  } else {
    neighbor->add_request(this, NeighConst::REQ_FULL);
  }
  
  // cache the bitmask for the user-defined ML group called “nnp”
  int igroup = group->find("nnp");
  if (igroup < 0)
    error->all(FLERR,
      "Pair MACE: you must define a group called 'nnp' that contains the ML atoms");
  groupbit_nnp = group->bitmask[igroup];


  // Set up unit conversion if not using metal units
  // MACE models expect metal units (eV for energy, Angstrom for distance)
  need_unit_conversion = (strcmp(update->unit_style, "metal") != 0);
  
  if (need_unit_conversion) {
    if (comm->me == 0) {
      std::cout << "MACE models are trained using metal units (eV, Angstrom)." << std::endl;
      std::cout << "Current simulation is using '" << update->unit_style 
                << "' units. Unit conversion will be applied." << std::endl;
    }
    
    // Set up conversion factors based on the current unit style
    if (strcmp(update->unit_style, "lj") == 0) {
      // LJ units to metal: energies, distances, forces
      energy_conv_factor = force->boltz;        // LJ energy to eV
      distance_conv_factor = 1.0;               // Need to be set based on the LJ parameters
      force_conv_factor = energy_conv_factor / distance_conv_factor;
      error->all(FLERR, "Unit conversion from LJ to metal not fully supported yet in pair_mace.");
    } 
    else if (strcmp(update->unit_style, "real") == 0) {
      // Real units to metal: energies (kcal/mol to eV), distances (Angstrom same)
      energy_conv_factor = 1.0 / 23.060549;     // kcal/mol to eV (1 eV = 23.060549 kcal/mol)
      distance_conv_factor = 1.0;               // Both are Angstrom
      force_conv_factor = energy_conv_factor / distance_conv_factor;
    }
    else if (strcmp(update->unit_style, "si") == 0) {
      // SI units to metal: energies (J to eV), distances (m to Angstrom)
      energy_conv_factor = 1.0 / 1.602176634e-19; // J to eV
      distance_conv_factor = 1.0e10;              // m to Angstrom
      force_conv_factor = energy_conv_factor / distance_conv_factor;
    }
    else if (strcmp(update->unit_style, "cgs") == 0) {
      // CGS units to metal: energies (erg to eV), distances (cm to Angstrom)
      energy_conv_factor = 1.0 / 1.602176634e-12; // erg to eV
      distance_conv_factor = 1.0e8;               // cm to Angstrom
      force_conv_factor = energy_conv_factor / distance_conv_factor;
    }
    else if (strcmp(update->unit_style, "electron") == 0) {
      // Electron units to metal: energies (Hartree to eV), distances (Bohr to Angstrom)
      energy_conv_factor = 27.211386245988;       // Hartree to eV
      distance_conv_factor = 0.529177210903;      // Bohr to Angstrom
      force_conv_factor = energy_conv_factor / distance_conv_factor;
    }
    else if (strcmp(update->unit_style, "micro") == 0) {
      // Micro units to metal: energies (picogram-micrometer^2/microsecond^2 to eV), distances
      energy_conv_factor = 6.02214076e-1;        // pg-um²/us² to eV (approx)
      distance_conv_factor = 1.0e4;              // um to Angstrom
      force_conv_factor = energy_conv_factor / distance_conv_factor;
    }
    else if (strcmp(update->unit_style, "nano") == 0) {
      // Nano units to metal: energies (attogram-nanometer^2/nanosecond^2 to eV), distances
      energy_conv_factor = 6.02214076e-4;        // ag-nm²/ns² to eV (approx)
      distance_conv_factor = 10.0;               // nm to Angstrom
      force_conv_factor = energy_conv_factor / distance_conv_factor;
    }
    else {
      error->all(FLERR, "Unit conversion to metal not supported for this unit style in pair_mace.");
    }
    
    if (comm->me == 0) {
      std::cout << "MACE unit conversion factors:" << std::endl;
      std::cout << "  Energy:   " << energy_conv_factor << std::endl;
      std::cout << "  Distance: " << distance_conv_factor << std::endl;
      std::cout << "  Force:    " << force_conv_factor << std::endl;
    }
  }
}

double PairMACE::init_one(int i, int j)
{
  // to account for message passing, require cutoff of n_layers * r_max
  return num_interactions*model.attr("r_max").toTensor().item<double>();
}

void PairMACE::allocate()
{
  allocated = 1;

  memory->create(setflag, atom->ntypes+1, atom->ntypes+1, "pair:setflag");
  for (int i=1; i<atom->ntypes+1; i++)
    for (int j=i; j<atom->ntypes+1; j++)
      setflag[i][j] = 0;

  memory->create(cutsq, atom->ntypes+1, atom->ntypes+1, "pair:cutsq");
}

int PairMACE::mace_type(int lammps_type)
{
    for (int i=0; i<mace_atomic_numbers.size(); ++i) {
      if (mace_atomic_numbers[i]==lammps_atomic_numbers[lammps_type-1]) {
        return i+1;
      }
    }
    error->all(FLERR, "Problem converting lammps_type to mace_type.");
    return -1;
 }
