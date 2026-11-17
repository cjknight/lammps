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
#include "modify.h"
#include "pair.h"

#include <cstring>

using namespace LAMMPS_NS;

// FixBAOAB is constructed with a literal narg of 7 (it would reject the derived styles'
// keywords), so the required arguments are checked here first: without this check a too-short
// command line would make FixBAOAB read past the end of the argument array
int FixBAOABTetherBase::base_narg(LAMMPS *lmp, int narg, const char *style)
{
  if (narg < 7) utils::missing_cmd_args(FLERR, style, lmp->error);
  return 7;
}

FixBAOABTetherBase::FixBAOABTetherBase(LAMMPS *lmp, int narg, char **arg, const char *style) :
    FixBAOAB(lmp, base_narg(lmp, narg, style), arg), n_recompute_calls(0)
{
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
