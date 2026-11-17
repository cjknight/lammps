/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_FIX_BAOAB_TETHER_BASE_H
#define LMP_FIX_BAOAB_TETHER_BASE_H

#include "fix_baoab.h"

#include <string>
#include <vector>

namespace LAMMPS_NS {

// Shared layer of the fix baoab/tether family (baoab/tether, baoab/molblock, baoab/hrespa):
// - out-of-band whole-system force evaluations at scratch configurations, and saving/restoring
//   what such an evaluation overwrites (forces, charges set by the pair style, accumulated
//   energies and virials);
// - the per-atom force kernel (partial_force(): pair local_partial_force()/single() or a
//   whole-system recompute, plus bonds/angles) and the finite-difference curvature probe built
//   on it (probe_atom(), collect_partners()), with neighbor clamping as a virtual hook.
// Not a style itself.

class FixBAOABTetherBase : public FixBAOAB {
 protected:
  // style is "Fix <style>" for error messages; FixBAOAB parses only the 4 required arguments
  FixBAOABTetherBase(class LAMMPS *, int, char **, const char *style);

  bigint n_recompute_calls;    // recompute_forces_local() calls

  void force_clear_local();
  // all force contributions at the current positions (ghosts must be current), as the
  // integrator computes them, without energies or virials
  void recompute_forces_local();

  // charges set by the pair style at every compute (Pair::kspace_response_flag, e.g. an LES
  // model): a scratch recompute leaves atom->q at the scratch geometry, so callers save the
  // real-geometry charges first and restore them afterwards
  bool charges_from_pair() const;
  void save_charges(std::vector<double> &qs) const;
  void restore_charges(const std::vector<double> &qs);

  // ---- per-atom force kernel and curvature probe (styles that probe single atoms) ----
  const char *style_name;                // "Fix <style>", for error messages
  class NeighList *list;                 // full neighbor list (requested by the style)
  void init_list(int, class NeighList *) override;
  double eps;                            // finite-difference probe displacement
  int eps_was_set;                       // 1 if the user gave "eps"
  int use_local_partial_force;           // 0 = pair->single() loop, 1 = pair->local_partial_force()
  // force_eval full (default off, byte-identical when unused): validation-
  // only mode that bypasses partial_force()'s local kernel entirely --
  // for EVERY caller (solve_center()'s Newton residual, refresh_blocks()/
  // refresh_one_atom()'s FD-Hessian curvature probe, sub_step_free_atom()'s
  // exile sub-stepping) --
  // substituting a genuine whole-system recompute_forces_local() per trial
  // evaluation instead (see full_system_force()). O(N) per probe and
  // single-process only -- not a production option, see
  // fix_baoab_tether.rst. Deliberately NOT performance-tuned: this exists
  // to find out whether local_partial_force()'s neighbor-shell truncation
  // of the EAM embedding term explains a curvature discrepancy against a
  // full-system reference (e.g. lj_longstep_prototype.py's FD Hessian),
  // not to be fast -- keep debug runs short.
  int force_eval_full;

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

  void init_probe_kernels();
  // force on atom i if it were at xtrial (pair part over its neighbor list + bonded part);
  // live_neighbor_positions = false substitutes clamp_centers()[j] for neighbors with
  // clamp_neighbor(j)
  bool partial_force(int i, const double *xtrial, double *fout, double *fneigh = nullptr,
                     bool live_neighbor_positions = false, tagint *bond_tag = nullptr,
                     double *bond_fneigh = nullptr, int *bond_jnum_out = nullptr);
  bool full_system_force(int i, const double *xtrial, double *fout, double *fneigh = nullptr,
                         int jnum = 0, const int *jlist = nullptr);
  void partial_force_add_bonded(int i, const double *xtrial, double *fout,
                                bool live_neighbor_positions, tagint *bond_tag,
                                double *bond_fneigh, int *bond_jnum_out) const;
  void probe_atom(int i, const double *xi, bool with_neighbors, bool with_bonds, double (*H)[3],
                  int &jnum, int *&jlist, std::vector<double> &Hc,
                  std::vector<tagint> &bond_tag_buf, std::vector<double> &Hc_bond, int &n_bond);
  static constexpr int MOLPART = 4;      // max molecule partners per atom (collect_partners())
  void collect_partners(const double (*H)[3], int jnum, const int *jlist,
                        const std::vector<double> &Hc, int n_bond,
                        const std::vector<tagint> &bond_tag_buf,
                        const std::vector<double> &Hc_bond, double *Hown, double *X, tagint *T,
                        int &np) const;
  // neighbor clamping used by partial_force() with live_neighbor_positions = false
  virtual bool clamp_neighbor(int) const { return false; }
  virtual double **clamp_centers() const { return nullptr; }

  // ---- a second, standalone pair style (the teacher of baoab/tether, the student of
  // baoab/hrespa): never assigned to force->pair, its own neighbor list and per-atom caches ----
  // "<keyword> <n> <word1> ... <wordn>" (an explicit count, since a pair style's own arguments
  // vary in number): parse the n words at arg[iarg + 2 ...]; returns the next iarg
  int parse_counted_args(int narg, char **arg, int iarg, std::vector<std::string> &out);
  // construct it (pair_style name + settings, then pair_coeff); role is for the log message
  class Pair *create_aux_pair(const std::string &name, const std::vector<std::string> &settings,
                              const std::vector<std::string> &coeff, const char *role);
  // its whole-system force on the owned atoms at the current positions (ghosts must be current)
  // into out (3 per owned atom); atom->f is left unchanged
  void aux_pair_force(class Pair *p, std::vector<double> &out);

  // pair and kspace energies and virials (thermo output of the current step)
  static constexpr int NACCUM = 2 + 6 + 1 + 6;
  void save_accumulators(double *) const;
  void restore_accumulators(const double *);

 private:
  static int base_narg(class LAMMPS *, int, const char *);
};

}    // namespace LAMMPS_NS

#endif
