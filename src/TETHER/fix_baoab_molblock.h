/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(baoab/molblock,FixBAOABMolblock);
// clang-format on
#else

#ifndef LMP_FIX_BAOAB_MOLBLOCK_H
#define LMP_FIX_BAOAB_MOLBLOCK_H

#include "fix_baoab_tether_base.h"

#include <vector>

namespace LAMMPS_NS {

// Hessian-built auto-RESPA for small stiff molecules (water, ...): the bonds and angles of each
// molecule get a fast harmonic force field in internal coordinates, built from finite-difference
// curvature probes (probe_atom()) and integrated with inner substeps; everything else is the
// slow force. Formerly "fix baoab/tether ... molblock yes".

class FixBAOABMolblock : public FixBAOABTetherBase {
 public:
  FixBAOABMolblock(class LAMMPS *, int, char **);
  ~FixBAOABMolblock() override;

  void init() override;
  void initial_integrate(int) override;
  void final_integrate() override;
  double memory_usage() override;
  void grow_arrays(int) override;
  void copy_arrays(int, int, int) override;
  int pack_exchange(int, double *) override;
  int unpack_exchange(int, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;
  int pack_border(int, int *, double *) override;
  int unpack_border(int, int, double *) override;

 private:
  double theta;          // a molecule is stiff if a fast mode has omega*dt > theta
  int refresh_every;     // steps between force-constant refreshes
  int heff_samples;      // > 0: refresh by trajectory estimate (EMA time constant, refreshes)
  double gamma_soft;     // thermostat friction (default 1/damp)
  int gamma_soft_was_set;

  // probe data per atom (owned atoms measured, ghosts filled by forward comm at a refresh)
  std::vector<double> mb_Hown;    // nall x 9: raw symmetrized own block H_ii
  std::vector<double> mb_X;       // nall x MOLPART x 9: dF_partner/dx_i (row = partner force comp)
  std::vector<tagint> mb_tag;     // nall x MOLPART: partner tags
  std::vector<int> mb_n;          // nall: number of partners
  std::vector<tagint> mb_btop;    // nall x mb_bpa: bond partner tags (own bond list)
  std::vector<tagint> mb_atop;    // nall x 3 mb_apa: angle atom tags (own angle list)
  std::vector<int> mb_nbt, mb_nat;    // nall: number of bonds / angles listed

  // molblock (auto-RESPA, see mb_refresh()): Hessian-built fast force field for the bonds and
  // angles of small stiff molecules, integrated with inner substeps
  static constexpr int MBA = 5;           // max atoms per molecule (atom + 4 bond/angle partners)
  static constexpr int MBN = 3 * MBA;     // max Cartesian dimension of a molecule
  struct MBMol {
    int n = 0;                            // atoms in the molecule
    int nm = 0;                           // fast modes (= nic, or 0 if the molecule is not stiff)
    bool valid = false;                   // all atoms local this step
    tagint tag[MBA];
    double m[MBA];
    double xref[MBA][3];                  // geometry (rel. to COM) at refresh
    double w[MBN];                        // GF angular frequencies of the fast set (1/time)
    double Dref[MBN][MBN];                // mass-weighted Hessian at refresh
    int nic = 0;                          // fast internal coordinates (bonds, angles)
    // 0 = bond (a,b), 1 = angle (a,b,c) with vertex b, 2..7 = linear bend (a,b,c): component
    // (type-2)%2 of b's displacement from line a-c, frame from lab axis (type-2)/2 (mb_lin_frame)
    int ic_type[MBN], ic_at[MBN][3];
    double Fic[MBN][MBN];                 // internal-coordinate force constants
    double sig0[MBN];                     // reference internal coordinates (anchored once)
    bool have_s0 = false;
    double Fold[MBN][MBN];                // force constants still in use until the next step
    bool pending = false;                 // Fold valid, switch to Fic at the next step start
    // trajectory estimate of F: sums of ds ds^T and dphi ds^T over consecutive steps since the
    // last update (phi = raw internal-coordinate force), and the previous step's s and phi
    int nd = 0;
    bool hprev = false;
    double sprev[MBN], pprev[MBN], dss[MBN][MBN], dps[MBN][MBN];
    bool has_mom = false;                 // averaged moments (finite-difference F as prior)
    double Sss[MBN][MBN], Sps[MBN][MBN];
  };
  int mb_verbose;
  int mb_inner_user, mb_nin;              // user substeps (0 = auto), current substeps
  double mb_wmax;                         // fastest refreshed frequency (1/time)
  void mb_fast_force(const MBMol &, const double (*)[3], double (*)[3], bool = false) const;
  void mb_ic_eval(const MBMol &, const double (*)[3], double *, double (*)[MBN]) const;
  int mb_stiff_modes(const MBMol &, const double (*)[3]) const;
  void mb_gf_forceconst(MBMol &, const double (*)[3], const double (*)[3]);
  void mb_gf_freqs(MBMol &, const double (*)[3]);
  void mb_ic_force(const MBMol &, const double (*)[3], const int *, double *, double *) const;
  void mb_reestimate();
  void mb_lin_sym(MBMol &, double (*)[MBN]) const;
  void mb_lin_reframe(MBMol &, const double (*)[3]) const;
  bigint mb_last_refresh;
  bigint mb_step;                         // step whose positions/forces mb_prepare() processes
  // parallel layout: every rank integrates its own copy of each molecule with an owned atom
  // (ghosts for the rest), so the inner loop needs no communication. A molecule's persistent
  // state lives on its anchor (lowest-ID) atom, travels with it (exchange) and to ghost copies
  // (borders); between reneighborings each copy evolves it identically.
  std::vector<MBMol> mbmols;              // this rank's molecule copies (rebuilt each call)
  std::vector<int> mb_aidx;               // local or ghost index of each copy's anchor atom
  std::vector<int> mb_member;
  int mb_nicmax, mb_ss, mb_bpa, mb_apa, mb_commflag, mb_probe_size, mb_nmax;
  double **mb_state;                      // per atom: [0] = 1 if anchor holding state, then MBMol
  tagint *mb_anc;                         // per atom: anchor tag of its molecule (0 = none)
  bool mb_locate(const MBMol &, int *, double (*)[3]) const;
  void mb_rel(const MBMol &, const double (*)[3], double (*)[3]) const;
  int mb_pack(const MBMol &, double *) const;
  void mb_unpack(MBMol &, const double *) const;
  void mb_gather();
  void mb_scatter();
  double mb_gauss(tagint, int) const;
  void mb_refresh();
  void mb_prepare();
  void mb_initial_integrate();
  void mb_final_integrate();
};

}    // namespace LAMMPS_NS

#endif
#endif
