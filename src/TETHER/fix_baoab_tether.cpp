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
    demote_cool(nullptr), quench_exempt(nullptr), last_ke(nullptr), teacher_pair(nullptr),
    f_delta(nullptr)
{
  theta = -1.0;
  refresh_every = -1;
  promote_every = 0;
  settle_confirm = 0;
  settle_streak = 0;
  eps = 0.0;
  eps_was_set = 0;
  heff_samples = 0;

  mollify = 0;
  newton_iters = 8;
  newton_damp = 0.6;
  solve_tol_rel = 0.005;
  use_local_partial_force = 0;
  force_eval_full = 0;
  a_th_mass_scale = 0;
  eligibility_bonded = 2;
  om_mask_thr = 0.0;
  bonded = nullptr;
  newton_res_length = 0;
  newton_tol_rel = 5.0;
  flip_rel = 14.0;
  energy_guard_rel = 25.0;
  newton_cap_rel = 8.0;

  adapt = 0;
  c_acc = 0.25;
  dt_min = -1.0;    // sentinel: default 0.02*dt, resolved once dt is known below
  dt_max = -1.0;    // sentinel: default the fix-creation-time dt, resolved below
  ke_rel = 12.0;
  gamma_quench = 20.0;
  gamma_soft = 0.0;
  gamma_soft_was_set = 0;
  gamma_quench_auto = 0;
  gamma_quench_last_logged = -1.0;
  gamma_quench_calib = 0;
  gamma_quench_calib_warmup = 0;
  gamma_quench_calib_period = 0;
  gamma_quench_calib_gain = 0.5;
  gamma_quench_calib_max_step = 2.0;
  gamma_quench_calib_confirm = 3;
  gamma_quench_calib_deadband_lo = 0.8;
  gamma_quench_calib_deadband_hi = 1.25;
  gamma_quench_calib_ceiling = 1.0e4;
  gamma_quench_calib_refine = 1;
  gamma_quench_calib_sum_ke = 0.0;
  gamma_quench_calib_sum_dof = 0.0;
  gamma_quench_calib_direction = 0;
  gamma_quench_calib_streak = 0;
  gamma_quench_calib_committed = 0;
  last_gamma_calib_check = -1;
  ke_rel_warmup_steps = 0;
  ke_rel_refine = 0;
  ke_rel_margin_blocks = 1000;
  for (int i = 0; i < KE_REL_NBINS; i++) ke_rel_hist[i] = 0.0;
  ke_rel_u = -1.0;
  ke_rel_count_u = 0.0;
  ke_rel_sum_exceed = 0.0;
  ke_rel_n_total = 0.0;
  ke_rel_last_logged = -1.0;
  refresh_skin = 0.0;
  refresh_skin_rel = 0.0;
  drift_response = 0;
  exile_substep = 0;
  recoil = 1;
  molblock_diag = 0;
  molblock = 0;
  mb_verbose = 0;
  mb_inner_user = 0;
  mb_nin = 1;
  mb_wmax = 0.0;
  mb_last_refresh = -1;
  mb_step = 0;
  mb_nicmax = mb_ss = mb_bpa = mb_apa = mb_commflag = mb_probe_size = mb_nmax = 0;
  mb_state = nullptr;
  mb_anc = nullptr;
  kspace_center = 0;
  mb_collect = false;
  preshrink_exile = 0;
  om_split = 0.0;
  om_split_set = 0;
  dt_lvl = -1;
  heavy_groupbit = 0;
  heavy_sample = 8;
  heavy_block_tr3 = 0.0;
  heavy_block_set = 0;
  j_free = 0;
  newton_debug_tag = -1;
  pforce_debug_tag = -1;

  teacher_every = 0;
  outer_cycle_start = 0;
  n_teacher_calls = 0;
  t_teacher_total = t_teacher_pair = 0.0;
  t_run_start = t_teacher_total_run0 = t_teacher_pair_run0 = 0.0;
  n_teacher_calls_run0 = 0;
  teacher_open_dt = -1.0;
  teacher_open_half = teacher_cycle_time = 0.0;
  teacher_dt_varied = 0;
  n_teacher_dt_varied = n_teacher_dt_varied_run0 = 0;
  diag_om_unres = diag_om_unres_p99 = diag_om_analytic = 0.0;
  diag_unres_type = 0;

  int iarg = 7;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "theta") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether theta", error);
      theta = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "om_split") == 0) {
      // DIAGNOSTIC-ONLY override: bypasses compute_om_split()'s one-shot
      // group-scoped spectral-gap search entirely and latches the supplied
      // value directly (om_split_set=1 so the refresh() bootstrap never
      // overwrites it). Exists to test the hypothesis that compute_om_split()
      // restricted to this fix's own group cannot find the true stiff/soft
      // boundary when a companion fix handles the rest of the system -- see
      // the om_split group-scope investigation. Not a production keyword.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether om_split", error);
      om_split = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      om_split_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "heavy_group") == 0) {
      // DIAGNOSTIC-ONLY, deliberate match to lj_longstep_prototype.py's
      // HPdEngine.hessian_blocks(): atoms in the named group get a single,
      // frozen, isotropic curvature estimate (see compute_heavy_block())
      // instead of a real per-refresh FD-Hessian probe. A known non-general
      // shortcut, not validated reference physics -- kept only to get C++
      // and Python behaviorally consistent for now. Not a production
      // keyword; see python_heavy_block_frozen_approximation memory.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether heavy_group", error);
      int igroup = group->find(arg[iarg + 1]);
      if (igroup == -1)
        error->all(FLERR, "Fix baoab/tether heavy_group: group {} does not exist", arg[iarg + 1]);
      heavy_groupbit = group->bitmask[igroup];
      iarg += 2;
    } else if (strcmp(arg[iarg], "heavy_sample") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether heavy_sample", error);
      heavy_sample = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "j_free") == 0) {
      // DIAGNOSTIC-ONLY: matches lj_longstep_prototype.py's H/Pd engine
      // having J identically zero (see header comment by j_free). Not a
      // production keyword.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether j_free", error);
      j_free = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "newton_debug_tag") == 0) {
      // THROWAWAY DIAGNOSTIC: not a production keyword, see header comment.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether newton_debug_tag", error);
      newton_debug_tag = utils::tnumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "pforce_debug_tag") == 0) {
      // THROWAWAY DIAGNOSTIC: not a production keyword, see header comment.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether pforce_debug_tag", error);
      pforce_debug_tag = utils::tnumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "refresh") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether refresh", error);
      refresh_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "promote_every") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether promote_every", error);
      promote_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "settle_confirm") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether settle_confirm", error);
      settle_confirm = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
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
    } else if (strcmp(arg[iarg], "a_th_mass_scale") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "fix baoab/tether a_th_mass_scale", error);
      a_th_mass_scale = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "eligibility") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether eligibility", error);
      if (strcmp(arg[iarg + 1], "all_modes") == 0) eligibility_bonded = 0;
      else if (strcmp(arg[iarg + 1], "bonded") == 0) eligibility_bonded = 1;
      else if (strcmp(arg[iarg + 1], "per_mode") == 0) eligibility_bonded = 2;
      else error->all(FLERR, "Unknown fix baoab/tether eligibility style {}; expected all_modes, bonded or per_mode",
                      arg[iarg + 1]);
      iarg += 2;
    } else if (strcmp(arg[iarg], "newton_res_length") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "fix baoab/tether newton_res_length", error);
      newton_res_length = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "newton_tol_rel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether newton_tol_rel", error);
      newton_tol_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "flip_rel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether flip_rel", error);
      flip_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "energy_guard_rel") == 0) {
      if (iarg + 2 > narg)
        utils::missing_cmd_args(FLERR, "fix baoab/tether energy_guard_rel", error);
      energy_guard_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "newton_cap_rel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether newton_cap_rel", error);
      newton_cap_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
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
      if (strcmp(arg[iarg + 1], "warmup") == 0) {
        if (iarg + 3 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether ke_rel warmup", error);
        ke_rel_warmup_steps = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
        iarg += 3;
      } else {
        ke_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        ke_rel_warmup_steps = 0;
        iarg += 2;
      }
    } else if (strcmp(arg[iarg], "ke_rel_refresh") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether ke_rel_refresh", error);
      if (strcmp(arg[iarg + 1], "freeze") == 0)
        ke_rel_refine = 0;
      else if (strcmp(arg[iarg + 1], "refine") == 0)
        ke_rel_refine = 1;
      else
        error->all(FLERR, iarg + 1, "Fix baoab/tether ke_rel_refresh must be freeze or refine");
      iarg += 2;
    } else if (strcmp(arg[iarg], "ke_rel_margin_blocks") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether ke_rel_margin_blocks", error);
      ke_rel_margin_blocks = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench", error);
      if (strcmp(arg[iarg + 1], "auto") == 0) {
        gamma_quench_auto = 1;
        iarg += 2;
        if (iarg < narg && strcmp(arg[iarg], "calib") == 0) {
          if (iarg + 3 > narg)
            utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench auto calib", error);
          gamma_quench_calib = 1;
          gamma_quench_calib_warmup = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
          gamma_quench_calib_period = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
          iarg += 3;
        }
      } else {
        gamma_quench = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
        gamma_quench_auto = 0;
        iarg += 2;
      }
    } else if (strcmp(arg[iarg], "gamma_soft") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_soft", error);
      gamma_soft = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      gamma_soft_was_set = 1;
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench_calib_gain") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench_calib_gain", error);
      gamma_quench_calib_gain = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench_calib_max_step") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench_calib_max_step", error);
      gamma_quench_calib_max_step = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench_calib_confirm") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench_calib_confirm", error);
      gamma_quench_calib_confirm = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench_calib_deadband") == 0) {
      if (iarg + 3 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench_calib_deadband", error);
      gamma_quench_calib_deadband_lo = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      gamma_quench_calib_deadband_hi = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      iarg += 3;
    } else if (strcmp(arg[iarg], "gamma_quench_calib_ceiling") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench_calib_ceiling", error);
      gamma_quench_calib_ceiling = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "gamma_quench_calib_refresh") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether gamma_quench_calib_refresh", error);
      if (strcmp(arg[iarg + 1], "freeze") == 0)
        gamma_quench_calib_refine = 0;
      else if (strcmp(arg[iarg + 1], "refine") == 0)
        gamma_quench_calib_refine = 1;
      else
        error->all(FLERR, iarg + 1, "Fix baoab/tether gamma_quench_calib_refresh must be freeze or refine");
      iarg += 2;
    } else if (strcmp(arg[iarg], "refresh_skin") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether refresh_skin", error);
      refresh_skin = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "refresh_skin_rel") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether refresh_skin_rel", error);
      refresh_skin_rel = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
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
    } else if (strcmp(arg[iarg], "kspace_center") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether kspace_center", error);
      kspace_center = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "molblock") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether molblock", error);
      molblock = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "molblock_inner") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether molblock_inner", error);
      mb_inner_user = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (mb_inner_user < 0) error->all(FLERR, "fix baoab/tether molblock_inner must be >= 0");
      iarg += 2;
    } else if (strcmp(arg[iarg], "molblock_verbose") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether molblock_verbose", error);
      mb_verbose = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "molblock_diag") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether molblock_diag", error);
      molblock_diag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "recoil") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether recoil", error);
      recoil = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "exile_substep") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether exile_substep", error);
      exile_substep = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "preshrink_exile") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether preshrink_exile", error);
      preshrink_exile = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "force_eval") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether force_eval", error);
      if (strcmp(arg[iarg + 1], "local") == 0)
        force_eval_full = 0;
      else if (strcmp(arg[iarg + 1], "full") == 0)
        force_eval_full = 1;
      else
        error->all(FLERR, iarg + 1, "Fix baoab/tether force_eval must be local or full");
      iarg += 2;
    } else if (strcmp(arg[iarg], "event_log") == 0) {
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether event_log", error);
      event_log_filename = arg[iarg + 1];
      iarg += 2;
    } else if (strcmp(arg[iarg], "teacher_style") == 0) {
      // EXPERIMENTAL (D.14 RESPA multi-fidelity, see header comment by
      // teacher_pair): "teacher_style <n> <pair_style_name> <arg1> ... <argN>",
      // an explicit leading count since a pair style's own settings() args
      // vary in number (e.g. "symmetrix/mace no_mpi_message_passing" is 2
      // tokens). Stored now, actually constructed in init() once the
      // production pair style (needed to size/validate against) is final.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether teacher_style", error);
      int n = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (n < 1) error->all(FLERR, iarg + 1, "Fix baoab/tether teacher_style count must be >= 1");
      if (iarg + 2 + n > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether teacher_style", error);
      teacher_style_name = arg[iarg + 2];
      teacher_settings_args.clear();
      for (int k = 1; k < n; k++) teacher_settings_args.emplace_back(arg[iarg + 2 + k]);
      iarg += 2 + n;
    } else if (strcmp(arg[iarg], "teacher_coeff") == 0) {
      // EXPERIMENTAL: "teacher_coeff <n> <arg1> ... <argN>", passed to
      // teacher_pair->coeff() verbatim in init() (e.g.
      // "5 * * mace-mp-0b3-medium-1-46.json H Pd"). Same leading-count
      // convention as teacher_style, for the same reason.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether teacher_coeff", error);
      int n = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      if (n < 1) error->all(FLERR, iarg + 1, "Fix baoab/tether teacher_coeff count must be >= 1");
      if (iarg + 2 + n > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether teacher_coeff", error);
      teacher_coeff_args.clear();
      for (int k = 0; k < n; k++) teacher_coeff_args.emplace_back(arg[iarg + 2 + k]);
      iarg += 2 + n;
    } else if (strcmp(arg[iarg], "teacher_every") == 0) {
      // EXPERIMENTAL: the D.14 outer ratio "m". 0 (default) = disabled.
      if (iarg + 2 > narg) utils::missing_cmd_args(FLERR, "fix baoab/tether teacher_every", error);
      teacher_every = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else {
      error->all(FLERR, iarg, "Unknown fix baoab/tether keyword {}", arg[iarg]);
    }
  }

  if (theta <= 0.0) error->all(FLERR, "Fix baoab/tether theta must be > 0");
  if (refresh_every <= 0) error->all(FLERR, "Fix baoab/tether refresh must be > 0");
  if (promote_every < 0) error->all(FLERR, "Fix baoab/tether promote_every must be >= 0");
  if (settle_confirm < 0) error->all(FLERR, "Fix baoab/tether settle_confirm must be >= 0");
  if (eps_was_set && eps <= 0.0) error->all(FLERR, "Fix baoab/tether eps must be > 0");
  if (heff_samples < 0) error->all(FLERR, "Fix baoab/tether heff_samples must be >= 0");
  if (newton_iters < 1) error->all(FLERR, "Fix baoab/tether newton_iters must be >= 1");
  if (newton_damp <= 0.0 || newton_damp > 1.0)
    error->all(FLERR, "Fix baoab/tether newton_damp must be in (0,1]");
  if (solve_tol_rel <= 0.0) error->all(FLERR, "Fix baoab/tether solve_tol_rel must be > 0");
  if (refresh_skin < 0.0) error->all(FLERR, "Fix baoab/tether refresh_skin must be >= 0");
  if (refresh_skin_rel < 0.0) error->all(FLERR, "Fix baoab/tether refresh_skin_rel must be >= 0");
  if (refresh_skin > 0.0 && refresh_skin_rel > 0.0)
    error->all(FLERR, "Fix baoab/tether refresh_skin and refresh_skin_rel are mutually exclusive");
  if (molblock && !mollify)
    error->all(FLERR, "Fix baoab/tether molblock requires mollify yes");
  if (molblock && adapt)
    error->all(FLERR, "Fix baoab/tether molblock cannot be combined with adapt yes");
  if (molblock && (atom->molecular != Atom::MOLECULAR))
    error->all(FLERR, "Fix baoab/tether molblock requires a molecular atom style with bonds");
  if (exile_substep && !adapt)
    error->all(FLERR, "Fix baoab/tether exile_substep requires adapt yes");
  if (preshrink_exile && !exile_substep)
    error->all(FLERR, "Fix baoab/tether preshrink_exile requires exile_substep yes");
  if (gamma_quench_auto && !adapt)
    error->all(FLERR, "Fix baoab/tether gamma_quench auto requires adapt yes");
  if (gamma_quench_calib && !gamma_quench_auto)
    error->all(FLERR, "Fix baoab/tether gamma_quench auto calib requires gamma_quench auto");
  if (gamma_quench_calib_warmup < 0)
    error->all(FLERR, "Fix baoab/tether gamma_quench auto calib warmup steps must be >= 0");
  if (gamma_quench_calib && gamma_quench_calib_period <= 0)
    error->all(FLERR, "Fix baoab/tether gamma_quench auto calib period must be > 0");
  if (gamma_quench_calib_max_step <= 1.0)
    error->all(FLERR, "Fix baoab/tether gamma_quench_calib_max_step must be > 1.0");
  if (gamma_quench_calib_confirm < 1)
    error->all(FLERR, "Fix baoab/tether gamma_quench_calib_confirm must be >= 1");
  if (gamma_quench_calib_deadband_lo <= 0.0 || gamma_quench_calib_deadband_lo >= 1.0)
    error->all(FLERR, "Fix baoab/tether gamma_quench_calib_deadband lo must be in (0,1)");
  if (gamma_quench_calib_deadband_hi <= 1.0)
    error->all(FLERR, "Fix baoab/tether gamma_quench_calib_deadband hi must be > 1.0");
  if (gamma_quench_calib_ceiling <= 0.0)
    error->all(FLERR, "Fix baoab/tether gamma_quench_calib_ceiling must be > 0");
  if (ke_rel_warmup_steps < 0)
    error->all(FLERR, "Fix baoab/tether ke_rel warmup steps must be >= 0");
  if (ke_rel_warmup_steps > 0 && !adapt)
    error->all(FLERR, "Fix baoab/tether ke_rel warmup requires adapt yes");
  if (ke_rel_margin_blocks < 1)
    error->all(FLERR, "Fix baoab/tether ke_rel_margin_blocks must be >= 1");
  if (force_eval_full && comm->nprocs > 1)
    error->all(FLERR, "Fix baoab/tether force_eval full is a single-process validation mode "
                       "and cannot run under MPI domain decomposition");

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
  last_promote_check = -1;
  last_gamma_calib_check = gamma_quench_calib_warmup;
  need_refresh = 1;
  need_reneighbor = 0;
  dt_floor_until = -1;
  n_guard_newton = n_guard_flip = n_guard_energy = 0;
  n_demoted = n_kinetic_guard = 0;
  n_partial_force_calls = n_recompute_calls = 0;
  n_ghost_miss = 0;
  n_mask_agree = n_mask_stale_soft = n_mask_stale_tight = 0;

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
  // Also carries om[] (3 more doubles) so the Jtag/Jblk-building loop in
  // refresh_one_atom() can test a possibly-ghost neighbor's CURRENT
  // stiff/rattler eligibility (om-based) instead of its group membership --
  // see that loop's comment for why group membership stopped being a valid
  // proxy once this fix started operating on a single whole-system group.
  comm_forward = 7;

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
  maxexchange = 3 + 3 + 9 + 3 + 9 + 1 + 1 + JMAX * (1 + 9) + 4 + 1 + 1;

  // molblock per-atom molecule state, sized for the largest set of bonds and angles any one atom
  // takes part in, counting each angle twice (a near-linear angle becomes two bend coordinates; a
  // molecule with more is rejected at its first refresh)
  if (molblock) {
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
  // dropped -- see n_ghost_miss in the header); [8..10] stage-20 read-only
  // diagnostic (cached-vs-fresh mode-eligibility disagreement) -- see
  // n_mask_agree/n_mask_stale_soft/n_mask_stale_tight in the header.
  vector_flag = 1;
  size_vector = 18;
  extvector = 0;

  // per-atom persistent state (curvature-block eigenvectors/frequencies,
  // frozen/mollified center, sparse response blocks) -- exchanged/reordered
  // like any other atom-based array, but not restart-aware yet: a restart
  // re-refreshes on the first post-restart step rather than resuming
  // mid-cycle (stage-1/2 limitation).
  FixBAOABTether::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  if (molblock) atom->add_callback(Atom::BORDER);
}

/* ---------------------------------------------------------------------- */

FixBAOABTether::~FixBAOABTether()
{
  if (copymode) return;

  atom->delete_callback(id, Atom::GROW);
  if (molblock) atom->delete_callback(id, Atom::BORDER);
  memory->destroy(mb_state);
  memory->destroy(mb_anc);

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
  memory->destroy(bonded);
  memory->destroy(last_demote);
  memory->destroy(demote_cool);
  memory->destroy(quench_exempt);
  memory->destroy(last_ke);
  memory->destroy(f_delta);

  delete teacher_pair;

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
  // Verlet::run() increments update->ntimestep at the TOP of its loop,
  // before initial_integrate() (hence this fix's refresh_blocks()/
  // solve_center()/etc.) ever runs -- so every event logged from inside
  // initial_integrate() is evaluated on the state left over from the
  // *previous* completed step, not the step ntimestep is about to take.
  // Logging ntimestep-1 reports the step actually reflected by the data
  // (and makes the very first, pre-run bootstrap REFRESH read step=0,
  // matching the convention every non-LAMMPS reference implementation
  // uses for its pre-loop state) instead of a LAMMPS-internal counter
  // value that is one step ahead of the physical state being described.
  fputs(fmt::format("{} step={} tag={} ke={:.6g} ke_cap={:.6g} drift={:.6g} skin={:.6g} "
                     "om0={:.6g} om1={:.6g} om2={:.6g} dt={:.6g} note={}\n",
                     event, update->ntimestep - 1, tag, ke, ke_cap, drift, skin,
                     om ? om[0] : -1.0, om ? om[1] : -1.0, om ? om[2] : -1.0, dt, note)
            .c_str(),
        event_log_fp);
  fflush(event_log_fp);
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::init()
{
  FixBAOAB::init();

  // Thermostat sanity (units-independent): the friction must act on a timescale longer
  // than one integrator step, i.e. gamma*dt <= 1, otherwise velocities are re-randomized
  // every step and the dynamics are thermostat noise, not physics. damp (and gamma_soft)
  // are in the run's own time units, so a value copied between unit systems (e.g. 0.1 in
  // metal-units ps used in real-units fs) is off by orders of magnitude and silently
  // produces a stable-looking but meaningless trajectory -- hence an error, not a warning.
  // gamma_soft, when not given, defaults to gamma = 1/damp: one physical relaxation time
  // governs the whole fix (an explicit gamma_soft 0 still disables the baseline soft-mode
  // thermostat).
  if (!gamma_soft_was_set) gamma_soft = gamma;
  {
    double dt_check = adapt ? dt_max : update->dt;
    if (gamma * dt_check > 1.0)
      error->all(FLERR, "Fix baoab/tether damp {} is shorter than one timestep ({}): the "
                        "thermostat would re-randomize velocities every step. damp is in the "
                        "run's time units; a typical choice is about 100 times the largest "
                        "timestep, here {}", 1.0 / gamma, dt_check, 100.0 * dt_check);
    if (gamma_soft * dt_check > 1.0)
      error->all(FLERR, "Fix baoab/tether gamma_soft {} (1/time) exceeds 1 per timestep ({}): "
                        "the soft-mode thermostat would re-randomize velocities every step. "
                        "gamma_soft is in the run's inverse time units; a typical choice is "
                        "1/(100 times the largest timestep), here {}",
                 gamma_soft, dt_check, 1.0 / (100.0 * dt_check));
  }

  // eps defaults to a small fraction of the neighbor skin (a system-
  // dependent length scale already known at this point) rather than a
  // hardcoded absolute distance; override with the "eps" keyword when the
  // system's own thermal amplitude calls for something tighter or looser.
  if (!eps_was_set) eps = 1.0e-4 * neighbor->skin;
  if (eps <= 0.0)
    error->all(FLERR, "Fix baoab/tether eps must be > 0 (neighbor->skin is 0 -- set eps explicitly)");

  if (mollify) {
    if (!force->pair) error->all(FLERR, "Fix baoab/tether mollify yes requires a pair style");
    // the per-atom force kernels (local_partial_force()/single()) read the
    // host-side per-atom state of the pair style (e.g. PairEAM's rho[]/fp[]),
    // which KOKKOS pair styles keep on the device instead; only
    // symmetrix/mace/kk implements its own device-aware kernel. Under -sf kk,
    // the student pair style must therefore be defined after "suffix off";
    // the teacher_style may be a KOKKOS style (see compute_force_delta()).
    if (force->pair->kokkosable && !utils::strmatch(force->pair_style, "^symmetrix/mace"))
      error->all(FLERR,
                 "Fix baoab/tether cannot use the KOKKOS pair style {} for its per-atom forces; "
                 "define the pair style after a 'suffix off' command (a KOKKOS teacher_style "
                 "is supported)",
                 force->pair_style);
    use_local_partial_force = force->pair->has_local_partial_force;
    if (!use_local_partial_force && !force->pair->single_enable)
      error->all(FLERR, "Fix baoab/tether mollify yes requires a pair style that supports "
                         "either single() or local_partial_force()");
    if (atom->map_style == Atom::MAP_NONE)
      error->all(FLERR, "Fix baoab/tether mollify yes requires atom_modify map (yes|array)");

    // Bonded local-force support (partial_force_add_bonded()/
    // refresh_one_atom()) does its own per-atom atom->bond_atom[]/
    // atom->angle_atom1/2/3[] lookup rather than LAMMPS's own
    // neighbor->bondlist/anglelist (those are deduplicated, non-atom-
    // indexed flat lists built assuming newton_bond's one-sided storage
    // convention -- wrong tool for "given atom i, find its bonded
    // partners now"). That per-atom lookup only works if every bond/angle
    // is stored symmetrically on ALL its member atoms, which requires
    // newton_bond off (LAMMPS's default, newton_bond on, stores each one
    // on only whichever atom came first in the data file's Bonds/Angles
    // section -- see water_bonded_kspace_scoping memory). No such
    // requirement exists for kspace: it is never evaluated by
    // partial_force() at all, only by the whole-system
    // recompute_forces_local() path, which doesn't need per-atom topology.
    if ((eligibility_bonded == 1) && !(adapt && atom->molecular == Atom::MOLECULAR && force->bond))
      error->all(FLERR, "Fix baoab/tether eligibility bonded requires adapt yes, a molecular "
                         "atom style, and a bond style (it assumes a fixed, non-reactive bond topology)");
    if (force->bond && force->newton_bond)
      error->all(FLERR, "Fix baoab/tether mollify yes with a bond style requires "
                         "newton_bond off (needed for symmetric per-atom bond storage)");
    if (force->angle && force->newton_bond)
      error->all(FLERR, "Fix baoab/tether mollify yes with an angle style requires "
                         "newton_bond off (needed for symmetric per-atom angle storage)");

    // private, non-occasional full neighbor list: the damped-Newton inner
    // loop (solve_center()/partial_force()) needs every one of a flagged
    // atom's own pair-style neighbors, every step -- not the half-list
    // "i<j" restriction, and not just occasionally (plan doc SS5.1).
    // molblock reads the list only in mb_refresh()'s curvature probes, so there it is
    // occasional and built on demand (rebuilding it with every pair list cost about as
    // much as the pair list itself)
    if (molblock)
      neighbor->add_request(this, NeighConst::REQ_FULL | NeighConst::REQ_OCCASIONAL);
    else
      neighbor->add_request(this, NeighConst::REQ_FULL);
  }

  // D.14 teacher/student composition (EXPERIMENTAL, see header comment by
  // teacher_pair): construct the standalone teacher Pair object ONCE (not
  // re-constructed/re-loaded on every "run" command), independent of
  // force->pair -- mirrors Input::pair_style()'s own
  // create_pair()+settings() sequence (see force.cpp), except the result is
  // never assigned to force->pair. teacher_pair->init() itself IS re-run
  // every init() call below, same as force->pair's own re-init on every run
  // and this fix's own neighbor->add_request() above -- it registers
  // teacher_pair's own neighbor-list request for this run.
  if (teacher_every > 0) {
    if (!mollify) error->all(FLERR, "Fix baoab/tether teacher_every requires mollify yes");
    if (teacher_style_name.empty())
      error->all(FLERR, "Fix baoab/tether teacher_every requires teacher_style");
    if (teacher_coeff_args.empty())
      error->all(FLERR, "Fix baoab/tether teacher_every requires teacher_coeff");

    // the teacher takes the active suffix like any pair_style command, so
    // "-sf kk" (or an explicit symmetrix/mace/kk) runs it on the GPU while
    // the student and this fix stay on the host
    if (!teacher_pair) {
      int sflag;
      teacher_pair = force->new_pair(teacher_style_name, lmp->suffix_enable, sflag);

      std::vector<char *> settings_argv;
      for (auto &s : teacher_settings_args) settings_argv.push_back(const_cast<char *>(s.c_str()));
      teacher_pair->settings((int) settings_argv.size(), settings_argv.data());

      std::vector<char *> coeff_argv;
      for (auto &s : teacher_coeff_args) coeff_argv.push_back(const_cast<char *>(s.c_str()));
      teacher_pair->coeff((int) coeff_argv.size(), coeff_argv.data());

      if (comm->me == 0) {
        std::string tname = teacher_style_name;
        if (sflag) tname += fmt::format("/{}", (sflag == 1) ? lmp->suffix : lmp->suffix2);
        utils::logmesg(lmp, "Fix baoab/tether teacher pair style: {} ({})\n", tname,
                       teacher_pair->kokkosable ? "KOKKOS" : "not KOKKOS, runs on the host");
      }
    }
    teacher_pair->init();
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

  if (molblock) {
    mb_initial_integrate();
    return;
  }

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
    if (kspace_center) kspace_snapshot();
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      double mi = rmass ? rmass[i] : mass[type[i]];
      solve_center(i, mi, kT, c[i]);
    }
    force_moll();

    // D.14 teacher/student composition (EXPERIMENTAL, teacher_every > 0
    // only): this step OPENS a new outer cycle -- apply the carried
    // f_delta impulse (computed at the close of the PREVIOUS cycle, or
    // left all-zero before the first one) before the ordinary B-kick below
    // consumes atom->f. See teacher_outer_kick()'s doc comment.
    // The cycle's duration is not known yet: assume teacher_every*dt and
    // record that, so final_integrate() can correct it at the close.
    if (teacher_every > 0 &&
        (update->ntimestep - outer_cycle_start) % teacher_every == 0) {
      teacher_open_dt = update->dt;
      teacher_open_half = 0.5 * teacher_every * update->dt;
      teacher_cycle_time = 0.0;
      teacher_dt_varied = 0;
      teacher_outer_kick(teacher_open_half);
    }
  }

  double one_minus_c1sq = 1.0 - c1 * c1;

  // strong-friction quench thermostat coefficients (adapt=yes only; harmless
  // to precompute unconditionally since kinetic_guard_and_quench() is only
  // ever called, and quench_active only ever comes back true, when adapt is
  // set -- see there). Full-step gamma_quench, same "O applies once, at full
  // dt" convention as c1/one_minus_c1sq above.
  double cq1 = exp(-gamma_quench * update->dt);
  double one_minus_cq1sq = 1.0 - cq1 * cq1;

  // Baseline thermostat coefficients for unmasked, non-quench modes (see
  // gamma_soft's header comment). Harmless to precompute unconditionally:
  // gamma_soft == 0.0 (default) gives c1_soft == 1.0, one_minus_c1sq_soft
  // == 0.0, so the O-step branch below reduces to the prior "p2[m] = p1[m]"
  // no-op exactly, byte-for-byte, when the keyword is not given.
  double c1_soft = exp(-gamma_soft * update->dt);
  double one_minus_c1sq_soft = 1.0 - c1_soft * c1_soft;

  energy_onestep = 0.0;

  double fran_total[3] = {0.0, 0.0, 0.0};
  double mass_total = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double invmass = 1.0 / mi;

    // exile_substep (adapt=yes only): while this atom is Verlet-exiled
    // (demote_cool[i] > ntimestep), skip the shared B-A-O-A pipeline below
    // entirely and instead integrate it for the WHOLE step via
    // sub_step_free_atom()'s already-safe, Allreduce-free, locally
    // resonance-bounded local sub-stepping -- same mechanism
    // kinetic_guard_and_quench() already uses for the remaining dtby2 of a
    // just-triggered step (see "sub_stepped" below), just spanning this
    // atom's full existing exile window instead of one rescue step. Mirrors
    // that call site's own early "continue" pattern. final_integrate() and
    // update_dt()'s om_unres loop / the PRESHRINK drift-skin loop carry the
    // matching exclusion -- see there.
    if (exile_substep && demote_cool[i] > update->ntimestep) {
      // Quench during exile (same condition the shared pipeline uses for quench_active):
      // sub_step_free_atom() is plain velocity-Verlet with no friction and no noise, so without
      // this a hot exiled atom (e.g. one released into a repulsive contact, ~3 eV) keeps that
      // energy for its whole exile window and re-enters the shared pipeline still hot, then
      // crosses >2 A in one full-dt step (found on H/Pd at energy_guard_rel 25 and on water).
      // The pipeline's own cooling (strong friction gamma_quench at the target temperature) is
      // applied here as a symmetric O(dt/2) - Verlet sub-steps - O(dt/2) split, with the same
      // energy bookkeeping.
      bool exile_quench = (quench_exempt[i] <= update->ntimestep) &&
                          (last_demote[i] != update->ntimestep);
      double ch = exp(-gamma_quench * 0.5 * update->dt);
      double noise_h = sqrt(kT * invmass * (1.0 - ch * ch));
      double ke_pre = 0.0;
      if (exile_quench) {
        ke_pre = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
        for (int k = 0; k < 3; k++) v[i][k] = ch * v[i][k] + noise_h * random->gaussian();
      }
      double x_mid[3] = {x[i][0], x[i][1], x[i][2]};
      double v_mid[3] = {v[i][0], v[i][1], v[i][2]};
      sub_step_free_atom(i, mi, x_mid, v_mid, update->dt, kT);
      if (exile_quench) {
        for (int k = 0; k < 3; k++) v[i][k] = ch * v[i][k] + noise_h * random->gaussian();
        double ke_post = v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2];
        energy_onestep += 0.5 * force->mvv2e * mi * (ke_pre - ke_post);
      }
      continue;
    }

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

    // stage-20 read-only diagnostic (long-run light-subsystem heating
    // investigation): measure how often this cached mask disagrees with a
    // fresh re-probe, BEFORE kinetic_guard_and_quench() below can mutate
    // msk[] in place. No effect on dynamics -- see diag_check_staleness().
    if (mollify) diag_check_staleness(i, mi, msk);

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
        // om[i][m] is in natural tau^-1 units (see mode_is_stiff()'s doc
        // comment in the header); dtby2 is real LAMMPS ps. Convert w before
        // this literal analytic harmonic rotation, or the oscillator phase
        // advances ~98x too slowly every step.
        double w = om[i][m] * sqrt(force->ftm2v);
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
    double c2_soft = sqrt(kT * invmass * one_minus_c1sq_soft);
    double r[3] = {random->gaussian(), random->gaussian(), random->gaussian()};
    double p2[3];
    double ke_before = 0.0, ke_after = 0.0;
    for (int m = 0; m < 3; m++) ke_before += p1[m] * p1[m];
    for (int m = 0; m < 3; m++) {
      if (msk[m]) p2[m] = c1 * p1[m] + c2 * r[m];
      else if (quench_active) p2[m] = cq1 * p1[m] + cq2 * r[m];
      else p2[m] = c1_soft * p1[m] + c2_soft * r[m];
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
        // same natural-tau -> real-ps conversion as the first-half drift above.
        double w = om[i][m] * sqrt(force->ftm2v);
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

  // x[]/v[] above were written through atom->x/atom->v raw pointers, which
  // VerletKokkos's initial_integrate() dispatch does not bracket with its
  // own sync/modified (unlike its per-style pair/bond/... compute() calls,
  // see src/KOKKOS/verlet_kokkos.cpp) -- so a subsequent Kokkos pair style's
  // own bracketed sync could otherwise see a stale "device already current"
  // flag. Same idiom as the SPECIAL_MASK hook in special.cpp.
  if (lmp->kokkos) dynamic_cast<AtomKokkos *>(atom)->modified(Host, X_MASK | V_MASK);

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

  if (molblock) {
    mb_final_integrate();
    return;
  }

  if (mollify) {
    double kT = force->boltz * t_target / force->mvv2e;
    // stage-4: see the matching comm->forward_comm(this) note in
    // initial_integrate() -- same reasoning applies here.
    comm->forward_comm(this);
    if (kspace_center) kspace_snapshot();
    for (int i = 0; i < nlocal; i++) {
      if (!(mask[i] & groupbit)) continue;
      double mi = rmass ? rmass[i] : mass[type[i]];
      solve_center(i, mi, kT, c[i]);
    }
    force_moll();

    // D.14 teacher/student composition (EXPERIMENTAL, teacher_every > 0
    // only): this step CLOSES the current outer cycle -- recompute f_delta
    // fresh at the NEW clamped geometry (one teacher call), apply it as
    // this cycle's closing half-kick, then advance outer_cycle_start so the
    // SAME freshly-computed f_delta becomes the next cycle's opening
    // half-kick in the following initial_integrate() call (the leapfrog
    // reuse the Python reference's run_mf() relies on).
    //
    // The two half-kicks must together deliver f_delta over the cycle's
    // ACTUAL duration (sum of its step sizes). The opening half-kick could
    // only assume teacher_every*dt at the time it was applied; if adapt
    // changed dt during the cycle, that assumption was wrong (by up to the
    // whole dt ratio, e.g. 4x after two demotions), and leaving it
    // uncorrected pumps energy into every cycle with a dt change -- which is
    // most cycles in a large system. So: top up (or take back) the opening
    // half-kick with the OLD f_delta, then apply the closing half-kick over
    // the actual duration. A cycle with constant dt keeps the original
    // teacher_every*dt formula exactly.
    if (teacher_every > 0) {
      teacher_cycle_time += update->dt;
      if (update->dt != teacher_open_dt) teacher_dt_varied = 1;
    }
    if (teacher_every > 0 &&
        (update->ntimestep - outer_cycle_start + 1) % teacher_every == 0) {
      double half_time = 0.5 * teacher_every * update->dt;
      if (teacher_dt_varied && (teacher_open_dt > 0.0)) {
        half_time = 0.5 * teacher_cycle_time;
        teacher_outer_kick(half_time - teacher_open_half);
        n_teacher_dt_varied++;
      }
      compute_force_delta();
      teacher_outer_kick(half_time);
      outer_cycle_start += teacher_every;
    }
  }

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;

    // exile_substep: this atom's initial_integrate() already took the
    // sub_step_free_atom() branch above and folded its own terminal local
    // half-kick (from a local partial_force() evaluation) into v[i] -- a
    // second closing kick here from the freshly-computed GLOBAL force would
    // double-apply it every step of the exile window (not just once, unlike
    // today's one-step kinetic-guard usage). demote_cool[i] is unchanged
    // since initial_integrate() ran this same step, so re-testing it here
    // is safe and needs no new per-atom flag.
    if (exile_substep && demote_cool[i] > update->ntimestep) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double dtfm = dtf / mi;

    double fcorr[3] = {0.0, 0.0, 0.0};
    if (!mollify) modal_force_correction(i, mi, fcorr);
    v[i][0] += dtfm * (f[i][0] + fcorr[0]);
    v[i][1] += dtfm * (f[i][1] + fcorr[1]);
    v[i][2] += dtfm * (f[i][2] + fcorr[2]);
  }

  // see the matching comment in initial_integrate()
  if (lmp->kokkos) dynamic_cast<AtomKokkos *>(atom)->modified(Host, V_MASK);
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
  bigint ntimestep = update->ntimestep;

  // promote_every (default 0 = disabled): re-run update_dt() more often than
  // refresh_blocks()'s refresh_every cadence, reusing the currently cached
  // om[] -- no new curvature probing, no change to update_dt()'s own
  // one-rung/1.25x-margin/dt_floor_until logic. Independent of, and placed
  // before, the drift_response==preshrink early return below: this only
  // closes the gap between a dt_floor_until lockout expiring and the next
  // scheduled refresh, which otherwise caps post-trigger recovery at 1-2
  // ladder rungs per cycle (see the plan file).
  if (adapt && promote_every > 0 && dt_lvl > 0 &&
      (ntimestep - last_promote_check) >= promote_every) {
    update_dt(false);    // reuses cached om[] -- must never count toward settle_confirm
    last_promote_check = ntimestep;
  }

  if (!adapt || (refresh_skin <= 0.0 && refresh_skin_rel <= 0.0) || drift_response != 1) return;

  if (ntimestep == last_refresh_step) return;    // x0[i] was just reset

  double **x = atom->x;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  if (igroup == atom->firstgroup) nlocal = atom->nfirst;

  double kT = force->boltz * t_target / force->mvv2e;

  double dt_local = dt_max;
  int i_trigger = -1;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    // exile_substep: already safely self-integrating at its own locally
    // resonance-bounded dt_sub (see initial_integrate()) -- it should not
    // also be allowed to float-trigger a global PRESHRINK shrink.
    if (exile_substep && demote_cool[i] > ntimestep) continue;

    double mi = rmass ? rmass[i] : mass[type[i]];
    double skin_i = effective_skin(i, mi, kT);

    double dx0[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};
    domain->minimum_image(FLERR, dx0);
    double drift2 = dx0[0] * dx0[0] + dx0[1] * dx0[1] + dx0[2] * dx0[2];
    if (skin_i <= 0.0 || drift2 <= skin_i * skin_i) continue;

    refresh_one_atom(i, x[i], true);

    // preshrink_exile (requires exile_substep yes): this atom's own drift
    // just tripped the skin threshold -- instead of letting it pull the
    // shared GLOBAL dt down for every other atom, demote it on the spot
    // (same flat 3*refresh_every backoff the kinetic (ke_hot) guard uses)
    // so the NEXT step's initial_integrate() routes it through
    // sub_step_free_atom()'s already-safe, Allreduce-free local sub-
    // stepping instead, and this step's dt_local/dt_global vote never
    // sees it. demote_atom() logs its own DEMOTE event (source=preshrink,
    // distinct from source=kinetic_guard) -- no separate PRESHRINK log
    // line for an atom handled this way.
    if (preshrink_exile) {
      demote_atom(i, 3 * (bigint) refresh_every, "preshrink");
      continue;
    }

    // om[] is in natural tau^-1 units, not real ps^-1 -- convert before using it
    // to size a real dt (see mode_is_stiff()'s doc comment in the header).
    double om_max_new = MAX(om[i][0], MAX(om[i][1], om[i][2])) * sqrt(force->ftm2v);
    double dt_target = (om_max_new > 0.0) ? MIN(dt_min, 2.2 / om_max_new) : dt_min;
    if (dt_target < dt_local) {
      dt_local = dt_target;
      i_trigger = i;
    }
  }

  double dt_global = dt_local;
  if (comm->nprocs > 1) MPI_Allreduce(&dt_local, &dt_global, 1, MPI_DOUBLE, MPI_MIN, world);

  if (dt_global >= dt_max) return;    // no candidate on any rank this step

  // settle_confirm (default 0 = disabled; see update_dt()): an out-of-
  // cadence drift-skin re-probe reaching this point at all is itself fresh
  // evidence this atom has not yet settled, whether or not dt_lvl actually
  // changes below (e.g. already pinned at the dt_min rung from a prior
  // trip) -- reset unconditionally, not only on an actual new shrink, so a
  // still-unresolved atom's own repeated re-trips can never be missed by a
  // scheduled update_dt() call that happens to land on a calm moment in
  // between. dt_global is already identical on every rank (Allreduce MIN
  // just above), so this reset is too.
  if (settle_confirm > 0) settle_streak = 0;

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

  bool skin_enabled = (refresh_skin > 0.0 || refresh_skin_rel > 0.0);
  double skin_i = skin_enabled ? effective_skin(i, mi, kT) : 0.0;

  double ke_max = 0.0;
  for (int m = 0; m < 3; m++) {
    if (!msk[m] && !skin_enabled) continue;
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

  // gamma_quench auto calib: rolling equipartition-temperature measurement
  // of the masked subsystem. Reuses the same per-mode ke term the ke_max
  // loop above just computed, but sums over EVERY masked mode-instance
  // (not just the max), and excludes ke_hot steps -- an outlier being
  // pulled out of the dynamics by the guard above, not representative
  // steady-state behavior. See update_dt()'s GAMMA_CALIB block for how
  // this rolling sum becomes a temperature and feeds the control law.
  if (gamma_quench_calib && !ke_hot) {
    for (int m = 0; m < 3; m++) {
      if (!msk[m]) continue;
      gamma_quench_calib_sum_ke += 0.5 * force->mvv2e * mi * p1[m] * p1[m];
      gamma_quench_calib_sum_dof += 1.0;
    }
  }

  // ke_rel warm-up calibration (Phase A histogram / Phase B exceedance
  // accumulation): reuses ke_max exactly as computed above -- the same
  // quantity ke_hot just compared against ke_cap -- so the fitted
  // threshold describes precisely the guard this fix enforces, not a
  // separate offline proxy. Phase A (ntimestep <= ke_rel_warmup_steps)
  // bins r = ke_max/kT into a fixed histogram; Phase B (once update_dt()
  // has closed Phase A and set ke_rel_u >= 0) accumulates the running
  // exceedance-over-threshold scalars the exponential-tail fit needs. See
  // update_dt() for the fit itself and the freeze/refine policies.
  if (ke_rel_warmup_steps > 0) {
    double r = ke_max / (kT * force->mvv2e);
    if (ntimestep <= ke_rel_warmup_steps) {
      int bin = (int) (r / KE_REL_HIST_MAX * KE_REL_NBINS);
      bin = MAX(0, MIN(KE_REL_NBINS - 1, bin));
      ke_rel_hist[bin] += 1.0;
    } else if (ke_rel_u >= 0.0) {
      ke_rel_n_total += 1.0;
      if (r > ke_rel_u) {
        ke_rel_count_u += 1.0;
        ke_rel_sum_exceed += (r - ke_rel_u);
      }
    }
  }

  bool drift_hot = false;
  double drift_mag = -1.0;
  if (!ke_hot && skin_i > 0.0) {
    double **x = atom->x;
    double dx0[3] = {x[i][0] - x0[i][0], x[i][1] - x0[i][1], x[i][2] - x0[i][2]};
    domain->minimum_image(FLERR, dx0);
    double drift2 = dx0[0] * dx0[0] + dx0[1] * dx0[1] + dx0[2] * dx0[2];
    if (drift2 > skin_i * skin_i) {
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
    log_event("GUARD_DRIFT", atom->tag[i], -1.0, -1.0, drift_mag, skin_i, nullptr, -1.0,
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
  // om[] is in natural tau^-1 units, not real ps^-1 -- convert before using it
  // to size a real dt (see mode_is_stiff()'s doc comment in the header).
  double om_max_new = MAX(om[i][0], MAX(om[i][1], om[i][2])) * sqrt(force->ftm2v);
  // Plain velocity Verlet (the integrator used below) is unstable for omega*dt > 2, so the old
  // cap of 2.2/omega could pick a single UNSTABLE sub-step whenever dt_min*omega landed in
  // (2, 2.2] (found on water with dt_min pinned at 3 fs: stretch omega*dt = 2.06 -> blowup).
  // 1.0/omega keeps the sub-step stable and ~accurate (<0.5% frequency error). It only binds
  // when dt_min*omega > 1, which never happens for H/Pd (dt_min*omega ~ 0.02).
  double dt_sub = (om_max_new > 0.0) ? MIN(dt_min, 1.0 / om_max_new) : dt_min;
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
   molblock_diag (DIAGNOSTIC ONLY, changes no dynamics): for every molecule
   (an atom plus its bond/angle partners, as recorded by refresh_one_atom()),
   assemble the full mass-weighted Hessian from the own blocks H_ii and the
   cross columns dF_j/dx_i that the curvature refresh already measured,
   diagonalize it, split eigenvectors into internal vs rigid-body by their
   overlap with the 6 rigid translation/rotation vectors at the current
   geometry, and log mean +- sd frequencies over all molecules. Each
   molecule is handled once, by its lowest-tag atom, and only if all of its
   atoms are owned by this rank.
------------------------------------------------------------------------- */

void FixBAOABTether::molblock_diagnostic()
{
  const int MBMAX = 4;
  int nlocal = atom->nlocal;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  double **x = atom->x;
  double conv = sqrt(force->ftm2v);
  // angular frequency per run time unit -> wavenumber (cm^-1); 0 = not reported
  double cm1 = 1.0;
  if (strcmp(update->unit_style, "real") == 0)
    cm1 = 1.0 / (2.0 * MathConst::MY_PI * 2.99792458e-5);
  else if (strcmp(update->unit_style, "metal") == 0)
    cm1 = 1.0 / (2.0 * MathConst::MY_PI * 2.99792458e-2);

  int nmol = 0, nskip = 0, nsz = -1;
  std::vector<double> s_int, s2_int, s_rig;
  double purity_sum = 0.0, worst_purity = 1.0, asym_sum = 0.0;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    int np = mb_n[i];
    if (np <= 0) continue;
    tagint tags[MBMAX + 1];
    tags[0] = atom->tag[i];
    bool owner = true;
    for (int p = 0; p < np; p++) {
      tags[p + 1] = mb_tag[(size_t) i * MBMAX + p];
      if (tags[p + 1] < tags[0]) owner = false;
    }
    if (!owner) continue;
    int n = np + 1, N = 3 * n;
    int loc[MBMAX + 1];
    bool ok = true;
    for (int a = 0; a < n; a++) {
      loc[a] = (a == 0) ? i : atom->map(tags[a]);
      if ((loc[a] < 0) || (loc[a] >= nlocal) || (mb_n[loc[a]] != np)) ok = false;
    }
    if (!ok) {
      nskip++;
      continue;
    }
    if (nsz < 0) {
      nsz = n;
      s_int.assign(N, 0.0);
      s2_int.assign(N, 0.0);
      s_rig.assign(N, 0.0);
    }
    if (n != nsz) {
      nskip++;
      continue;
    }

    // raw Hessian H[(a,r),(b,s)] = -dF_{a,r}/dx_{b,s}; column b from atom b's own probes
    std::vector<double> Hm((size_t) N * N, 0.0);
    for (int b = 0; b < n; b++) {
      int lb = loc[b];
      for (int r = 0; r < 3; r++)
        for (int s = 0; s < 3; s++)
          Hm[(size_t) (3 * b + r) * N + 3 * b + s] = mb_Hown[(size_t) lb * 9 + 3 * r + s];
      for (int p = 0; p < mb_n[lb]; p++) {
        tagint ta = mb_tag[(size_t) lb * MBMAX + p];
        int a = -1;
        for (int q = 0; q < n; q++)
          if (tags[q] == ta) a = q;
        if ((a < 0) || (a == b)) continue;
        const double *X = &mb_X[((size_t) lb * MBMAX + p) * 9];
        for (int r = 0; r < 3; r++)
          for (int s = 0; s < 3; s++) Hm[(size_t) (3 * a + r) * N + 3 * b + s] = -X[3 * r + s];
      }
    }
    // asymmetry of the assembled matrix (two independent FD estimates of each cross block)
    double asym = 0.0, nrm = 0.0;
    for (int A = 0; A < N; A++)
      for (int B = 0; B < N; B++) {
        double d = Hm[(size_t) A * N + B] - Hm[(size_t) B * N + A];
        asym += d * d;
        nrm += Hm[(size_t) A * N + B] * Hm[(size_t) A * N + B];
      }
    asym_sum += (nrm > 0.0) ? sqrt(0.5 * asym / nrm) : 0.0;

    double ms[MBMAX + 1];
    for (int a = 0; a < n; a++) ms[a] = rmass ? rmass[loc[a]] : mass[type[loc[a]]];
    double **D, **V, *ev;
    memory->create(D, N, N, "baoab/tether:mbD");
    memory->create(V, N, N, "baoab/tether:mbV");
    memory->create(ev, N, "baoab/tether:mbev");
    for (int A = 0; A < N; A++)
      for (int B = 0; B < N; B++) {
        double hs = 0.5 * (Hm[(size_t) A * N + B] + Hm[(size_t) B * N + A]);
        D[A][B] = hs / sqrt(ms[A / 3] * ms[B / 3]);
      }
    MathEigen::Jacobi<double, double *, double **, double const *const *> ecalc(N);
    ecalc.Diagonalize(D, ev, V);    // decreasing eigenvalues, eigenvectors in rows

    // orthonormal rigid-body basis (mass-weighted), positions minimum-imaged to atom 0
    double xr[MBMAX + 1][3], com[3] = {0.0, 0.0, 0.0}, mt = 0.0;
    for (int a = 0; a < n; a++) {
      double d[3] = {x[loc[a]][0] - x[i][0], x[loc[a]][1] - x[i][1], x[loc[a]][2] - x[i][2]};
      domain->minimum_image(FLERR, d);
      for (int k = 0; k < 3; k++) {
        xr[a][k] = d[k];
        com[k] += ms[a] * d[k];
      }
      mt += ms[a];
    }
    for (int k = 0; k < 3; k++) com[k] /= mt;
    std::vector<std::vector<double>> R;
    for (int k = 0; k < 6; k++) {
      std::vector<double> u(N, 0.0);
      for (int a = 0; a < n; a++) {
        double sm = sqrt(ms[a]);
        double rr[3] = {xr[a][0] - com[0], xr[a][1] - com[1], xr[a][2] - com[2]};
        if (k < 3) {
          u[3 * a + k] = sm;
        } else {
          double e[3] = {0.0, 0.0, 0.0};
          e[k - 3] = 1.0;
          u[3 * a + 0] = sm * (e[1] * rr[2] - e[2] * rr[1]);
          u[3 * a + 1] = sm * (e[2] * rr[0] - e[0] * rr[2]);
          u[3 * a + 2] = sm * (e[0] * rr[1] - e[1] * rr[0]);
        }
      }
      for (auto &w : R) {
        double dp = 0.0;
        for (int A = 0; A < N; A++) dp += w[A] * u[A];
        for (int A = 0; A < N; A++) u[A] -= dp * w[A];
      }
      double nn = 0.0;
      for (int A = 0; A < N; A++) nn += u[A] * u[A];
      if (nn > 1.0e-12) {
        for (int A = 0; A < N; A++) u[A] /= sqrt(nn);
        R.push_back(u);
      }
    }
    std::vector<std::pair<double, int>> rf(N);
    for (int m = 0; m < N; m++) {
      double f = 0.0;
      for (auto &w : R) {
        double dp = 0.0;
        for (int A = 0; A < N; A++) dp += w[A] * V[m][A];
        f += dp * dp;
      }
      rf[m] = {f, m};
    }
    std::sort(rf.begin(), rf.end());
    int nint = N - (int) R.size();
    std::vector<double> wint, wrig;
    for (int q = 0; q < N; q++) {
      int m = rf[q].second;
      double w = sqrt(MAX(ev[m], 0.0)) * conv;
      if (q < nint) {
        wint.push_back(w);
        purity_sum += 1.0 - rf[q].first;
        worst_purity = MIN(worst_purity, 1.0 - rf[q].first);
      } else {
        wrig.push_back(w);
      }
    }
    std::sort(wint.rbegin(), wint.rend());
    std::sort(wrig.rbegin(), wrig.rend());
    for (int q = 0; q < (int) wint.size(); q++) {
      s_int[q] += wint[q];
      s2_int[q] += wint[q] * wint[q];
    }
    for (int q = 0; q < (int) wrig.size(); q++) s_rig[q] += wrig[q];
    nmol++;
    memory->destroy(D);
    memory->destroy(V);
    memory->destroy(ev);
  }

  if (comm->me != 0) return;
  if (nmol == 0) {
    utils::logmesg(lmp, "MOLBLOCK step {} no complete molecules (skipped {})\n", update->ntimestep,
                   nskip);
    return;
  }
  int N = 3 * nsz, nint = N - 6;
  std::string line = fmt::format("MOLBLOCK step {} molecules {} skipped {} | internal:",
                                 update->ntimestep, nmol, nskip);
  for (int q = 0; q < nint; q++) {
    double m = s_int[q] / nmol, sd = sqrt(MAX(s2_int[q] / nmol - m * m, 0.0));
    line += fmt::format(" {:.0f}+-{:.0f}", m * cm1, sd * cm1);
  }
  line += " | rigid-body:";
  for (int q = 0; q < 6; q++) line += fmt::format(" {:.0f}", s_rig[q] / nmol * cm1);
  line += fmt::format(" | internal purity mean {:.3f} worst {:.3f} | cross-block asymmetry {:.3f}\n",
                      purity_sum / (nmol * nint), worst_purity, asym_sum / nmol);
  utils::logmesg(lmp, line);
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
bool FixBAOABTether::mb_locate(const MBMol &M, int *loc, double (*xr)[3]) const
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
void FixBAOABTether::mb_rel(const MBMol &M, const double (*xa)[3], double (*xr)[3]) const
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
int FixBAOABTether::mb_pack(const MBMol &M, double *buf) const
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

void FixBAOABTether::mb_unpack(MBMol &M, const double *buf) const
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
void FixBAOABTether::mb_gather()
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
      error->one(FLERR, "Fix baoab/tether molblock: the molecule of atom ID {} is not fully within "
                 "this processor's ghost atoms; increase the communication cutoff (comm_modify cutoff)",
                 tag[i]);
    MBMol M;
    mb_unpack(M, mb_state[j]);
    mbmols.push_back(M);
    mb_aidx.push_back(j);
  }
}

// write each copy's state back to its anchor (owned or ghost) and the anchor tag to owned members
void FixBAOABTether::mb_scatter()
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

double FixBAOABTether::mb_gauss(tagint tag, int k) const
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

void FixBAOABTether::mb_refresh()
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
  mb_collect = true;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    refresh_one_atom(i, xi, false);
  }
  mb_collect = false;
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
      if (M.nic >= MBN) error->one(FLERR, "Fix baoab/tether molblock: molecule too large");
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
      error->one(FLERR, "Fix baoab/tether molblock: the molecule with atom ID {} has more bonds and "
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
    error->all(FLERR, "Fix baoab/tether molblock: {} groups of bonded atoms could not be treated as "
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
void FixBAOABTether::mb_lin_sym(MBMol &M, double (*F)[MBN]) const
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
void FixBAOABTether::mb_lin_reframe(MBMol &M, const double (*xr)[3]) const
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
void FixBAOABTether::mb_ic_eval(const MBMol &M, const double (*xr)[3], double *sig,
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
void FixBAOABTether::mb_gf_forceconst(MBMol &M, const double (*xr)[3], const double (*fr)[3])
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
void FixBAOABTether::mb_gf_freqs(MBMol &M, const double (*xr)[3])
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
    error->one(FLERR, "Fix baoab/tether molblock: the bonds and angles of the molecule with atom "
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
void FixBAOABTether::mb_fast_force(const MBMol &M, const double (*xr)[3], double (*ff)[3],
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
void FixBAOABTether::mb_ic_force(const MBMol &M, const double (*xr)[3], const int *loc, double *sg,
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
void FixBAOABTether::mb_reestimate()
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
void FixBAOABTether::mb_prepare()
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
      error->one(FLERR, "Fix baoab/tether molblock: the molecule of atom ID {} is not fully within "
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

void FixBAOABTether::mb_initial_integrate()
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
int FixBAOABTether::mb_stiff_modes(const MBMol &M, const double (*xr)[3]) const
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
void FixBAOABTether::mb_final_integrate()
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
      error->one(FLERR, "Fix baoab/tether molblock: the molecule of atom ID {} is not fully within "
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

  // eligibility bonded: refresh each owned atom's "has a bond" flag from the
  // (symmetric, newton_bond off) per-atom bond count. Ghosts receive it via
  // the same forward comm as om[] (see pack_forward_comm()).
  if ((eligibility_bonded == 1) && atom->num_bond) {
    for (int i = 0; i < nlocal; i++) bonded[i] = (atom->num_bond[i] > 0) ? 1 : 0;
  }

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

  // DIAGNOSTIC-ONLY heavy_group shortcut (see header comment by
  // heavy_groupbit): must run BEFORE the om_split bootstrap below, since
  // Python's reference caches its frozen heavy-block estimate on the very
  // first hessian_blocks() call too -- om_split itself is computed from a
  // spectrum that already has every heavy atom pinned to the same frozen
  // value, on both sides, for a true apples-to-apples match.
  if (heavy_groupbit && !heavy_block_set) compute_heavy_block();

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

      if (heavy_groupbit && (mask[i] & heavy_groupbit)) {
        double mi = rmass ? rmass[i] : mass[type[i]];
        double om_scalar = sqrt(MAX(heavy_block_tr3 / mi, 0.0));
        for (int m = 0; m < 3; m++) {
          om[i][m] = om_scalar;
          for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = (a == m) ? 1.0 : 0.0;
        }
        continue;
      }

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

  if (molblock_diag) {
    mb_Hown.assign((size_t) nlocal * 9, 0.0);
    mb_X.assign((size_t) nlocal * 4 * 9, 0.0);
    mb_tag.assign((size_t) nlocal * 4, 0);
    mb_n.assign(nlocal, 0);
    mb_collect = true;
  }
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    double xi[3] = {x[i][0], x[i][1], x[i][2]};
    refresh_one_atom(i, xi, false);
  }
  if (molblock_diag) {
    mb_collect = false;
    molblock_diagnostic();
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

void FixBAOABTether::recoil_accumulate(int i, const double *Hck, double mj, double *dom2) const
{
  if (!recoil || mj <= 0.0) return;
  for (int m = 0; m < 3; m++) {
    if (!mode_is_stiff(i, m)) continue;
    const double *e = &evecs[i][3 * m];
    double q = 0.0;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) q += e[a] * Hck[3 * a + b] * e[b];
    if (q > 0.0) dom2[m] += q / mj;
  }
}

void FixBAOABTether::refresh_one_atom(int i, const double *xi, bool instantaneous)
{
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;

  // DIAGNOSTIC-ONLY heavy_group shortcut (see header comment by
  // heavy_groupbit): skip the real FD-Hessian probe entirely for this
  // atom, broadcasting the single frozen isotropic estimate instead --
  // matches Python's HPdEngine.hessian_blocks() returning the same cached
  // _heavy_block for every heavy atom on every call. No response block is
  // built (nJ[i] = 0): mode_is_stiff() can never go true against the
  // frozen, below-om_split estimate this is meant to reproduce, so the
  // real Si/Jblk computation would be all zeros anyway.
  if (heavy_groupbit && (mask[i] & heavy_groupbit)) {
    double mi = rmass ? rmass[i] : mass[type[i]];
    double om_scalar = sqrt(MAX(heavy_block_tr3 / mi, 0.0));
    for (int m = 0; m < 3; m++) {
      om[i][m] = om_scalar;
      for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = (a == m) ? 1.0 : 0.0;
    }
    log_event("REFRESH", atom->tag[i], -1.0, -1.0, -1.0, -1.0, om[i], -1.0,
              instantaneous ? "forced_heavy_frozen" : "scheduled_heavy_frozen");
    x0[i][0] = xi[0];
    x0[i][1] = xi[1];
    x0[i][2] = xi[2];
    c[i][0] = xi[0];
    c[i][1] = xi[1];
    c[i][2] = xi[2];
    nJ[i] = 0;
    return;
  }

  double mi = rmass ? rmass[i] : mass[type[i]];
  double H[3][3];

  int jnum = 0;
  int *jlist = nullptr;
  if (mollify && list && !j_free) {
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

  // Bonded-partner response columns (see partial_force_add_bonded()): a
  // stiff bond's partner is typically excluded from the pairwise jlist
  // above by special_bonds, so it needs its own, separate Hc-style buffer
  // -- n_bond is the same every FD evaluation (atom i's own bond topology
  // doesn't change within one refresh_one_atom() call), so the last
  // (b==2) probe's count is authoritative. No-op allocation (BOND_JMAX is
  // a small fixed cap) when force->bond is unset.
  std::vector<tagint> bond_tag_buf;
  std::vector<double> bj_plus, bj_minus, Hc_bond;
  int n_bond = 0;
  if (mollify && force->bond) {
    bond_tag_buf.resize(BOND_JMAX);
    bj_plus.resize((size_t) BOND_JMAX * 3);
    bj_minus.resize((size_t) BOND_JMAX * 3);
    Hc_bond.assign((size_t) BOND_JMAX * 9, 0.0);
  }

  // THROWAWAY DIAGNOSTIC (pforce_debug_tag): zero-perturbation sanity
  // check, independent of the eps-probe loop below -- do local_partial_force()
  // and full_system_force() agree with EACH OTHER (not with any cached
  // atom->f[i]) at xtrial == xi exactly? Any disagreement here is a pure
  // formula/implementation bug, not a staleness or clamp-convention
  // question, since no displacement or clamp substitution is involved
  // (jnum==0 below forces live_neighbor_positions's clamp branch moot).
  if (!force_eval_full && pforce_debug_tag >= 0 && atom->tag[i] == pforce_debug_tag) {
    double f0_local[3], f0_full[3];
    partial_force(i, xi, f0_local, nullptr, true);
    full_system_force(i, xi, f0_full, nullptr, 0, nullptr);
    utils::logmesg(lmp, "PFORCE_ZEROPROBE step={} tag={} f0_local=({:.6g},{:.6g},{:.6g}) "
                   "f0_full=({:.6g},{:.6g},{:.6g})\n", update->ntimestep, pforce_debug_tag,
                   f0_local[0], f0_local[1], f0_local[2], f0_full[0], f0_full[1], f0_full[2]);
    // undo full_system_force()'s rho[]/fp[] cache corruption -- see the
    // identical note in partial_force()'s pforce_debug_tag branch.
    double xnow[3] = {atom->x[i][0], atom->x[i][1], atom->x[i][2]};
    double f0_restore_dummy[3];
    full_system_force(i, xnow, f0_restore_dummy, nullptr, 0, nullptr);
  }

  for (int b = 0; b < 3; b++) {
    double fplus[3], fminus[3];
    double xt[3] = {xi[0], xi[1], xi[2]};
    int n_bond_plus = 0, n_bond_minus = 0;
    bool have_bond = mollify && force->bond;

    xt[b] = xi[b] + eps;
    partial_force(i, xt, fplus, jnum > 0 ? fj_plus.data() : nullptr, true,
                  have_bond ? bond_tag_buf.data() : nullptr, have_bond ? bj_plus.data() : nullptr,
                  have_bond ? &n_bond_plus : nullptr);

    xt[b] = xi[b] - eps;
    partial_force(i, xt, fminus, jnum > 0 ? fj_minus.data() : nullptr, true,
                  have_bond ? bond_tag_buf.data() : nullptr, have_bond ? bj_minus.data() : nullptr,
                  have_bond ? &n_bond_minus : nullptr);

    for (int a = 0; a < 3; a++) H[a][b] = -(fplus[a] - fminus[a]) / (2.0 * eps);

    for (int k = 0; k < jnum; k++)
      for (int a = 0; a < 3; a++)
        Hc[(size_t) k * 9 + 3 * a + b] =
            (fj_plus[(size_t) k * 3 + a] - fj_minus[(size_t) k * 3 + a]) / (2.0 * eps);

    // n_bond_plus == n_bond_minus by construction (atom i's bond topology
    // is fixed within this call); take either, last iteration wins.
    n_bond = n_bond_plus;
    for (int k = 0; k < n_bond; k++)
      for (int a = 0; a < 3; a++)
        Hc_bond[(size_t) k * 9 + 3 * a + b] =
            (bj_plus[(size_t) k * 3 + a] - bj_minus[(size_t) k * 3 + a]) / (2.0 * eps);
  }

  double Hs[3][3];
  for (int a = 0; a < 3; a++)
    for (int b = 0; b < 3; b++) Hs[a][b] = 0.5 * (H[a][b] + H[b][a]) / mi;

  if (mb_collect && i < (int) mb_n.size()) {
    const int MBMAX = 4;
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) mb_Hown[(size_t) i * 9 + 3 * a + b] = 0.5 * (H[a][b] + H[b][a]);
    int np = 0;
    double *X = &mb_X[(size_t) i * MBMAX * 9];
    tagint *T = &mb_tag[(size_t) i * MBMAX];
    for (int s = 0; s < MBMAX * 9; s++) X[s] = 0.0;
    // molecule partners = bonded/angle partners (their bonded columns), plus any pairwise column
    // for the same tag (intramolecular non-bonded part, e.g. the excluded-pair Coulomb correction).
    for (int k = 0; k < n_bond; k++) {
      tagint t = bond_tag_buf[k];
      int p = -1;
      for (int q = 0; q < np; q++)
        if (T[q] == t) p = q;
      if (p < 0) {
        if (np >= MBMAX) continue;
        T[np] = t;
        p = np++;
      }
      for (int s = 0; s < 9; s++) X[p * 9 + s] += Hc_bond[(size_t) k * 9 + s];
    }
    for (int k = 0; k < jnum; k++) {
      tagint t = atom->tag[jlist[k] & NEIGHMASK];
      int p = -1;
      for (int q = 0; q < np; q++)
        if (T[q] == t) p = q;
      if (p < 0) continue;
      for (int s = 0; s < 9; s++) X[p * 9 + s] += Hc[(size_t) k * 9 + s];
    }
    mb_n[i] = np;
  }

  // thermally-averaged effective curvature (CLAUDE.md convention #3):
  // heff_samples==0 keeps the legacy instantaneous-every-refresh write
  // below unchanged. heff_samples>0 maintains a continuously-updating
  // exponential moving average (EMA) of the raw Hessian, re-diagonalizing
  // the working om[i]/evecs[i] -- the values mode_is_stiff() actually masks
  // against -- at EVERY scheduled sample, for as long as the run continues.
  // heff_samples is interpreted as the EMA time constant: alpha = 1/N gives
  // the same steady-state noise suppression as an N-sample box average,
  // with a half-life of ln(2)*N refresh cycles instead of a hard N-cycle
  // cutoff. This replaces an earlier sum-then-freeze-forever scheme (average
  // once when an N-sample window closed, then never touch om[i]/evecs[i]
  // again) that measurably made cached-mask staleness worse, not better
  // (see [[heff_samples_increases_mode_staleness]]) -- a wider, one-shot
  // freeze window is not the same thing as a live average, and does not
  // track a genuinely anharmonic rattler's slowly-evolving local well the
  // way a continuously-updated EMA does. An instantaneous=true emergency
  // re-tether always takes the first branch below, bypassing the EMA
  // entirely -- it needs the current local curvature right now, not the
  // smoothed estimate. When heff_samples>0 it also RE-ARMS the EMA by
  // seeding Hacc[i] with this same probe's Hs and setting
  // n_hess_samples[i]=1, i.e. treating the emergency sample as a fresh
  // starting point (same idiom as the om_split bootstrap pass above: reuse
  // the instantaneous probe rather than discard it) rather than blending it
  // into whatever average had accumulated before the guard tripped.
  if (instantaneous || heff_samples == 0) {
    double eval[3], evec[3][3];
    MathEigen::jacobi3(Hs, eval, evec);
    for (int m = 0; m < 3; m++) {
      om[i][m] = sqrt(MAX(eval[m], 0.0));
      for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = evec[a][m];
    }
    if (instantaneous && heff_samples > 0) {
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) Hacc[i][3 * a + b] = Hs[a][b];
      n_hess_samples[i] = 1;
    }
  } else {
    bool first_sample = (n_hess_samples[i] == 0);
    if (first_sample) {
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++) Hacc[i][3 * a + b] = Hs[a][b];
    } else {
      double alpha = 1.0 / heff_samples;
      for (int a = 0; a < 3; a++)
        for (int b = 0; b < 3; b++)
          Hacc[i][3 * a + b] = (1.0 - alpha) * Hacc[i][3 * a + b] + alpha * Hs[a][b];
    }
    // saturating counter: no longer gates anything, kept only so
    // first_sample above and any future diagnostic can tell "never
    // sampled" (0) from "running" (>0) without a separate flag.
    if (n_hess_samples[i] < heff_samples) n_hess_samples[i]++;

    double Havg[3][3];
    for (int a = 0; a < 3; a++)
      for (int b = 0; b < 3; b++) Havg[a][b] = Hacc[i][3 * a + b];

    double eval[3], evec[3][3];
    MathEigen::jacobi3(Havg, eval, evec);
    for (int m = 0; m < 3; m++) {
      om[i][m] = sqrt(MAX(eval[m], 0.0));
      for (int a = 0; a < 3; a++) evecs[i][3 * m + a] = evec[a][m];
    }
  }

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
  //
  // "Flagged (stiff/rattler)" used to mean "in this fix's own group" back
  // when that group WAS exactly the rattler population (groupbit ==
  // light-only, two-fix era) -- mask[j] & groupbit correctly meant "j is a
  // rattler." Now that this fix can operate on a single whole-system group
  // (e.g. "all"), every neighbor carries groupbit regardless of species,
  // which silently excluded EVERY neighbor and zeroed the entire
  // back-reaction (nJ[i] == 0 for every atom, confirmed via identical event
  // logs with and without the diagnostic j_free flag -- see
  // python_heavy_block_frozen_approximation memory). The correct, general
  // test is CURRENT stiff/rattler ELIGIBILITY (the same om-based gate
  // mode_is_stiff() uses, minus its demote-cooldown check, which would need
  // demote_cool[] forward-communicated too and isn't) -- om[] now IS
  // forward-communicated to ghosts (see pack_forward_comm()) specifically
  // so this works for a possibly-ghost neighbor j.
  double dom2[3] = {0.0, 0.0, 0.0};    // recoil: sum_j max(0, e_m^T Hc_j e_m)/m_j per mode
  int nk = 0;
  for (int k = 0; k < jnum; k++) {
    int j = jlist[k] & NEIGHMASK;
    double om_max_j = MAX(om[j][0], MAX(om[j][1], om[j][2]));
    bool j_eligible;
    if (adapt) {
      double om_min_j = MIN(om[j][0], MIN(om[j][1], om[j][2]));
      j_eligible = (om_max_j > om_split) && (bonded_ok(j) || (om_min_j > 0.6 * om_split));
    } else {
      j_eligible = (om_max_j * sqrt(force->ftm2v) * update->dt) > theta;
    }
    if (j_eligible) continue;
    Jtag[i][nk] = atom->tag[j];
    const double *Hck = &Hc[(size_t) k * 9];
    recoil_accumulate(i, Hck, rmass ? rmass[j] : mass[type[j]], dom2);
    // Jblk_k = Hc_k . S_i, so that the back-reaction is a plain matvec
    // at force_moll() time: f[j] += Jblk_k * g[i] (see force_moll()).
    mat3_matmat(Hck, Si, Jblk[i][nk]);
    nk++;
  }

  // Bonded-partner back-reaction slots: a bond partner (e.g. water's O-H)
  // is typically EXCLUDED from the pairwise jlist above by special_bonds,
  // so it would otherwise never receive a back-reaction block at all --
  // yet it is usually the single most important neighbor for a stiff bond
  // (O's response to its own bonded H's fast motion, and vice versa). Same
  // eligibility test and S_i matvec as the pairwise loop above, just a
  // different source for (tag, Hc). No dedup needed against the pairwise
  // slots above: a partner that happened to appear in both (not expected
  // here, given special_bonds exclusion) would just contribute two
  // genuinely-additive partial force-response terms in two separate slots
  // -- force_moll()'s back-reaction loop already sums over all nJ[i] slots
  // regardless of repeated tags, which is exactly the physically correct
  // total (dF_j/dx_i = pairwise + bonded contributions).
  for (int k = 0; k < n_bond; k++) {
    int j = atom->map(bond_tag_buf[k]);
    if (j < 0) continue;
    j = domain->closest_image(i, j);
    if (neighbor_clamp_eligible(j)) continue;
    if (nk >= JMAX)
      error->one(FLERR, "Fix baoab/tether: atom exceeds JMAX ({}) combined pair+bond "
                         "neighbors for the sparse response block -- raise JMAX and recompile",
                 JMAX);
    recoil_accumulate(i, &Hc_bond[(size_t) k * 9], rmass ? rmass[j] : mass[type[j]], dom2);
    Jtag[i][nk] = bond_tag_buf[k];
    mat3_matmat(&Hc_bond[(size_t) k * 9], Si, Jblk[i][nk]);
    nk++;
  }

  // Partner-recoil (reduced-mass) correction to the analytic mode frequencies. The stiff mode
  // above was extracted from atom i's OWN curvature block with every neighbor held fixed, i.e.
  // omega^2 = k/m_i; a free partner j sharing that coupling recoils and the mode actually runs at
  // k/m_i + k/m_j (diatomic: reduced mass). Hc_j = dF_j/dx_i is exactly that coupling, so add
  // e_m^T Hc_j e_m / m_j for every non-clamped neighbor. Negligible for a light atom in a heavy
  // cage (H/Pd: ~mH/mPd), ~3% in omega for a water O-H stretch (m_H/m_O). Applied to om[] after
  // the compliance Si above (the static response of the neighbors keeps the local stiffness).
  if (recoil) {
    for (int m = 0; m < 3; m++) {
      if (dom2[m] > 0.0 && mode_is_stiff(i, m) && om[i][m] > 0.0)
        om[i][m] = sqrt(om[i][m] * om[i][m] + dom2[m]);
    }
  }

  nJ[i] = nk;
}

/* ----------------------------------------------------------------------
   Stage-20 read-only diagnostic (long-run light-subsystem heating
   investigation): re-probes atom i's curvature at its CURRENT position
   (same FD-Hessian + jacobi3 math as refresh_one_atom(), minus the
   neighbor response-block bookkeeping, which this diagnostic does not
   need) and re-applies mode_is_stiff()'s eligibility logic against the
   fresh estimate, using the SAME frozen om_split threshold -- never
   recomputed here. Writes nothing to om[]/evecs[]/c[]/x0[]/Jtag/Jblk/
   Hacc/n_hess_samples -- purely additive, no effect on dynamics.

   Comparison is by COUNT of masked modes per atom (0..3), not per mode
   index: H-in-Pd's three curvature modes are nearly degenerate (see
   effective_skin()'s doc comment, om~0.6 for every mode), so two
   independent diagonalizations of nearly-equal eigenvalues can permute
   which physical mode lands in jacobi3's slot 0/1/2 even when nothing
   has gone stale -- an index-wise comparison was tried first and showed
   ~100% spurious "disagreement" from this basis-labeling artifact alone,
   with no sensitivity to actual staleness. The masked-mode COUNT is
   permutation-invariant and reflects the only thing that actually
   matters for mollify's tethering decision: how many of this atom's
   modes get the analytic/quench treatment vs. free flight.
------------------------------------------------------------------------- */

void FixBAOABTether::diag_check_staleness(int i, double mi, const bool *msk_cached)
{
  double xi[3] = {atom->x[i][0], atom->x[i][1], atom->x[i][2]};
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
  double om_fresh[3];
  for (int m = 0; m < 3; m++) om_fresh[m] = sqrt(MAX(eval[m], 0.0));

  // mode_is_stiff()'s eligibility test, evaluated against om_fresh instead
  // of the cached om[i] -- deliberately duplicated rather than refactored
  // out of mode_is_stiff(), so this diagnostic stays trivially removable.
  bool msk_fresh[3];
  if (!adapt) {
    // same natural-tau -> real-ps conversion as mode_is_stiff() -- see its
    // doc comment in fix_baoab_tether.h.
    double om2real = sqrt(force->ftm2v);
    for (int m = 0; m < 3; m++) msk_fresh[m] = (om_fresh[m] * om2real * update->dt) > theta;
  } else if (demote_cool[i] > update->ntimestep) {
    for (int m = 0; m < 3; m++) msk_fresh[m] = false;
  } else {
    double om_max = MAX(om_fresh[0], MAX(om_fresh[1], om_fresh[2]));
    double om_min = MIN(om_fresh[0], MIN(om_fresh[1], om_fresh[2]));
    bool eligible = (om_max > om_split) && (bonded_ok(i) || (om_min > 0.6 * om_split));
    for (int m = 0; m < 3; m++) msk_fresh[m] = eligible && (om_fresh[m] > om_split);
  }

  int n_cached_masked = 0, n_fresh_masked = 0;
  for (int m = 0; m < 3; m++) {
    if (msk_cached[m]) n_cached_masked++;
    if (msk_fresh[m]) n_fresh_masked++;
  }
  if (n_cached_masked == n_fresh_masked) {
    n_mask_agree++;
  } else if (n_fresh_masked > n_cached_masked) {
    n_mask_stale_soft++;
  } else {
    n_mask_stale_tight++;
  }
}

/* ----------------------------------------------------------------------
   Effective drift-guard skin for atom i: refresh_skin_rel > 0 scales to a
   fixed multiple of this atom's OWN thermal ballistic step at dt_max,
   sqrt(kT/m_i)*dt_max (RMS thermal speed times one full-size timestep),
   so one setting works across species of very different mass instead of a
   single absolute distance tuned for one species. Deliberately NOT a
   harmonic-equipartition amplitude (sqrt(kT/m_i)/om): the atoms this guard
   exists for are precisely the ones with persistently weak/soft curvature
   in some or all modes (om -> 0 for an unconfined rattler), where a 1/om
   amplitude diverges and silently disables the guard it is supposed to be
   -- confirmed by direct test (H/Pd EAM: om~0.6 for every mode, any
   om-based skin > ~20x refresh_skin's original 0.1 A, drift guard never
   trips, run diverges within 14 steps). The ballistic form has no such
   singularity and uses dt_max (not the current ladder rung) so the skin
   stays a fixed length regardless of dt_lvl, matching refresh_skin's own
   fixed-distance convention. Falls back to the plain refresh_skin (0.0 if
   also unset) when refresh_skin_rel is disabled.
------------------------------------------------------------------------- */

double FixBAOABTether::effective_skin(int i, double mi, double kT) const
{
  if (refresh_skin_rel <= 0.0) return refresh_skin;
  return refresh_skin_rel * sqrt(kT / mi) * dt_max;
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
  // theta/update->dt is a real ps^-1 quantity; om_split is compared directly
  // against om[]/om_fresh elsewhere, which are natural tau^-1 units (see
  // mode_is_stiff()'s doc comment) -- divide by sqrt(force->ftm2v) to land
  // om_split in that same natural-unit scale.
  if (freqs.size() < 2) {
    om_split = (theta / update->dt) / sqrt(force->ftm2v);
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

  // best_gap branch: geometric mean of two already-natural-unit frequencies,
  // self-consistent as-is. Fallback branch: same real-ps -> natural-unit
  // conversion as the degenerate-group fallback above.
  om_split = (best_gap > 0.0) ? sqrt(freqs[best_k] * freqs[best_k + 1])
                               : (theta / update->dt) / sqrt(force->ftm2v);
  om_split_set = 1;
}

/* ----------------------------------------------------------------------
   DIAGNOSTIC-ONLY, deliberate match to lj_longstep_prototype.py's
   HPdEngine.hessian_blocks(): sample heavy_sample atoms from heavy_group
   ONCE, average their raw (not mass-weighted) FD-Hessian trace/3, and
   freeze that single isotropic scalar for the rest of the run.
   refresh_one_atom() broadcasts sqrt(heavy_block_tr3 / mass[i]) to every
   mode of every heavy_group atom instead of running a real per-refresh
   probe on them -- see the header comment by heavy_groupbit and the
   python_heavy_block_frozen_approximation memory for why this exists and
   why it is NOT validated reference physics. Each rank samples only its
   own locally-owned heavy_group atoms, in local-index order (closely
   matching, not byte-identical to, Python's self.heavy[::step][:N] slice
   order); an MPI_Allreduce sum of the per-rank trace sum and sample count
   gives every rank the identical global average under multi-rank, cheaper
   than compute_om_split()'s full sorted-array Allgatherv since only two
   scalars need to agree here.
------------------------------------------------------------------------- */

void FixBAOABTether::compute_heavy_block()
{
  double **x = atom->x;
  int *mask = atom->mask;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  std::vector<int> heavy_idx;
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & heavy_groupbit) heavy_idx.push_back(i);

  int n_heavy = (int) heavy_idx.size();
  int step = MAX(1, n_heavy / MAX(heavy_sample, 1));

  double trace_sum = 0.0;
  int n_sampled = 0;
  for (int k = 0; k < n_heavy && n_sampled < heavy_sample; k += step) {
    int i = heavy_idx[k];
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
    // raw (not mass-weighted) trace/3, matching Python's np.trace(B)/3 --
    // the symmetrization Python skips too (trace is unaffected by it).
    trace_sum += (H[0][0] + H[1][1] + H[2][2]) / 3.0;
    n_sampled++;
  }

  if (comm->nprocs > 1) {
    double local[2] = {trace_sum, (double) n_sampled};
    double global[2];
    MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, world);
    trace_sum = global[0];
    n_sampled = (int) global[1];
  }

  heavy_block_tr3 = (n_sampled > 0) ? trace_sum / n_sampled : 0.0;
  heavy_block_set = 1;
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

void FixBAOABTether::update_dt(bool fresh_sample)
{
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double om_unres = 0.0;
  double om_analytic_max = 0.0;
  int unres_type = 0;
  std::vector<double> unres_all;
  // per_mode percentile rule (see below): log-spaced histogram of maskable mode frequencies
  // (1/ps, bins 1% wide from 0.1 to about 1e5), and the maximum over modes that cannot be masked
  const bool per_mode_dt = (eligibility_bonded == 2);
  constexpr int PM_NBINS = 1400;
  constexpr double PM_LO = 0.1, PM_RATIO = 1.01;
  std::vector<double> pm_hist(per_mode_dt ? PM_NBINS : 0, 0.0);
  auto pm_bin = [&](double w) {
    if (w <= PM_LO) return 0;
    return MIN(PM_NBINS - 1, (int) (log(w / PM_LO) / log(PM_RATIO)));
  };
  double om_hard = 0.0, om_pop_max = 0.0;
  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    // exile_substep: this atom is already safely self-integrating via
    // sub_step_free_atom()'s own locally resonance-bounded dt_sub,
    // decoupled from the shared update->dt -- its (possibly large,
    // rattling) curvature must not be allowed to veto the dt this fix picks
    // for every OTHER atom, which is exactly the coupling this keyword
    // exists to remove. See initial_integrate()/final_integrate().
    if (exile_substep && demote_cool[i] > update->ntimestep) continue;
    // Convert here, at the natural-tau -> real-ps boundary (see
    // mode_is_stiff()'s doc comment): om_analytic_max/om_unres must be real
    // ps^-1 throughout the rest of this function (they feed gamma_quench and
    // dt_c directly), including composing correctly with om_unres's already-
    // real-unit zero-fallback (c_acc/dt_max) below -- converting again at any
    // later point of use would double-convert that fallback.
    double om2real = sqrt(force->ftm2v);
    bool cooling = (demote_cool[i] > update->ntimestep);
    for (int m = 0; m < 3; m++) {
      double w = om[i][m] * om2real;
      if (mode_is_stiff(i, m))
        om_analytic_max = MAX(om_analytic_max, w);
      else {
        unres_all.push_back(w);
        if (w > om_unres) {
          om_unres = w;
          unres_type = atom->type[i];
        }
      }
      if (per_mode_dt) {
        if (cooling)
          om_hard = MAX(om_hard, w);
        else if (om[i][m] <= om_split) {
          pm_hist[pm_bin(w)] += 1.0;
          om_pop_max = MAX(om_pop_max, w);
        }
      }
    }
  }
  diag_om_unres = om_unres;
  diag_unres_type = unres_type;
  diag_om_analytic = om_analytic_max;
  diag_om_unres_p99 = 0.0;
  if (!unres_all.empty()) {
    size_t kq = (size_t) (0.99 * (double) (unres_all.size() - 1));
    std::nth_element(unres_all.begin(), unres_all.begin() + kq, unres_all.end());
    diag_om_unres_p99 = unres_all[kq];
  }

  if (comm->nprocs > 1) {
    double local[4] = {om_unres, om_analytic_max, om_hard, om_pop_max};
    double global[4];
    MPI_Allreduce(local, global, 4, MPI_DOUBLE, MPI_MAX, world);
    om_unres = global[0];
    om_analytic_max = global[1];
    om_hard = global[2];
    om_pop_max = global[3];
  }
  // frequency that decides a step-size INCREASE (and the first rung); see per_mode below
  double om_up = om_unres;

  // per_mode: a rung is ENTERED (first rung, relax-up) only if the maximum of every maskable
  // mode allows it, exactly as before; it is LEFT (shrink) only if the 99th percentile of
  // the maskable modes, or the maximum over atoms that cannot be masked (demotion cooldown),
  // requires it. Every maskable mode above c_acc/(0.85*dt) is masked (om_mask_thr, set in
  // apply_new_dt()), so no unresolved mode violates the accuracy bound while the step holds.
  // The maximum of N modes keeps growing with N, a percentile does not: a rare fast mode in
  // a large system no longer halves dt for every atom (H/Pd: 2.8 fs instead of 5 fs at
  // 24624 atoms with the maximum), while a shift of the bulk of the spectrum still shrinks
  // it. Promoting on the percentile instead would mask the top 1% of the bulk spectrum at a
  // larger rung (seen on the 114-atom production deck: 10 fs, 10x more Newton guard trips).
  if (per_mode_dt) {
    if (comm->nprocs > 1) MPI_Allreduce(MPI_IN_PLACE, pm_hist.data(), PM_NBINS, MPI_DOUBLE,
                                        MPI_SUM, world);
    double total = 0.0;
    for (int b = 0; b < PM_NBINS; b++) total += pm_hist[b];
    double om_q = 0.0;
    if (total > 0.0) {
      double target = 0.99 * total, cum = 0.0;
      for (int b = 0; b < PM_NBINS; b++) {
        cum += pm_hist[b];
        if (cum >= target) {
          om_q = PM_LO * pow(PM_RATIO, b + 1);   // upper bin edge (conservative)
          break;
        }
      }
    }
    diag_om_unres_p99 = om_q;
    om_unres = MAX(om_q, om_hard);
    om_up = MAX(om_pop_max, om_hard);
  }

  // gamma_quench auto (continuous, not a one-time warm-up calibration):
  // re-derive gamma_quench from the SAME already-collective, already-
  // mode_is_stiff()/exile_substep-filtered global curvature signal that
  // governs dt itself, every time this function runs -- i.e. every refresh
  // cycle for the life of the run, not just once. Critically-damped
  // ansatz, margin 1.0: friction on the order of the stiffest mode it is
  // meant to damp. om_analytic_max <= 0.0 (no stiff atom classified yet)
  // holds the previous value rather than collapsing gamma_quench to a
  // no-damping cq1 = exp(0) = 1 coefficient, mirroring the om_unres
  // fallback immediately below. The relative-change log gate only bounds
  // event_log volume -- it does not throttle how often the value itself
  // updates.
  // gamma_quench_calib (see below) takes over entirely once its own
  // warm-up window has elapsed -- until then, this plain "auto" heuristic
  // IS the warm-up estimate ("estimate from an initial portion of the
  // simulation"), unchanged from today's behavior. Once warm-up ends this
  // clause never fires again for the rest of the run, regardless of
  // gamma_quench_calib_refresh freeze/refine, so a frozen calibrated value
  // can never be silently overwritten back down to om_analytic_max by this
  // block.
  if (gamma_quench_auto && om_analytic_max > 0.0 &&
      (!gamma_quench_calib || update->ntimestep < gamma_quench_calib_warmup)) {
    gamma_quench = om_analytic_max;
    if (gamma_quench_last_logged < 0.0 ||
        fabs(gamma_quench - gamma_quench_last_logged) > 0.2 * gamma_quench_last_logged) {
      log_event("GAMMA_AUTO", -1, -1.0, -1.0, -1.0, -1.0, nullptr, gamma_quench,
                fmt::format("om_analytic_max={}", om_analytic_max));
      gamma_quench_last_logged = gamma_quench;
    }
  }

  // gamma_quench auto calib: closed-loop controller layered on top of plain
  // "auto" (stage-23/24 finding: a flat hardcoded multiplier on
  // om_analytic_max does not generalize across target temperatures or
  // systems -- see fix_baoab_tether.rst and gamma_quench_undercalibrated).
  // After warm-up, every gamma_quench_calib_period steps, Allreduce-close
  // the rolling window kinetic_guard_and_quench() has been filling, turn it
  // into an equipartition temperature of the MASKED subsystem, compare
  // against t_target (self-scaling: rho=1.0 at correct damping regardless
  // of what the target temperature or system actually is), and -- only
  // once gamma_quench_calib_confirm consecutive windows agree on the same
  // direction, the same single-window chaotic-trajectory-noise mitigation
  // settle_confirm already uses elsewhere in this fix -- commit a clamped,
  // log-space multiplicative nudge. gain/max_step/confirm/deadband/ceiling
  // are control-loop stability knobs (how cautiously the loop reacts to its
  // own measurement noise), never a claim about what the correct damping
  // value is -- that is the entire point of this mechanism over a flat C.
  if (gamma_quench_calib && update->ntimestep >= gamma_quench_calib_warmup &&
      !(gamma_quench_calib_committed && !gamma_quench_calib_refine)) {
    if (last_gamma_calib_check < 0) last_gamma_calib_check = gamma_quench_calib_warmup;
    if (update->ntimestep - last_gamma_calib_check >= gamma_quench_calib_period) {
      double sum_ke_g = gamma_quench_calib_sum_ke, sum_dof_g = gamma_quench_calib_sum_dof;
      if (comm->nprocs > 1) {
        double local[2] = {gamma_quench_calib_sum_ke, gamma_quench_calib_sum_dof};
        double global[2];
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, world);
        sum_ke_g = global[0];
        sum_dof_g = global[1];
      }
      // genuine rolling window, not cumulative-forever like ke_rel's Phase
      // B -- reset every close so the controller can track a non-stationary
      // run instead of averaging over its entire history.
      gamma_quench_calib_sum_ke = 0.0;
      gamma_quench_calib_sum_dof = 0.0;
      last_gamma_calib_check = update->ntimestep;

      if (sum_dof_g > 0.0 && t_target > 0.0) {
        double T_masked = 2.0 * sum_ke_g / (force->boltz * sum_dof_g);
        double rho = T_masked / t_target;

        int direction = 0;
        if (rho > gamma_quench_calib_deadband_hi) direction = 1;
        else if (rho < gamma_quench_calib_deadband_lo) direction = -1;

        if (direction == 0 || direction != gamma_quench_calib_direction) {
          gamma_quench_calib_direction = direction;
          gamma_quench_calib_streak = (direction != 0) ? 1 : 0;
        } else {
          gamma_quench_calib_streak++;
        }

        if (direction != 0 && gamma_quench_calib_streak >= gamma_quench_calib_confirm) {
          double step = pow(rho, gamma_quench_calib_gain);
          step = MAX(1.0 / gamma_quench_calib_max_step, MIN(gamma_quench_calib_max_step, step));
          gamma_quench *= step;
          gamma_quench = MAX(om_analytic_max, MIN(gamma_quench_calib_ceiling, gamma_quench));
          log_event("GAMMA_CALIB", -1, -1.0, -1.0, -1.0, -1.0, nullptr, gamma_quench,
                    fmt::format("rho={} T_masked={} streak={} dir={}", rho, T_masked,
                                gamma_quench_calib_streak, direction));
          gamma_quench_last_logged = gamma_quench;
          gamma_quench_calib_committed = 1;
          gamma_quench_calib_streak = 0;
        }
      }
    }
  }

  // ke_rel warm-up calibration (Phase A close + freeze/refine fit):
  // unlike gamma_quench_auto's continuous re-derivation, ke_rel is a
  // rare-event guard threshold, not an ordinary dynamical property, so it
  // cannot be safely read off ordinary-dynamics statistics the way
  // gamma_quench is -- see the header's doc comment and
  // workloads/phase2_stage14_calibrated_params/README.md. Instead: fit an
  // exponential tail (Peaks-Over-Threshold, shape fixed at 0) beyond a
  // fixed Phase A threshold u, and extrapolate to a target rarity
  // (1-in-ke_rel_margin_blocks) rather than reading an empirical
  // percentile, which structurally caps out at whatever was observed.
  // This function is the existing collective, refresh-cadence hook
  // (same one gamma_quench_auto uses) so Phase A closes the first time
  // this runs at or after ke_rel_warmup_steps, and refine mode re-fits
  // here every subsequent refresh cycle.
  if (ke_rel_warmup_steps > 0 && update->ntimestep >= ke_rel_warmup_steps) {
    if (ke_rel_u < 0.0) {
      double hist_global[KE_REL_NBINS];
      if (comm->nprocs > 1)
        MPI_Allreduce(ke_rel_hist, hist_global, KE_REL_NBINS, MPI_DOUBLE, MPI_SUM, world);
      else
        for (int b = 0; b < KE_REL_NBINS; b++) hist_global[b] = ke_rel_hist[b];

      double total = 0.0;
      for (int b = 0; b < KE_REL_NBINS; b++) total += hist_global[b];
      if (total > 0.0) {
        double target = 0.9 * total;
        double cum = 0.0;
        int ubin = KE_REL_NBINS - 1;
        for (int b = 0; b < KE_REL_NBINS; b++) {
          cum += hist_global[b];
          if (cum >= target) {
            ubin = b;
            break;
          }
        }
        ke_rel_u = (ubin + 1) * (KE_REL_HIST_MAX / KE_REL_NBINS);
      }
      // total == 0.0 (no masked/skin-tracked atoms sampled any ke_max
      // during Phase A): leave ke_rel_u at -1.0 and retry at the next
      // refresh cycle rather than fitting a threshold from zero data.
    }

    if (ke_rel_u >= 0.0) {
      double count_u_g = ke_rel_count_u, sum_exceed_g = ke_rel_sum_exceed;
      if (comm->nprocs > 1) {
        double local[2] = {ke_rel_count_u, ke_rel_sum_exceed};
        double global[2];
        MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, world);
        count_u_g = global[0];
        sum_exceed_g = global[1];
      }

      if (count_u_g > 0.0 && sum_exceed_g > 0.0) {
        double lambda = count_u_g / sum_exceed_g;
        double x_star = ke_rel_u + (1.0 / lambda) * log((double) ke_rel_margin_blocks * count_u_g);
        x_star = MAX(ke_rel, MIN(100.0, x_star));
        bool first_fit = (ke_rel_last_logged < 0.0);

        if (ke_rel_refine) {
          if (x_star > ke_rel) {
            ke_rel = x_star;
            if (first_fit || fabs(ke_rel - ke_rel_last_logged) > 0.2 * ke_rel_last_logged) {
              log_event("KE_REL_CAL", -1, -1.0, -1.0, -1.0, -1.0, nullptr, ke_rel,
                        fmt::format("refine lambda={} u={}", lambda, ke_rel_u));
              ke_rel_last_logged = ke_rel;
            }
          }
        } else if (first_fit) {
          // freeze (default): one-time fit, then disable all further
          // Phase A/B sampling and re-fitting for the rest of the run.
          ke_rel = x_star;
          log_event("KE_REL_CAL", -1, -1.0, -1.0, -1.0, -1.0, nullptr, ke_rel,
                    fmt::format("freeze lambda={} u={}", lambda, ke_rel_u));
          ke_rel_last_logged = ke_rel;
          ke_rel_warmup_steps = 0;
        }
      }
    }
  }

  // no unresolved mode on ANY rank at all (every flagged mode is either
  // analytic or zero) -- fall back to a floor that keeps dt_c well inside
  // [dt_min, dt_max] rather than dividing by zero.
  if (om_unres <= 0.0) om_unres = c_acc / dt_max;
  if (om_up <= 0.0) om_up = c_acc / dt_max;

  double dt_c = c_acc / om_unres;
  double dt_c_up = c_acc / om_up;    // == dt_c unless per_mode
  if (om_analytic_max > 0.0) {
    dt_c = MIN(dt_c, 2.2 / om_analytic_max);
    dt_c_up = MIN(dt_c_up, 2.2 / om_analytic_max);
  }
  dt_c = MAX(dt_min, MIN(dt_max, dt_c));
  dt_c_up = MAX(dt_min, MIN(dt_max, dt_c_up));

  if (dt_lvl < 0) {
    // first-ever call: initialize the rung directly from dt_c_up, no hysteresis.
    dt_lvl = MAX(0, (int) lround(log(dt_max / dt_c_up) / log(2.0)));
  } else {
    double rung_dt = dt_max / pow(2.0, dt_lvl);
    // dt_floor_until (Attempt 7 Part B, preshrink only): captured once at
    // entry, deliberately NOT re-read after settle_confirm below might
    // release it early this same call -- an early release found this call
    // only takes effect starting the NEXT call, exactly mirroring how a
    // lockout that runs its full natural duration already only unblocks
    // relax-up on the next call that happens to see ntimestep >=
    // dt_floor_until.
    bool locked = (update->ntimestep < dt_floor_until);

    // per_mode: hold a rung while the 99th percentile still meets the bound that ENTERING it
    // required of every mode (dt_c_up > 1.87 * rung/2, i.e. 0.935 * rung); outliers above that
    // are masked (up to c_acc/(0.85*dt)). Holding up to the plain 0.85 shrink bound instead
    // masked the upper edge of the bulk spectrum (114-atom production deck at 10 fs: 10x more
    // Newton guard trips).
    if (dt_c < (per_mode_dt ? 0.935 : 0.85) * rung_dt) {
      dt_lvl = MAX(dt_lvl + 1, (int) ceil(log(dt_max / dt_c) / log(2.0)));
      if (fresh_sample) settle_streak = 0;
    } else {
      // Relax up only if dt_c would ALSO survive the shrink test at the next (2x) rung, which
      // fires below 0.85*(2*rung_dt) = 1.7*rung_dt. The old flat 1.25*rung_dt margin sat inside
      // that band, so any dt_c in (1.25, 1.7)*rung_dt flipped the step between two rungs on
      // every refresh (found on water: dt alternated 1.0/2.0 fs at every refresh). 1.87 =
      // 1.7 plus a 10% margin; dt_c is clamped to dt_max, so the top rung stays reachable.
      bool margin_ok = (dt_c_up > 1.87 * rung_dt);

      // settle_confirm (default 0 = disabled, exactly today's fixed-
      // duration lockout): while a dt_floor_until lockout from
      // end_of_step()'s preshrink response is active, count CONSECUTIVE
      // fresh (never promote_every's stale-cache reuse -- fresh_sample is
      // false there) refresh_every-cadence calls that independently find
      // the same 1.25x margin already required for ordinary relax-up.
      // Only once settle_confirm of these agree in a row -- each one a
      // genuine new global reduction over every flagged atom's
      // just-reprobed curvature, never a single sample reused or taken
      // mid-encounter -- release the lockout early so the unchanged
      // one-rung climb below can resume on its own NEXT call. Any margin
      // miss, or no active lockout at all, resets the streak; a fresh
      // end_of_step() preshrink re-trigger resets it too (see there), so a
      // still-active encounter can never rack up a streak between its own
      // out-of-cadence re-trips.
      if (fresh_sample && settle_confirm > 0 && locked) {
        if (margin_ok) {
          if (++settle_streak >= settle_confirm) {
            log_event("SETTLE_RELEASE", -1, -1.0, -1.0, -1.0, -1.0, nullptr, dt_c,
                      fmt::format("streak={} old_floor={}", settle_streak, dt_floor_until));
            dt_floor_until = update->ntimestep;
            settle_streak = 0;
          }
        } else {
          settle_streak = 0;
        }
      } else if (fresh_sample) {
        settle_streak = 0;
      }

      // dt_floor_until (Attempt 7 Part B, preshrink only): skip relax-up
      // while an end_of_step() emergency shrink's short backoff is still
      // active -- om_unres above is blind to which atom justified that
      // shrink, so without this guard a scheduled refresh right after it
      // could raise dt_lvl back down before the atom has actually cleared
      // the encounter.
      if (margin_ok && dt_lvl > 0 && !locked) dt_lvl--;
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
  // per_mode: mask every mode that would fail update_dt()'s shrink test at this dt
  // (om_mask_thr is in natural units, like om[] and om_split)
  if (eligibility_bonded == 2) om_mask_thr = c_acc / (0.85 * dt_new) / sqrt(force->ftm2v);
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

/* squared contribution of one modal force residual r to solve_center()'s
   convergence/guard residual. newton_res_length off (default): the raw force
   (historical; compared against the LENGTH a_th -- a dimensional mismatch by
   a factor k = mi*om^2, ~1.1 for H/Pd metal but ~800 for water real units).
   On: the Newton step length r/(mi*om^2), matching the Python reference's
   _solve_centers() (lj_longstep_prototype.py), so res, tol and a_th are all
   lengths. See solve_center_force_vs_length_residual_bug memory. */
double FixBAOABTether::newton_res_comp(double r, double mi, double om_m) const
{
  if (!newton_res_length) return r * r;
  double o = MAX(om_m, 1.0e-8);
  double st = r / (mi * o * o);
  return st * st;
}

void FixBAOABTether::solve_center(int i, double mi, double kT, double *cnew)
{
  double **x = atom->x;

  // THROWAWAY DIAGNOSTIC (newton_debug_tag), see header comment.
  bool dbg = (newton_debug_tag >= 0) && (atom->tag[i] == newton_debug_tag);
  if (dbg)
    fprintf(stderr, "[newton_debug tag=%d step=" BIGINT_FORMAT "] om=(%.6g %.6g %.6g)\n",
            (int) atom->tag[i], update->ntimestep, om[i][0], om[i][1], om[i][2]);

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
  // a_th_mass_scale (EXPERIMENTAL, default off): see its header doc comment
  // and water_newton_guard_unit_scale_bug memory for the full derivation
  // and why this is opt-in rather than an unconditional fix. Off (default)
  // reproduces the file's original formula byte-for-byte, the
  // dimensionally-wrong-but-H/Pd-road-tested `sqrt(kT)/om_top` (kT already
  // divided by force->mvv2e, no division by the atom's own mass mi). On:
  // the dimensionally-correct thermal amplitude of the stiffest masked
  // mode, a_th = sqrt(force->boltz*t_target/mi)/om_top (standard 1D-
  // harmonic-oscillator equipartition <x^2> = kB*T/k_spring, with
  // k_spring = om_top^2*mi since om_top is om[i][m] as stored -- not yet
  // the sqrt(force->ftm2v)-converted real angular frequency), expressed
  // via the existing kT as sqrt(kT*force->mvv2e/mi)/om_top.
  double a_th = (om_top > 0.0)
      ? (a_th_mass_scale ? sqrt(kT * force->mvv2e / mi) : sqrt(kT)) / om_top
      : 0.0;
  double tol = solve_tol_rel * a_th;
  double damp = newton_damp;
  double res = 0.0;
  bool newton_diverged = false;

  if (dbg)
    fprintf(stderr, "  msk=(%d %d %d) any_masked=%d a_th=%.6g tol=%.6g q_now=(%.6g %.6g %.6g) "
                     "q_warm=(%.6g %.6g %.6g)\n",
            msk[0], msk[1], msk[2], any_masked, a_th, tol, q_now[0], q_now[1], q_now[2],
            q_warm[0], q_warm[1], q_warm[2]);

  if (any_masked) {
    const int max_backtrack = 6;
    // Backtrack-exhaustion is only a genuine hard failure when the residual
    // is still this many multiples of tol away from converged -- see the
    // "BACKTRACK_EXHAUSTED" comment below. No Python analog (its reactive-
    // damping Newton scheme always accepts a step, so it never hits this
    // code path at all); chosen empirically via newton_debug_tag tracing,
    // which showed benign stalls landing at 1.03-1.06x tol.
    const double newton_stall_tol_mult = 3.0;

    double trial[3];
    for (int k = 0; k < 3; k++) {
      trial[k] = x0_eff[k];
      for (int m = 0; m < 3; m++) trial[k] += evecs[i][3 * m + k] * q[m];
    }
    double fout[3];
    partial_force(i, trial, fout);   // dispatches to full_system_force() internally when force_eval_full
    n_partial_force_calls++;
    if (kspace_center && (i < (int) fks.size() / 3))
      for (int k = 0; k < 3; k++) fout[k] += fks[3 * i + k];
    double r[3] = {0, 0, 0};
    for (int m = 0; m < 3; m++)
      if (msk[m])
        for (int k = 0; k < 3; k++) r[m] += evecs[i][3 * m + k] * fout[k];
    res = 0.0;
    for (int m = 0; m < 3; m++)
      if (msk[m]) res += newton_res_comp(r[m], mi, om[i][m]);
    res = sqrt(res);

    if (dbg)
      fprintf(stderr, "  initial: trial=(%.6g %.6g %.6g) res=%.6g (tol=%.6g, res/tol=%.4g)\n",
              trial[0], trial[1], trial[2], res, tol, res / MAX(tol, 1e-300));

    // Every candidate step below is accepted only if it both lowers the
    // residual AND lands within cutoff of a real neighbor (trial_in_range).
    // Without the range check, a step that overshoots past the last real
    // neighbor reports a trivial fout=0/res=0 "success" -- the false-
    // convergence failure mode that let a divergent solve escape the box
    // entirely on a real H/Pd EAM run (see the solve_center() backtracking
    // plan). Backtracking (shrinking scale) replaces the old scheme's
    // reactive, after-the-fact damp halving, which could still commit an
    // overshooting step before noticing it made things worse.
    for (int iter = 0; iter < newton_iters; iter++) {
      if (trial_in_range(i, trial) && res < tol) break;

      double raw[3] = {0, 0, 0};
      for (int m = 0; m < 3; m++)
        if (msk[m]) raw[m] = r[m] / (mi * MAX(om[i][m], 1.0e-8) * MAX(om[i][m], 1.0e-8));

      if (dbg)
        fprintf(stderr, "  iter=%d q=(%.6g %.6g %.6g) r=(%.6g %.6g %.6g) raw=(%.6g %.6g %.6g) "
                         "damp=%.6g in_range=%d\n",
                iter, q[0], q[1], q[2], r[0], r[1], r[2], raw[0], raw[1], raw[2], damp,
                trial_in_range(i, trial));

      double scale = damp;
      bool accepted = false;
      double q_try[3] = {q[0], q[1], q[2]};
      double trial_try[3], fout_try[3], r_try[3], res_try = 0.0;
      for (int bt = 0; bt < max_backtrack; bt++) {
        for (int m = 0; m < 3; m++) q_try[m] = q[m];
        for (int m = 0; m < 3; m++) {
          if (!msk[m]) continue;
          double om_safe = MAX(om[i][m], 1.0e-8);
          // Same a_th_mass_scale scale issue as a_th/eharm above (found in
          // a systematic sweep of every kT/om-combining formula in this
          // file after those two) -- this per-iteration Newton step cap is
          // a position-space amplitude bound, structurally identical to
          // a_th (sqrt(kT)/om, no /mi). Gated by the same flag for the
          // same reason: untested whether tightening it for metal units
          // is safe for H/Pd without its own multi-seed regression.
          // newton_cap_rel exposes the multiplier (default 8.0, the old
          // hardcoded literal) as a keyword -- negative disables the cap
          // entirely (no finite "neutral" value exists for a clamp).
          double d = scale * raw[m];
          if (newton_cap_rel >= 0.0) {
            double cap = (a_th_mass_scale ? newton_cap_rel * sqrt(kT * force->mvv2e / mi)
                                           : newton_cap_rel * sqrt(kT)) /
                om_safe;
            if (d > cap) d = cap;
            if (d < -cap) d = -cap;
          }
          q_try[m] = q[m] + d;
        }

        for (int k = 0; k < 3; k++) {
          trial_try[k] = x0_eff[k];
          for (int m = 0; m < 3; m++) trial_try[k] += evecs[i][3 * m + k] * q_try[m];
        }
        if (force_eval_full)
          full_system_force(i, trial_try, fout_try);
        else
          partial_force(i, trial_try, fout_try);
        n_partial_force_calls++;
        if (kspace_center && (i < (int) fks.size() / 3))
          for (int k = 0; k < 3; k++) fout_try[k] += fks[3 * i + k];
        r_try[0] = r_try[1] = r_try[2] = 0.0;
        for (int m = 0; m < 3; m++)
          if (msk[m])
            for (int k = 0; k < 3; k++) r_try[m] += evecs[i][3 * m + k] * fout_try[k];
        res_try = 0.0;
        for (int m = 0; m < 3; m++)
          if (msk[m]) res_try += newton_res_comp(r_try[m], mi, om[i][m]);
        res_try = sqrt(res_try);

        bool in_range_try = trial_in_range(i, trial_try);
        if (dbg)
          fprintf(stderr, "    bt=%d scale=%.6g q_try=(%.6g %.6g %.6g) res_try=%.6g "
                           "in_range=%d %s\n",
                  bt, scale, q_try[0], q_try[1], q_try[2], res_try, in_range_try,
                  (in_range_try && res_try <= res) ? "ACCEPT" : "reject");
        if (in_range_try && res_try <= res) {
          accepted = true;
          break;
        }
        scale *= 0.5;
      }

      if (!accepted) {
        // Backtracking exhausted without finding an improving, in-range
        // step. Found via newton_debug_tag tracing (see
        // test_ladder_isolates_newton_divergence_at_t300 memory): this
        // commonly happens with the CURRENT residual already within a few
        // percent of tol -- a benign nonlinear residual floor the local
        // linear Newton model can't quite close, not a genuine divergence.
        // Only flag the hard-failure guard (g_n below) when backtracking
        // gives up with the residual still meaningfully unconverged (more
        // than newton_stall_tol_mult * tol); otherwise accept the current,
        // near-converged q/trial as the final answer for this call, same as
        // a normal tol-satisfied exit, and let the loop end normally below.
        if (res > newton_stall_tol_mult * tol) newton_diverged = true;
        if (dbg)
          fprintf(stderr, "  BACKTRACK_EXHAUSTED at iter=%d res=%.6g (tol=%.6g, res/tol=%.4g) "
                           "-> %s\n",
                  iter, res, tol, res / MAX(tol, 1e-300),
                  newton_diverged ? "NEWTON_DIVERGED" : "accepted as near-converged");
        break;
      }

      for (int m = 0; m < 3; m++)
        if (msk[m]) q[m] = q_try[m];
      damp = scale;
      for (int k = 0; k < 3; k++) trial[k] = trial_try[k];
      for (int m = 0; m < 3; m++) r[m] = r_try[m];
      res = res_try;
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
  // adaptive controller. newton_tol_rel/flip_rel/energy_guard_rel are now
  // user-settable keywords (see fix_baoab_tether.h and the "Parameterize
  // solve_center()'s guard-threshold constants" plan) expressing each
  // threshold directly as its effective multiple of a_th (or of kT for
  // energy_guard_rel) -- default values (5.0/14.0/25.0) reproduce the
  // original hardcoded formulas exactly (newton_tol_rel's default collapses
  // the old reference-ported "20.0*0.25" indirection into one number;
  // flip_rel/energy_guard_rel were already that scale, just unexposed).
  // Negative disables the corresponding guard (no finite "neutral" value
  // exists for an inequality threshold, see a_th_mass_scale's analogous
  // negative-sentinel precedent at newton_cap_rel above).
  if (any_masked) {
    bool g_n = newton_diverged || ((newton_tol_rel >= 0.0) && (res > newton_tol_rel * a_th));
    if (g_n) n_guard_newton++;

    double flip = 0.0;
    for (int m = 0; m < 3; m++)
      if (msk[m]) flip += (q[m] - q_warm[m]) * (q[m] - q_warm[m]);
    flip = sqrt(flip);
    bool g_f = (flip_rel >= 0.0) && (a_th > 0.0) && (flip > flip_rel * a_th);
    if (g_f) n_guard_flip++;

    // Over-excitation energy: harmonic energy of the ATOM's displacement from its clamped
    // center (q_now - q in the masked modes), exactly lj_longstep_prototype.py's
    // harmE = 0.5*om_eff^2*|to_modal(xw - c)|^2. This used to be 0.5*mi*om^2*q^2 with q the
    // CENTER's offset from the refresh position x0: that measures how far the clamped
    // equilibrium has drifted since the last refresh, which is ~0 for a caged rattler (H/Pd)
    // but grows ballistically for any atom whose cage moves with it (a water molecule
    // translating/rotating carries its O-H axis along), tripping the guard on ordinary motion
    // (water O-H: eharm/kT 0.1 -> 25+ within 10 steps, ~165 spurious demotions per 1500 steps).
    double eharm = 0.0;
    for (int m = 0; m < 3; m++)
      if (msk[m]) {
        double dq_osc = q_now[m] - q[m];
        eharm += 0.5 * mi * om[i][m] * om[i][m] * dq_osc * dq_osc;
      }
    // a_th_mass_scale (see a_th's doc comment above): eharm =
    // 0.5*mi*om_native^2*q^2 is a RAW native-energy quantity (no leftover
    // mvv2e factor), so comparing it against energy_guard_rel*kT (kT
    // already divided by force->mvv2e off a_th_mass_scale) has the
    // identical scale mismatch as a_th's -- gated by the SAME flag since
    // it's the same underlying bug, so a system either gets both
    // corrections or neither.
    bool g_e = (energy_guard_rel >= 0.0) &&
        (eharm > energy_guard_rel * kT * (a_th_mass_scale ? force->mvv2e : 1.0));
    if (g_e) n_guard_energy++;

    if (dbg)
      fprintf(stderr, "  FINAL: res=%.6g g_n=%d(diverged=%d) g_f=%d(flip-rel=%.4g) "
                       "g_e=%d(eharm/kT=%.4g)\n",
              res, g_n, newton_diverged, g_f, (a_th > 0.0) ? flip / a_th : -1.0, g_e,
              eharm / (kT * (a_th_mass_scale ? force->mvv2e : 1.0)));

    // demotion action (adapt=yes only): exponential backoff capped at 2000
    // steps, direct port of the reference's
    // "3*K*2**min(demote_count-1,6)" -- demote_count[i] here is still the
    // PRE-increment value, so MIN(demote_count[i], 6) is exactly
    // MIN(demote_count-1, 6) after demote_atom()'s increment below.
    if (adapt && (g_n || g_f || g_e)) {
      int expo = MIN(demote_count[i], 6);
      bigint mult = ((bigint) 1) << expo;
      bigint backoff = MIN((bigint) refresh_every * 3 * mult, (bigint) 2000);
      demote_atom(i, backoff,
                  g_e ? (g_n ? (g_f ? "solve_center_gn_gf_ge" : "solve_center_gn_ge")
                             : (g_f ? "solve_center_gf_ge" : "solve_center_ge"))
                      : (g_n ? (g_f ? "solve_center_gn_gf" : "solve_center_gn")
                             : "solve_center_gf"));
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

/* ----------------------------------------------------------------------
   true if atom j is CURRENTLY (or as of its last refresh) a stiff/rattler
   atom, and so should contribute to a Newton-residual evaluation via its
   clamped center c[j] rather than its live x[j] -- the actual intent
   behind every "flagged neighbor" clamp substitution below (see
   trial_in_range()/partial_force()'s doc comments for why the
   substitution exists at all: suppressing resonant rattler-rattler
   coupling in the Newton solve). Deliberately NOT `mask[j] & groupbit`:
   that bit meant "j is a rattler" back when this fix's own group WAS
   exactly the rattler population (two-fix era), but on a single
   whole-system group (e.g. "all") it is true for EVERY atom, including
   an ordinary cage atom undergoing perfectly normal thermal motion --
   clamping those to a stale, refresh-cadence-frozen c[j] instead of their
   real position is not a resonance-suppression measure for them, it is
   just wrong, and was never caught because it doesn't crash or show up in
   the standard diagnostic suite (see local_partial_force_two_real_bugs_found
   memory, confirmed directly via a small controlled reproducer: solve_center()
   Newton-residual calls showed 50-220% relative force error, even at zero
   atom-i displacement, traced to exactly this). Same om-based eligibility
   test as refresh_one_atom()'s nk/Jtag loop (duplicated rather than shared,
   same reasoning as that site: trivially removable, and the two loops
   want OPPOSITE senses of "eligible" -- that loop excludes eligible
   neighbors from the back-reaction, this one clamps them).
------------------------------------------------------------------------- */

bool FixBAOABTether::neighbor_clamp_eligible(int j) const
{
  double om_max_j = MAX(om[j][0], MAX(om[j][1], om[j][2]));
  if (adapt) {
    double om_min_j = MIN(om[j][0], MIN(om[j][1], om[j][2]));
    return (om_max_j > om_split) && (bonded_ok(j) || (om_min_j > 0.6 * om_split));
  }
  return (om_max_j * sqrt(force->ftm2v) * update->dt) > theta;
}

/* ----------------------------------------------------------------------
   true if xtrial is within cutoff of at least one of atom i's real
   neighbors (same clamped-center substitution and neighbor list as
   partial_force()'s fallback loop below). Used by solve_center()'s
   backtracking line search to reject a candidate step that overshoots
   past every neighbor into empty space -- such a step would otherwise
   report a trivial fout=0/res=0 "converged" from partial_force(), which
   is a false convergence, not a real equilibrium.
------------------------------------------------------------------------- */

bool FixBAOABTether::trial_in_range(int i, const double *xtrial) const
{
  if (!list) return false;

  double **x = atom->x;
  int *type = atom->type;
  int itype = type[i];
  double **cutsq = force->pair->cutsq;

  int *jlist = list->firstneigh[i];
  int jnum = list->numneigh[i];

  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj] & NEIGHMASK;
    int jtype = type[j];
    const double *xj = neighbor_clamp_eligible(j) ? c[j] : x[j];

    double delx = xtrial[0] - xj[0];
    double dely = xtrial[1] - xj[1];
    double delz = xtrial[2] - xj[2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq < cutsq[itype][jtype]) return true;
  }
  return false;
}

bool FixBAOABTether::partial_force(int i, const double *xtrial, double *fout, double *fneigh,
                                    bool live_neighbor_positions, tagint *bond_tag,
                                    double *bond_fneigh, int *bond_jnum_out)
{
  fout[0] = fout[1] = fout[2] = 0.0;
  if (bond_jnum_out) *bond_jnum_out = 0;
  if (!list) return false;

  double **x = atom->x;
  int *type = atom->type;
  int itype = type[i];
  double **cutsq = force->pair->cutsq;

  int *jlist = list->firstneigh[i];
  int jnum = list->numneigh[i];

  // force_eval full: every caller of partial_force() (not just
  // solve_center()'s two sites) now gets a genuine whole-system force
  // evaluation instead of the local kernel below -- see full_system_force()'s
  // doc comment and force_eval_full's doc comment in the header. Note this
  // silently drops the live_neighbor_positions c[j]-substitution behavior
  // (full_system_force() always uses real positions, a documented scope
  // limitation of this validation mode). fout is already bond/angle-complete
  // here (full_system_force() -> recompute_forces_local() calls force->bond/
  // force->angle unconditionally) but bond_tag/bond_fneigh are left empty --
  // no per-partner breakdown is available from a whole-system recompute, so
  // the bonded back-reaction is silently absent under force_eval_full, the
  // same documented scope limitation as the c[j] substitution above.
  if (force_eval_full) return full_system_force(i, xtrial, fout, fneigh, jnum, jlist);

  if (use_local_partial_force) {
    // Mirror the clamped-center substitution used by the single()-based
    // fallback below (real x[j] -> c[j] for a CURRENTLY stiff/rattler-
    // eligible atom, when !live_neighbor_positions): local_partial_force()
    // applies this itself, internally, via the clamp_mask/clamp_c pair,
    // indexed by atom index (not `mask[j] & groupbit` -- see
    // neighbor_clamp_eligible()'s doc comment for why that would be wrong
    // on a single whole-system group) so a multi-hop implementation (e.g.
    // PairSymmetrixMACE) can apply it to any atom it encounters, not just
    // atom i's direct jlist neighbors. Sized nlocal+nghost and built fresh
    // each call -- a real but modest O(N) cost on top of
    // local_partial_force()'s own, only paid for Newton-residual calls.
    // c[] is kept forward-comm'd/PBC-shifted to match x[]'s image (see
    // pack_forward_comm()/unpack_forward_comm()), so it is safe to hand to
    // the pair style for ghost indices too.
    // std::vector<bool> is bit-packed and has no .data(); use a plain
    // heap buffer so this can be handed to the Pair interface as bool*.
    std::unique_ptr<bool[]> clamp_mask;
    if (!live_neighbor_positions) {
      int nall = atom->nlocal + atom->nghost;
      clamp_mask = std::make_unique<bool[]>(nall);
      for (int k = 0; k < nall; k++) clamp_mask[k] = neighbor_clamp_eligible(k);
    }
    const bool *clamp_mask_ptr = clamp_mask.get();
    double *const *clamp_c = live_neighbor_positions ? nullptr : (double *const *) c;
    bool ok = force->pair->local_partial_force(i, xtrial, jnum, jlist, fout, fneigh,
                                                clamp_mask_ptr, clamp_c);

    // THROWAWAY DIAGNOSTIC (pforce_debug_tag), see header comment: side-
    // channel comparison against full_system_force() for the identical
    // (i, xtrial), without altering which result (fout above) is actually
    // used by the caller. IMPORTANT: full_system_force() recomputes and
    // leaves the pair style's GLOBAL rho[]/fp[] caches in the TRIAL
    // configuration's state -- it only restores f[] (see its own doc
    // comment/implementation), not rho[]/fp[]. That is harmless under
    // normal force_eval_full operation (every call there is a fresh
    // full_system_force() call too, so nothing ever reads a stale
    // in-between cache), but calling it here, as a side channel while
    // use_local_partial_force is the ACTIVE path, would otherwise corrupt
    // the very rho[j] cache local_partial_force()'s delta-correction
    // depends on for every subsequent call this step -- silently
    // contaminating the "local" results this diagnostic is supposed to be
    // comparing against, and compounding across repeated probe/Newton
    // calls. Undo that corruption immediately with one extra no-op
    // full_system_force() call (xtrial == the real x[i], captured before
    // the comparison call moved it) before returning -- 2x the diagnostic's
    // own O(N) cost, never incurred unless pforce_debug_tag is set.
    if (pforce_debug_tag >= 0 && atom->tag[i] == pforce_debug_tag) {
      double xreal_now[3] = {x[i][0], x[i][1], x[i][2]};
      double fout_full[3];
      full_system_force(i, xtrial, fout_full, nullptr, 0, nullptr);
      // bonded terms are added to fout only AFTER this block (below), so compare the
      // pair-only local force PLUS the bonded kernel against the complete full force
      double fout_tot[3] = {fout[0], fout[1], fout[2]};
      partial_force_add_bonded(i, xtrial, fout_tot, live_neighbor_positions, nullptr, nullptr, nullptr);
      double dmag = sqrt((fout_tot[0]-fout_full[0])*(fout_tot[0]-fout_full[0]) +
                          (fout_tot[1]-fout_full[1])*(fout_tot[1]-fout_full[1]) +
                          (fout_tot[2]-fout_full[2])*(fout_tot[2]-fout_full[2]));
      double fmag = sqrt(fout_full[0]*fout_full[0] + fout_full[1]*fout_full[1] +
                          fout_full[2]*fout_full[2]);
      double disp = sqrt((xtrial[0]-xreal_now[0])*(xtrial[0]-xreal_now[0]) +
                          (xtrial[1]-xreal_now[1])*(xtrial[1]-xreal_now[1]) +
                          (xtrial[2]-xreal_now[2])*(xtrial[2]-xreal_now[2]));
      // TEMPORARY forensic check (user's hypothesis): is any neighbor BOTH
      // currently clamp-eligible AND within force cutoff of xtrial? If so,
      // local correctly clamps it (c[j]) while full_system_force() always
      // uses its real x[j] (documented scope limitation, not a bug) -- the
      // two are EXPECTED to disagree there, not a sign of a local defect.
      int n_eligible_in_range = 0;
      double min_eligible_r = -1.0;
      if (!live_neighbor_positions) {
        double **cutsq = force->pair->cutsq;
        int *type = atom->type;
        int itype = type[i];
        for (int jj = 0; jj < jnum; jj++) {
          int j = jlist[jj] & NEIGHMASK;
          if (!neighbor_clamp_eligible(j)) continue;
          double delx = xtrial[0]-x[j][0], dely = xtrial[1]-x[j][1], delz = xtrial[2]-x[j][2];
          double rsq = delx*delx+dely*dely+delz*delz;
          if (rsq < cutsq[itype][type[j]]) {
            n_eligible_in_range++;
            double r = sqrt(rsq);
            if (min_eligible_r < 0.0 || r < min_eligible_r) min_eligible_r = r;
          }
        }
      }

      utils::logmesg(lmp, "PFORCE_DEBUG step={} tag={} live={} jnum={} disp={:.6g} "
                     "n_elig_in_range={} min_elig_r={:.6g} "
                     "fout_local=({:.6g},{:.6g},{:.6g}) fout_full=({:.6g},{:.6g},{:.6g}) "
                     "|delta|={:.6g} |full|={:.6g} rel={:.6g}\n",
                     update->ntimestep, pforce_debug_tag, live_neighbor_positions, jnum, disp,
                     n_eligible_in_range, min_eligible_r,
                     fout_tot[0], fout_tot[1], fout_tot[2], fout_full[0], fout_full[1], fout_full[2],
                     dmag, fmag, fmag > 0.0 ? dmag / fmag : 0.0);

      double fout_restore_dummy[3];
      full_system_force(i, xreal_now, fout_restore_dummy, nullptr, 0, nullptr);
    }
    partial_force_add_bonded(i, xtrial, fout, live_neighbor_positions, bond_tag, bond_fneigh,
                              bond_jnum_out);
    return ok;
  }

  for (int jj = 0; jj < jnum; jj++) {
    int j = jlist[jj] & NEIGHMASK;
    int jtype = type[j];

    // A neighbor that is CURRENTLY stiff/rattler-eligible must contribute
    // via its own current CENTER, not its instantaneous real position --
    // otherwise atom i's clamped equilibrium responds directly to atom j's
    // fast real-time oscillation phase, a resonant parametric-coupling
    // channel between adjacent rattlers. This mirrors
    // lj_longstep_prototype.py's _solve_centers(), which places ALL
    // flagged atoms at their (jointly converging) trial centers before
    // every force evaluation, never at their real xw positions; it is the
    // analogous fix for the Newton solve itself, complementing (not
    // duplicating) the rattler-rattler Jblk exclusion in refresh_blocks(),
    // which only concerns the back-reaction operator. See
    // neighbor_clamp_eligible()'s doc comment for why this is an om-based
    // eligibility test, not `mask[j] & groupbit`.
    const double *xj = (!live_neighbor_positions && neighbor_clamp_eligible(j)) ? c[j] : x[j];

    double delx = xtrial[0] - xj[0];
    double dely = xtrial[1] - xj[1];
    double delz = xtrial[2] - xj[2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq >= cutsq[itype][jtype]) {
      if (fneigh) fneigh[3 * jj] = fneigh[3 * jj + 1] = fneigh[3 * jj + 2] = 0.0;
      continue;
    }

    // Honor the special-bond bits the neighbor list carries on jlist[jj]: with a kspace style
    // LAMMPS keeps excluded (bonded 1-2/1-3/1-4) pairs in the list so the pair style can apply
    // its excluded-pair correction (factor 0 => subtract the reciprocal-space part). Hard-coding
    // factors of 1.0 here evaluated those pairs as full Coulomb, a spurious ~100 kcal/mol/A
    // force on a water O-H pair in the Newton solve and FD curvature probe (found 2026-11-05;
    // force_eval full, which calls the real pair->compute(), was stable). For pair styles
    // without special neighbors the bits are 0 => factors 1.0, identical to before.
    int sb = (jlist[jj] >> SBBITS) & 3;
    double fpair = 0.0;
    force->pair->single(i, j, itype, jtype, rsq, force->special_coul[sb], force->special_lj[sb],
                        fpair);
    fout[0] += delx * fpair;
    fout[1] += dely * fpair;
    fout[2] += delz * fpair;
    if (fneigh) {
      fneigh[3 * jj + 0] = -delx * fpair;
      fneigh[3 * jj + 1] = -dely * fpair;
      fneigh[3 * jj + 2] = -delz * fpair;
    }
  }
  partial_force_add_bonded(i, xtrial, fout, live_neighbor_positions, bond_tag, bond_fneigh,
                            bond_jnum_out);
  return true;
}

/* ----------------------------------------------------------------------
   Bonded (bond/angle-style) contribution to partial_force()'s per-atom
   local force kernel: a stiff bond's partner (e.g. water's O-H) is
   typically EXCLUDED from the pairwise neighbor list by special_bonds, so
   it is invisible to the pairwise logic above -- yet it is often the
   dominant stiff mode partial_force()/solve_center()'s Newton solve needs
   to see (design doc: O-H stretch ~3700 cm^-1, the stiffest mode in
   water). The H-O-H bend (~1600 cm^-1) is softer but NOT negligible for
   the Newton solve specifically: omitting it leaves solve_center()
   searching for a force balance against an incomplete force law (bond
   only), while the real per-step force (recompute_forces_local(), via
   force_moll()) is bond+angle complete -- confirmed empirically on this
   project's water workload: bond-only produced a large, persistent
   Newton-guard rate (n_guard_newton/n_guard_flip climbing continuously),
   which is the expected symptom of chasing a root that doesn't exist for
   the (incomplete) force law being solved. A no-op (fout unchanged, no
   slots written) whenever force->bond AND force->angle are both unset, so
   this has zero effect on the existing EAM/MACE pair-only systems.

   Uses the SAME tag->local-index resolution LAMMPS's own topology-list
   builders use (ntopo_bond_all.cpp: atom->map() then
   domain->closest_image()) rather than neighbor->bondlist/anglelist,
   which are deduplicated, non-atom-indexed flat lists -- the wrong tool
   for "given atom i, find its bonded partners now". This requires every
   bond/angle to be stored symmetrically on ALL its member atoms, i.e.
   newton_bond off (checked in init(); see water_bonded_kspace_scoping
   memory).

   Applies the identical clamped-center substitution as the pairwise loop
   above (neighbor_clamp_eligible()) so a bonded partner that is itself
   currently flagged stiff doesn't inject its own fast oscillation phase
   into atom i's Newton solve -- the same resonant-coupling concern that
   substitution exists to suppress for ordinary pairwise neighbors.

   Angle support is AngleHarmonic-only (Angle::single_force() returns
   false, silently skipped, for any other angle style) -- a known, scoped
   limitation, not a general angle-style framework.

   bond_tag/bond_fneigh/bond_jnum_out are optional (nullptr skips writing
   them, matching fneigh's existing convention): fout always gets the
   bond/angle contribution added regardless, but only refresh_one_atom()
   (which needs each partner's own force response for the Jtag/Jblk
   back-reaction) passes non-null buffers; solve_center() just wants the
   correct net fout.
------------------------------------------------------------------------- */

void FixBAOABTether::partial_force_add_bonded(int i, const double *xtrial, double *fout,
                                               bool live_neighbor_positions, tagint *bond_tag,
                                               double *bond_fneigh, int *bond_jnum_out) const
{
  if (!force->bond && !force->angle) return;

  double **x = atom->x;
  int nk = 0;

  auto clamped = [&](int j) -> const double * {
    return (!live_neighbor_positions && neighbor_clamp_eligible(j)) ? c[j] : x[j];
  };

  if (force->bond) {
    int nb = atom->num_bond[i];
    for (int k = 0; k < nb; k++) {
      if (nk >= BOND_JMAX)
        error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded local-force "
                           "partner slots -- raise BOND_JMAX and recompile", BOND_JMAX);
      int j = atom->map(atom->bond_atom[i][k]);
      if (j < 0) continue;    // lost bond partner -- fails safe, same convention as n_ghost_miss
      j = domain->closest_image(i, j);

      const double *xj = clamped(j);

      double delx = xtrial[0] - xj[0];
      double dely = xtrial[1] - xj[1];
      double delz = xtrial[2] - xj[2];
      double rsq = delx * delx + dely * dely + delz * delz;

      double fforce = 0.0;
      force->bond->single(atom->bond_type[i][k], rsq, i, j, fforce);
      fout[0] += delx * fforce;
      fout[1] += dely * fforce;
      fout[2] += delz * fforce;

      if (bond_tag && bond_fneigh) {
        bond_tag[nk] = atom->tag[j];
        bond_fneigh[3 * nk + 0] = -delx * fforce;
        bond_fneigh[3 * nk + 1] = -dely * fforce;
        bond_fneigh[3 * nk + 2] = -delz * fforce;
        nk++;
      }
    }
  }

  if (force->angle) {
    int na = atom->num_angle[i];
    tagint my_tag = atom->tag[i];
    for (int k = 0; k < na; k++) {
      tagint t1 = atom->angle_atom1[i][k];
      tagint t2 = atom->angle_atom2[i][k];
      tagint t3 = atom->angle_atom3[i][k];
      int role = (t1 == my_tag) ? 1 : (t2 == my_tag) ? 2 : 3;

      int j1 = (role == 1) ? i : atom->map(t1);
      int j2 = (role == 2) ? i : atom->map(t2);
      int j3 = (role == 3) ? i : atom->map(t3);
      if (j1 < 0 || j2 < 0 || j3 < 0) continue;
      if (role != 1) j1 = domain->closest_image(i, j1);
      if (role != 2) j2 = domain->closest_image(i, j2);
      if (role != 3) j3 = domain->closest_image(i, j3);

      const double *x1 = (role == 1) ? xtrial : clamped(j1);
      const double *x2 = (role == 2) ? xtrial : clamped(j2);
      const double *x3 = (role == 3) ? xtrial : clamped(j3);

      double f1[3], f3[3];
      if (!force->angle->single_force(atom->angle_type[i][k], x1, x2, x3, f1, f3)) continue;
      double f2[3] = {-(f1[0] + f3[0]), -(f1[1] + f3[1]), -(f1[2] + f3[2])};

      const double *fi = (role == 1) ? f1 : (role == 2) ? f2 : f3;
      fout[0] += fi[0];
      fout[1] += fi[1];
      fout[2] += fi[2];

      if (bond_tag && bond_fneigh) {
        if (role != 1) {
          if (nk >= BOND_JMAX)
            error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded "
                               "local-force partner slots -- raise BOND_JMAX and recompile",
                       BOND_JMAX);
          bond_tag[nk] = atom->tag[j1];
          bond_fneigh[3 * nk + 0] = f1[0];
          bond_fneigh[3 * nk + 1] = f1[1];
          bond_fneigh[3 * nk + 2] = f1[2];
          nk++;
        }
        if (role != 2) {
          if (nk >= BOND_JMAX)
            error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded "
                               "local-force partner slots -- raise BOND_JMAX and recompile",
                       BOND_JMAX);
          bond_tag[nk] = atom->tag[j2];
          bond_fneigh[3 * nk + 0] = f2[0];
          bond_fneigh[3 * nk + 1] = f2[1];
          bond_fneigh[3 * nk + 2] = f2[2];
          nk++;
        }
        if (role != 3) {
          if (nk >= BOND_JMAX)
            error->one(FLERR, "Fix baoab/tether: atom exceeds BOND_JMAX ({}) bonded "
                               "local-force partner slots -- raise BOND_JMAX and recompile",
                       BOND_JMAX);
          bond_tag[nk] = atom->tag[j3];
          bond_fneigh[3 * nk + 0] = f3[0];
          bond_fneigh[3 * nk + 1] = f3[1];
          bond_fneigh[3 * nk + 2] = f3[2];
          nk++;
        }
      }
    }
  }
  if (bond_jnum_out) *bond_jnum_out = nk;
}

/* ----------------------------------------------------------------------
   force_eval full validation mode: an exact whole-system force on atom i
   (and, when fneigh/jnum/jlist are supplied, on each of its listed
   neighbors too) if atom i were placed at xtrial. partial_force() routes
   EVERY one of its callers here -- solve_center()'s Newton residual,
   refresh_blocks()/refresh_one_atom()'s FD-Hessian curvature probe (both
   the atom-i diagonal block and the Hc_j/Jblk cross-derivative columns),
   sub_step_free_atom()'s exile sub-stepping, diag_check_staleness()'s
   per-step diagnostic -- whenever force_eval_full is set (see
   partial_force()'s own dispatch at its top). Unlike partial_force(),
   this does NOT substitute a flagged neighbor's clamped center c[j] for
   its real position -- every other atom (rattler or not) is evaluated at
   its actual current x[j]. That is a deliberate, known scope limitation,
   not an oversight: it reintroduces the resonant rattler-rattler coupling
   partial_force()'s c[j] substitution exists to suppress (see that
   comment above), so this mode is only a clean validation of the
   embedding-truncation hypothesis for configurations with at most one
   flagged atom in range of any other (e.g. the dilute single-H-
   interstitial H/Pd runs it was built to test) -- not a general
   substitute for the local kernel.

   Single-process only (guarded in the constructor): moving xtrial into
   atom->x[i] and recomputing requires atom i's own ghost images to be
   refreshed via comm->forward_comm() for correctness under periodic
   boundaries, but no cross-rank communication of the trial position
   itself is attempted here. Reuses recompute_forces_local() (the same
   whole-system pair/bond/.../kspace recompute force_moll() already uses
   for its own clamped-configuration force field) rather than calling
   force->pair->compute() directly, so every force contribution (not just
   the pair style) is included -- the same completeness Python's
   lj_longstep_prototype.py/lammps_hpd_phase0.py get "for free" from
   calling the real engine's full energy_forces() for every FD probe.
   O(N) per call -- expected and acceptable for a validation mode, not
   something to wire into a hot path (see force_eval_full's doc comment
   in the header for which call sites this makes very slow).

   LATENT CONSTRAINT: only f[] (and atom i's position) are saved/restored
   around the trial recompute -- the pair style's own GLOBAL rho[]/fp[]
   caches are deliberately left in the TRIAL configuration's state, not
   restored. Harmless under force_eval_full's actual (sole, exclusive) use
   -- every partial_force() call routes here when it's set, so every call
   is its own fresh, self-consistent full recompute and nothing ever reads
   a stale in-between rho[]/fp[] value. But this function must NEVER be
   called as a side comparison while use_local_partial_force's path is the
   one actually in effect without separately restoring rho[]/fp[] first
   (e.g. a second full_system_force() call at the real, unperturbed
   configuration) -- local_partial_force()'s rho_j_trial delta-correction
   reads rho[j] directly, and would otherwise silently use a corrupted
   baseline for every subsequent call this step. pforce_debug_tag's
   comparison code does this restore explicitly; see its doc comment.
------------------------------------------------------------------------- */

bool FixBAOABTether::full_system_force(int i, const double *xtrial, double *fout,
                                        double *fneigh, int jnum, const int *jlist)
{
  double **x = atom->x;
  double **f = atom->f;
  int ntotal = atom->nlocal + atom->nghost;

  std::vector<double> fsave(3 * ntotal);
  if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));

  double xreal[3] = {x[i][0], x[i][1], x[i][2]};
  x[i][0] = xtrial[0];
  x[i][1] = xtrial[1];
  x[i][2] = xtrial[2];

  comm->forward_comm();
  recompute_forces_local();

  fout[0] = f[i][0];
  fout[1] = f[i][1];
  fout[2] = f[i][2];

  // read back each listed neighbor's own total force at this trial
  // configuration -- the whole-system analog of partial_force()'s fneigh
  // output, needed by refresh_one_atom()'s Hc_j/Jblk cross-derivative
  // columns. f[j] here already reflects every force contribution on j
  // (not just the i-j pair), which is the correct quantity for a
  // genuine dF_j/dx_i finite difference -- recompute_forces_local() was
  // just run for the whole local domain, not atom i alone.
  if (fneigh)
    for (int jj = 0; jj < jnum; jj++) {
      int j = jlist[jj] & NEIGHMASK;
      fneigh[3 * jj + 0] = f[j][0];
      fneigh[3 * jj + 1] = f[j][1];
      fneigh[3 * jj + 2] = f[j][2];
    }

  x[i][0] = xreal[0];
  x[i][1] = xreal[1];
  x[i][2] = xreal[2];
  comm->forward_comm();

  if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));

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

// Extracted from force_moll() (pure refactor, no behavior change there) so
// the D.14 teacher/student composition (compute_force_delta()) can apply
// the IDENTICAL modal-mask-zeroing + sparse back-reaction filter to the
// teacher's raw force that force_moll() already applies to the student's.
// Always operates on atom->f directly, for every currently-flagged atom:
// not a generic "filter an arbitrary per-atom array" helper -- the
// comm->reverse_comm() call below is hardwired to atom->f by LAMMPS itself
// (see comm.h), so a caller-supplied buffer would silently NOT get the
// back-reaction's ghost contributions shipped home. Callers that need to
// filter something other than the production force (e.g. the teacher's raw
// compute() output) must first PLACE that result into atom->f themselves
// (zeroing/saving/restoring whatever was there before, exactly as
// force_moll() itself and recompute_forces_local() already do for their own
// out-of-band recomputes) before calling this.
void FixBAOABTether::apply_modal_filter_and_backreaction()
{
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  std::vector<int> flagged;
  flagged.reserve(nlocal);
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) flagged.push_back(i);

  // stage-4: the caller's own recompute (recompute_forces_local() or an
  // equivalent out-of-band compute()) already sent home every ghost's raw
  // pair-force contribution via its own reverse_comm -- but
  // AtomVec::unpack_reverse() only adds into the OWNER's f[], it never
  // clears the GHOST's own f[] slot afterward. Zero the ghost region now so
  // the back-reaction loop below writes into a clean slate; the second
  // comm->reverse_comm() after that loop then sends home exactly the NEW
  // back-reaction contribution, with no double-counting of the
  // already-reconciled raw pair force.
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
}

/* ---------------------------------------------------------------------- */

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
  apply_modal_filter_and_backreaction();

  for (size_t k = 0; k < flagged.size(); k++) {
    int i = flagged[k];
    x[i][0] = xsave[3 * k + 0];
    x[i][1] = xsave[3 * k + 1];
    x[i][2] = xsave[3 * k + 2];
  }
  comm->forward_comm();

  // The recompute_forces_local() call above ran at the CLAMPED geometry,
  // so the pair style's global rho[]/fp[] caches are now left reflecting
  // "every flagged atom sitting at its clamped center" -- not the true,
  // just-restored real positions. f[i] (this function's actual output:
  // masked-mode-zeroed force plus the Jblk back-reaction) is correct and
  // must be preserved exactly as computed above, but every OTHER caller
  // of local_partial_force() for the rest of this step (sub_step_free_atom(),
  // kinetic_guard_and_quench()'s instantaneous re-tether, exile handling,
  // diag_check_staleness(), and refresh_one_atom() if called again later
  // via a mid-step forced refresh) would otherwise silently read this
  // stale clamped-state rho[j]/fp[j] until the step's own ordinary
  // pair->compute() eventually runs -- confirmed as a real, large (often
  // order-unity relative force error) effect via force_eval_local_regression's
  // pforce_debug_tag instrumentation, not merely a hypothetical risk. Save
  // f[] (the output above), do one more recompute at the now-real
  // positions to refresh rho[]/fp[] correctly, then restore f[] -- same
  // save/recompute/restore idiom full_system_force() already uses for its
  // own trial evaluations. Skipped entirely under force_eval_full: every
  // local_partial_force()-based consumer is unreachable there (partial_force()
  // routes everything through full_system_force() instead, which always
  // does its own fresh recompute and never reads this cache), so the extra
  // O(N) pass would be pure overhead with no behavioral effect.
  if (!force_eval_full) {
    int ntotal = atom->nlocal + atom->nghost;
    std::vector<double> fsave(3 * ntotal);
    if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));
    recompute_forces_local();
    if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));
  }
}

/* ----------------------------------------------------------------------
   D.14 teacher/student composition (EXPERIMENTAL, teacher_every > 0 only):
   called once per outer cycle CLOSE, from final_integrate(), immediately
   after this step's own force_moll() call -- at that point atom->f holds
   the FILTERED STUDENT force for every flagged atom (what final_integrate()'s
   own closing B-kick loop, right after this call, must still see
   byte-identical). Computes f_delta[i] = filtered-teacher - filtered-student
   at the SAME clamped geometry, mirroring the Python reference's
   _force_delta() (lj_multifidelity_experiment.py): since
   apply_modal_filter_and_backreaction() is linear, filter(Ft) - filter(Fs)
   equals filter(Ft - Fs) -- re-filtering the teacher's raw force separately
   and subtracting the already-filtered student force (snapshotted below)
   avoids needing a second communication primitive for an arbitrary non-f[]
   buffer (reverse_comm()/forward_comm() are hardwired to atom->f/atom->x).

   Snapshots and restores the WHOLE local+ghost f[] array, not just flagged
   atoms: teacher_pair->compute(0,0) is a genuine whole-system call (every
   atom within its cutoff, not only flagged ones), and
   apply_modal_filter_and_backreaction()'s own back-reaction loop can touch
   arbitrary neighbors of a flagged atom too -- every one of those must come
   back exactly as force_moll() left it, same idiom as force_moll()'s own
   save/recompute/restore tail just above. Does NOT need an analogous
   rho[]/fp[]-style cache restore for teacher_pair itself: nothing else this
   step calls teacher_pair again (unlike force->pair, which
   local_partial_force() callers read all over this file), and force->pair's
   own caches are untouched here (only teacher_pair->compute() runs).
------------------------------------------------------------------------- */

void FixBAOABTether::compute_force_delta()
{
  const double t_start = platform::walltime();

#ifdef LMP_KOKKOS
  // KOKKOS (-k on): this fix works on the host arrays, while a KOKKOS
  // teacher_pair (and device-side communication) reads the device copies.
  // Every host write below that a device consumer must see is therefore
  // claimed with modified(Host, ...) before that consumer runs. Anything
  // claimed on the device is copied straight back to the host because
  // auto_sync is on -- ModifyKokkos already sets it around this
  // (non-KOKKOS) fix's callbacks; it is set here too so this function does
  // not depend on its caller -- and the sync(Host, ...) calls below make
  // that explicit. Without the claims the device would silently keep the
  // unclamped positions and a stale force array.
  AtomKokkos *atomKK = lmp->kokkos ? dynamic_cast<AtomKokkos *>(atom) : nullptr;
  const int prev_auto_sync = lmp->kokkos ? lmp->kokkos->auto_sync : 0;
  if (lmp->kokkos) lmp->kokkos->auto_sync = 1;
#endif

  double **x = atom->x;
  double **f = atom->f;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  int ntotal = nlocal + atom->nghost;

  std::vector<int> flagged;
  flagged.reserve(nlocal);
  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) flagged.push_back(i);

  std::vector<double> fsave(3 * ntotal);
  if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));

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
#ifdef LMP_KOKKOS
  if (atomKK) atomKK->modified(Host, X_MASK);
#endif
  comm->forward_comm();

  // raw teacher force at the clamped geometry -- a genuine whole-system
  // compute(), NOT local_partial_force(): intentionally rare (once per
  // outer cycle, teacher_every steps) and O(N) per call, see teacher_pair's
  // header doc comment for why this is the chosen design.
  force_clear_local();
  const double t_pair_start = platform::walltime();
#ifdef LMP_KOKKOS
  // same bracketing as VerletKokkos::run() uses for force->pair: the zeroed
  // host forces (and the clamped x, if forward_comm() ran on the host) go
  // to the teacher's execution space; its forces are claimed there and
  // auto_sync brings them back to the host
  if (atomKK) {
    atomKK->modified(Host, F_MASK);
    atomKK->sync(teacher_pair->execution_space, teacher_pair->datamask_read);
  }
  teacher_pair->compute(0, 0);
  if (atomKK) {
    atomKK->modified(teacher_pair->execution_space, teacher_pair->datamask_modify);
    atomKK->sync(Host, F_MASK);
  }
#else
  teacher_pair->compute(0, 0);
#endif
  t_teacher_pair += platform::walltime() - t_pair_start;
  if (force->newton) comm->reverse_comm();
#ifdef LMP_KOKKOS
  if (atomKK) atomKK->sync(Host, X_MASK | F_MASK);
#endif
  n_teacher_calls++;

  apply_modal_filter_and_backreaction();

  // f[] now holds the filtered TEACHER force; extract f_delta for flagged
  // atoms before restoring f[] to the student's snapshot below. A genuine
  // FORCE (eV/Angstrom), not yet divided by mass -- teacher_outer_kick()
  // divides by mass at the point of use, same convention as every other
  // velocity update in this file (see dtfm = dtf*invmass elsewhere).
  for (size_t k = 0; k < flagged.size(); k++) {
    int i = flagged[k];
    f_delta[i][0] = f[i][0] - fsave[3 * i + 0];
    f_delta[i][1] = f[i][1] - fsave[3 * i + 1];
    f_delta[i][2] = f[i][2] - fsave[3 * i + 2];
  }

  for (size_t k = 0; k < flagged.size(); k++) {
    int i = flagged[k];
    x[i][0] = xsave[3 * k + 0];
    x[i][1] = xsave[3 * k + 1];
    x[i][2] = xsave[3 * k + 2];
  }
#ifdef LMP_KOKKOS
  if (atomKK) atomKK->modified(Host, X_MASK);
#endif
  comm->forward_comm();

  if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));

#ifdef LMP_KOKKOS
  if (atomKK) {
    atomKK->sync(Host, X_MASK);
    atomKK->modified(Host, X_MASK | F_MASK);
  }
  if (lmp->kokkos) lmp->kokkos->auto_sync = prev_auto_sync;
#endif
  t_teacher_total += platform::walltime() - t_start;
}

/* ----------------------------------------------------------------------
   D.14 teacher/student composition (EXPERIMENTAL): the half-kick impulse
   half_m_dt*f_delta[i]/m_i bracketing the teacher_every inner steps
   (half_m_dt = half the cycle's duration, 0.5*teacher_every*dt when dt is
   constant; see final_integrate() for cycles in which adapt changed dt), applied directly to v[i] -- deliberately NOT folded into
   atom->f (which modal_force_correction()/force_moll() already manage for
   the student force), so this stays a clean, separate addition mirroring
   the Python reference's own "vw += 0.5*m*dt*gd" (lj_multifidelity_
   experiment.py), independent of whichever force-based bookkeeping is
   happening elsewhere this step. Called from both initial_integrate()
   (the carried f_delta from the previous cycle's close) and
   final_integrate() (the freshly computed f_delta, right after
   compute_force_delta() refreshes it) -- see the D.14 outer-cycle-boundary
   checks there. Same exile_substep exclusion as the ordinary B-kick: an
   exiled atom already gets its own full-step local treatment via
   sub_step_free_atom() and must not also receive this shared-pipeline kick.
------------------------------------------------------------------------- */

void FixBAOABTether::teacher_outer_kick(double half_m_dt)
{
  double **v = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (int i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    if (exile_substep && demote_cool[i] > update->ntimestep) continue;
    double mi = rmass ? rmass[i] : mass[type[i]];
    double scale = half_m_dt * force->ftm2v / mi;
    v[i][0] += scale * f_delta[i][0];
    v[i][1] += scale * f_delta[i][1];
    v[i][2] += scale * f_delta[i][2];
  }
}

/* ----------------------------------------------------------------------
   kspace_center: per-atom k-space force at the current configuration
------------------------------------------------------------------------- */

void FixBAOABTether::kspace_snapshot()
{
  int nlocal = atom->nlocal;
  fks.assign((size_t) 3 * nlocal, 0.0);
  if (!force->kspace) return;
  double **f = atom->f;
  int ntotal = nlocal + atom->nghost;
  std::vector<double> fsave(3 * ntotal);
  if (ntotal) memcpy(fsave.data(), &f[0][0], 3 * ntotal * sizeof(double));
  force_clear_local();
  force->kspace->compute(0, 0);
  for (int i = 0; i < nlocal; i++)
    for (int k = 0; k < 3; k++) fks[3 * i + k] = f[i][k];
  if (ntotal) memcpy(&f[0][0], fsave.data(), 3 * ntotal * sizeof(double));
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
  double bytes = (double) atom->nmax * (3 + 3 + 9 + 3 + 1 + 3) * sizeof(double);
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
   generous cutoff/skin). [8..10] stage-20 read-only diagnostic: cached-vs-
   fresh mode-eligibility agreement/disagreement counts from
   diag_check_staleness() -- see n_mask_agree/n_mask_stale_soft/
   n_mask_stale_tight in the header. [11] D.14 (EXPERIMENTAL): teacher_pair
   ->compute() calls (n_teacher_calls) -- the MLIP-call-count headline
   metric for the teacher/student composition, always 0 when
   teacher_every == 0.
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
    case 8: return (double) n_mask_agree;
    case 9: return (double) n_mask_stale_soft;
    case 10: return (double) n_mask_stale_tight;
    case 11: return (double) n_teacher_calls;
    case 12: return diag_om_unres;
    case 13: return (double) diag_unres_type;
    case 14: return diag_om_unres_p99;
    case 15: return diag_om_analytic;
    case 16: return t_teacher_total;
    case 17: return t_teacher_pair;
    default: return 0.0;
  }
}

/* ----------------------------------------------------------------------
   mark the start of a run for the teacher timing summary in post_run()
------------------------------------------------------------------------- */

void FixBAOABTether::setup(int /*vflag*/)
{
  t_run_start = platform::walltime();
  t_teacher_total_run0 = t_teacher_total;
  t_teacher_pair_run0 = t_teacher_pair;
  n_teacher_calls_run0 = n_teacher_calls;
  n_teacher_dt_varied_run0 = n_teacher_dt_varied;
}

/* ----------------------------------------------------------------------
   teacher/student runs: report how the run's wall time splits between the
   teacher calls and everything else (the inner steps, output, ...). The
   slowest rank's teacher time is reported, since the call is collective.
------------------------------------------------------------------------- */

void FixBAOABTether::post_run()
{
  if ((teacher_every <= 0) || !teacher_pair) return;

  double tloc[2] = {t_teacher_total - t_teacher_total_run0, t_teacher_pair - t_teacher_pair_run0};
  double tmax[2];
  MPI_Allreduce(tloc, tmax, 2, MPI_DOUBLE, MPI_MAX, world);
  const double trun = platform::walltime() - t_run_start;
  const bigint ncalls = n_teacher_calls - n_teacher_calls_run0;
  const bigint nsteps = update->nsteps;

  if ((comm->me == 0) && (ncalls > 0)) {
    const double trest = trun - tmax[0];
    utils::logmesg(lmp,
                   "Fix baoab/tether teacher timing: {} calls in {} steps, run wall time {:.4g} s\n"
                   "  teacher calls   {:.4g} s ({:.1f}%), {:.4g} ms per call, of which the "
                   "teacher pair style {:.4g} ms\n"
                   "  everything else {:.4g} s ({:.1f}%), {:.4g} ms per step\n",
                   ncalls, nsteps, trun, tmax[0], (trun > 0.0) ? 100.0 * tmax[0] / trun : 0.0,
                   1000.0 * tmax[0] / ncalls, 1000.0 * tmax[1] / ncalls, trest,
                   (trun > 0.0) ? 100.0 * trest / trun : 0.0,
                   (nsteps > 0) ? 1000.0 * trest / nsteps : 0.0);
    if (n_teacher_dt_varied > n_teacher_dt_varied_run0)
      utils::logmesg(lmp, "  {} outer cycles changed timestep; their teacher kicks used the actual cycle "
                     "duration\n", n_teacher_dt_varied - n_teacher_dt_varied_run0);
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
  memory->grow(bonded, nmax, "baoab/tether:bonded");
  memory->grow(last_demote, nmax, "baoab/tether:last_demote");
  memory->grow(demote_cool, nmax, "baoab/tether:demote_cool");
  memory->grow(quench_exempt, nmax, "baoab/tether:quench_exempt");
  memory->grow(last_ke, nmax, "baoab/tether:last_ke");
  memory->grow(f_delta, nmax, 3, "baoab/tether:f_delta");
  if (molblock) {
    // molecule state must survive a mid-run grow (more ghosts): clear only the new rows
    memory->grow(mb_state, nmax, mb_ss, "baoab/tether:mb_state");
    memory->grow(mb_anc, nmax, "baoab/tether:mb_anc");
    for (int i = mb_nmax; i < nmax; i++) {
      mb_state[i][0] = 0.0;
      mb_anc[i] = 0;
    }
    mb_nmax = MAX(mb_nmax, nmax);
  }

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
    bonded[i] = 0;
    last_demote[i] = 0;
    demote_cool[i] = 0;
    quench_exempt[i] = 0;
    last_ke[i] = 0.0;
    f_delta[i][0] = f_delta[i][1] = f_delta[i][2] = 0.0;
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
  bonded[j] = bonded[i];
  last_demote[j] = last_demote[i];
  demote_cool[j] = demote_cool[i];
  quench_exempt[j] = quench_exempt[i];
  last_ke[j] = last_ke[i];
  for (int k = 0; k < 3; k++) f_delta[j][k] = f_delta[i][k];
  if (molblock) {
    mb_anc[j] = mb_anc[i];
    if (mb_state[i][0] != 0.0)
      for (int k = 0; k < mb_ss; k++) mb_state[j][k] = mb_state[i][k];
    else
      mb_state[j][0] = 0.0;
  }
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
  buf[m++] = ubuf(bonded[i]).d;
  buf[m++] = ubuf(last_demote[i]).d;
  buf[m++] = ubuf(demote_cool[i]).d;
  buf[m++] = ubuf(quench_exempt[i]).d;
  buf[m++] = last_ke[i];
  for (int k = 0; k < 3; k++) buf[m++] = f_delta[i][k];
  if (molblock) {
    buf[m++] = ubuf(mb_anc[i]).d;
    if (mb_state[i][0] != 0.0)
      for (int k = 0; k < mb_ss; k++) buf[m++] = mb_state[i][k];
    else
      buf[m++] = 0.0;
  }
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
  bonded[nlocal] = (int) ubuf(buf[m++]).i;
  last_demote[nlocal] = (bigint) ubuf(buf[m++]).i;
  demote_cool[nlocal] = (bigint) ubuf(buf[m++]).i;
  quench_exempt[nlocal] = (bigint) ubuf(buf[m++]).i;
  last_ke[nlocal] = buf[m++];
  for (int k = 0; k < 3; k++) f_delta[nlocal][k] = buf[m++];
  if (molblock) {
    mb_anc[nlocal] = (tagint) ubuf(buf[m++]).i;
    mb_state[nlocal][0] = buf[m++];
    if (mb_state[nlocal][0] != 0.0)
      for (int k = 1; k < mb_ss; k++) mb_state[nlocal][k] = buf[m++];
  }
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
  // molblock: velocities and forces (once per outer step), or the refresh probe data
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
    return m;
  }
  if (pbc_flag == 0) {
    for (int i = 0; i < n; i++) {
      int j = list[i];
      buf[m++] = c[j][0];
      buf[m++] = c[j][1];
      buf[m++] = c[j][2];
      buf[m++] = om[j][0];
      buf[m++] = om[j][1];
      buf[m++] = om[j][2];
      buf[m++] = (double) bonded[j];
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
      // om[] is a frequency, not a coordinate -- no PBC offset applies.
      buf[m++] = om[j][0];
      buf[m++] = om[j][1];
      buf[m++] = om[j][2];
      buf[m++] = (double) bonded[j];
    }
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBAOABTether::unpack_forward_comm(int n, int first, double *buf)
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
    return;
  }
  for (int i = first; i < last; i++) {
    c[i][0] = buf[m++];
    c[i][1] = buf[m++];
    c[i][2] = buf[m++];
    om[i][0] = buf[m++];
    om[i][1] = buf[m++];
    om[i][2] = buf[m++];
    bonded[i] = (int) buf[m++];
  }
}

/* ----------------------------------------------------------------------
   molblock: ghost copies of anchor atoms get the owner's molecule state at
   every reneighboring (state of other atoms: a single 0 flag)
------------------------------------------------------------------------- */

int FixBAOABTether::pack_border(int n, int *list, double *buf)
{
  int m = 0;
  if (!molblock) return m;
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

int FixBAOABTether::unpack_border(int n, int first, double *buf)
{
  int m = 0;
  if (!molblock) return m;
  int last = first + n;
  for (int i = first; i < last; i++) {
    mb_state[i][0] = buf[m++];
    if (mb_state[i][0] != 0.0)
      for (int k = 1; k < mb_ss; k++) mb_state[i][k] = buf[m++];
    mb_anc[i] = 0;
  }
  return m;
}
