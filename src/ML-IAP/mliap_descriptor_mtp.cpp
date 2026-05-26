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

#ifdef MLIAP_MTP

#include "mliap_descriptor_mtp.h"

// #include "ace-evaluator/ace_abstract_basis.h"
// #include "ace-evaluator/ace_c_basis.h"
// #include "ace-evaluator/ace_evaluator.h"
// #include "ace-evaluator/ace_types.h"

#include "atom.h"
#include "comm.h"
#include "memory.h"
#include "mliap_data.h"
#include "pair_mliap.h"

namespace LAMMPS_NS {
// struct ACE_ML_impl {
//   ACE_ML_impl() : basis_set(nullptr), ace(nullptr) {}
//   ~ACE_ML_impl()
//   {
//     delete basis_set;
//     delete ace;
//   }
//   ACECTildeBasisSet *basis_set;
//   ACECTildeEvaluator *ace;
// };
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MLIAPDescriptorMTP::MLIAPDescriptorMTP(LAMMPS *_lmp) :
    MLIAPDescriptor(_lmp)
{
  cutoff = 5.0;
  rmin = 0.8;

  n_radial = 12;
  n_rf = 8;

  max_nu = 2;
  max_level = 24;

  n_descriptors = 0;
}

void MLIAPDescriptorMTP::read_paramfile(char *fname)
{
  std::ifstream fp(fname);

  if (!fp.good()) {
    error->all(FLERR, "Could not open MTP descriptor parameter file");
  }

  std::string line;

  species.clear();

  bool in_radial_block = false;

  while (std::getline(fp, line)) {

    // remove comments
    size_t comment = line.find('#');
    if (comment != std::string::npos)
      line = line.substr(0, comment);

    std::stringstream ss(line);
    std::string key;
    ss >> key;

    if (key.empty()) continue;

    // ------------------------------------------------------------
    // scalar parameters
    // ------------------------------------------------------------
    if (key == "cutoff") {
      ss >> cutoff;

    } else if (key == "rmin") {
      ss >> rmin;

    } else if (key == "n_radial") {
      ss >> n_radial;

    } else if (key == "n_rf") {
      ss >> n_rf;

    } else if (key == "max_nu") {
      ss >> max_nu;

    } else if (key == "max_level") {
      ss >> max_level;

    // ------------------------------------------------------------
    // species list
    // ------------------------------------------------------------
    } else if (key == "species") {

      std::string sp;
      while (ss >> sp)
        species.push_back(sp);

    // ------------------------------------------------------------
    // radial coefficient block start
    // ------------------------------------------------------------
    } else if (key == "radial_coeffs") {

      in_radial_block = true;

    } else if (key == "end_radial_coeffs") {

      in_radial_block = false;

    // ------------------------------------------------------------
    // radial coefficient entries
    // format:
    // mu itype jtype n value
    // ------------------------------------------------------------
    } else if (in_radial_block) {

      int mu, itype, jtype, n;
      double val;

      std::stringstream ls(line);
      ls >> mu >> itype >> jtype >> n >> val;

      if (mu < 0 || mu >= n_rf)
        error->all(FLERR, "Invalid mu index in radial_coeffs");

      if ((int)radial_coeffs.size() == 0) {

        radial_coeffs.resize(n_rf);
        for (int a = 0; a < n_rf; a++) {
          radial_coeffs[a].resize(10); // assume max 10 species (resize later)
          for (int i = 0; i < 10; i++) {
            radial_coeffs[a][i].resize(10);
            for (int j = 0; j < 10; j++) {
              radial_coeffs[a][i][j].resize(n_radial, 0.0);
            }
          }
        }
      }

      // ensure capacity
      if (itype >= (int)radial_coeffs[mu].size())
        radial_coeffs[mu].resize(itype+1);

      if (jtype >= (int)radial_coeffs[mu][itype].size())
        radial_coeffs[mu][itype].resize(jtype+1);

      if (n >= n_radial)
        error->all(FLERR, "Invalid radial basis index");

      radial_coeffs[mu][itype][jtype][n] = val;

    } else {

      error->warning(FLERR, "Unknown keyword in MTP descriptor file");
    }
  }

  fp.close();

  // ------------------------------------------------------------
  // finalize internal structures
  // ------------------------------------------------------------
  build_basis_index();

  // sanity checks
  if (species.size() == 0)
    error->all(FLERR, "No species defined in MTP descriptor");

  if (n_rf <= 0)
    error->all(FLERR, "Invalid n_rf in MTP descriptor");
}

/* ---------------------------------------------------------------------- */

MLIAPDescriptorMTP::~MLIAPDescriptorMTP() {}


void MLIAPDescriptorMTP::build_basis_index()
{
  basis_specs.clear();

  // nu=0 scalars
  for (int mu = 0; mu < n_rf; mu++) {
    if (2 * mu <= max_level) {
      BasisSpec spec;
      spec.type = NU0;
      spec.mu[0] = mu;
      basis_specs.push_back(spec);
    }
  }

  // nu=1 dot products
  for (int mu1 = 0; mu1 < n_rf; mu1++) {
    for (int mu2 = mu1; mu2 < n_rf; mu2++) {
      if (2 * mu1 + 4 + 2 * mu2 + 4 <= max_level) {
        BasisSpec spec;
        spec.type = NU1_DOT;
        spec.mu[0] = mu1;
        spec.mu[1] = mu2;
        basis_specs.push_back(spec);
      }
    }
  }

  // nu=2 Frobenius products
  if (max_nu >= 2) {
    for (int mu1 = 0; mu1 < n_rf; mu1++) {
      for (int mu2 = mu1; mu2 < n_rf; mu2++) {
        if (2 * mu1 + 8 + 2 * mu2 + 8 <= max_level) {
          BasisSpec spec;
          spec.type = NU2_FROB;
          spec.mu[0] = mu1;
          spec.mu[1] = mu2;
          basis_specs.push_back(spec);
        }
      }
    }
  }

  // nu0 x nu1^2
  for (int mu0 = 0; mu0 < n_rf; mu0++) {
    for (int mu1 = 0; mu1 < n_rf; mu1++) {
      if (2 * mu0 + 2 * (2 * mu1 + 4) <= max_leve) {
        BasisSpec spec;
        spec.type = NU0_X_NU1SQ;
        spec.mu[0] = mu0;
        spec.mu[1] = mu1;
        basis_specs.push_back(spec);
      }
    }
  }

  // nu0 x nu1 . nu1'
  for (int mu0 = 0; mu0 < n_rf; mu0++) {
    for (int mu1 = 0; mu1 < n_rf; mu1++) {
      for (int mu2 = mu1 + 1; mu2 < n_rf; mu2++) {
        if (2 * mu0 + (2 * mu1 + 4) + (2 * mu2 + 4) <= max_level) {
          BasisSpec spec;
          spec.type = NU0_X_NU1_NU1;
          spec.mu[0] = mu0;
          spec.mu[1] = mu1;
          spec.mu[2] = mu2;
          basis_specs.push_back(spec);
        }
      }
    }
  }

  // v^T M v
  if (max_nu >= 2) {
    for (int mu1 = 0; mu1 < n_rf; mu1++) {
      for (int mu2 = 0; mu2 < n_rf; mu2++) {
        for (int mu3 = mu1; mu3 < n_rf; mu3++) {
          if ((2 * mu1 + 4) + (2 * mu2 + 8) + (2 * mu3 + 4) <= max_level) {
            BasisSpec spec;
            spec.type = V_T_V;
            spec.mu[0] = mu1;
            spec.mu[1] = mu2;
            spec.mu[2] = mu3;
            basis_specs.push_back(spec);
          }
        }
      }
    }
  }

  // nu0 x nu0'
  for (int mu0 = 0; mu0 < n_rf; mu0++) {
    for (int mu1 = mu0 + 1; mu1 < n_rf; mu1++) {
      if (2 * mu0 + 2 * mu1 <= max_level) {
        BasisSpec spec;
        spec.type = NU0_X_NU0;
        spec.mu[0] = mu0;
        spec.mu[1] = mu1;
        basis_specs.push_back(spec);
      }
    }
  }

  // nu0^2
  for (int mu0 = 0; mu0 < n_rf; mu0++) {
    if (4 * mu0 <= max_level) {
      BasisSpec spec;
      spec.type = NU0_SQ;
      spec.mu[0] = mu0;
      basis_specs.push_back(spec);
    }
  }

  n_descriptors = species.size() + basis_specs.size();
}


double MLIAPDescriptorMTP::cutoff_function(double r)
{
  if (r >= cutoff)
    return 0.0;

  double x = (cutoff - r) / (cutoff - rmin);
  if (x < 0.0)
    x = 0.0;

  return x * x;
}

// TODO
void MLIAPDescriptorMTP::chebyshev_basis(double r, std::vector<double> &T)
{
  T.resize(n_radial);

  double x = (2.0 * r - rmin - cutoff) / (cutoff - rmin + 1e-10);

  if (x > 1.0)
    x = 1.0;
  if (x < -1.0)
    x = -1.0;

  double fc = cutoff_function(r);

  T[0] = 1.0;

  if (n_radial >= 2)
    T[1] = x;

  for (int n = 2; n < n_radial; n++) {
    T[n] = 2.0 * x * T[n - 1] - T[n - 2];
  }

  for (int n = 0; n < n_radial; n++)
    T[n] *= fc;
}

// TODO
void MLIAPDescriptorMTP::compute_radial_functions(
  double r,
  int itype,
  int jtype,
  std::vector<double> &fmu)
{
  std::vector<double> T;
  chebyshev_basis(r, T);

  fmu.resize(n_rf);

  if (itype < 0 || jtype < 0)
    error->all(FLERR, "Negative species index in MTP descriptor");
  if (itype >= (int)radial_coeffs[0].size())
    error->all(FLERR, "itype out of bounds in MTP descriptor");
  if (jtype >= (int)radial_coeffs[0][itype].size())
    error->all(FLERR, "jtype out of bounds in MTP descriptor");

  for (int mu = 0; mu < n_rf; mu++) {

    double val = 0.0;

    for (int n = 0; n < n_radial; n++) {
      val += radial_coeffs[mu][itype][jtype][n] * T[n];
    }

    fmu[mu] = val;
  }
}


void MLIAPDescriptorMTP::compute_descriptors(MLIAPData *data)
{
  double **x = atom->x;
  int *type = atom->type;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {

    int i = ilist[ii];

    std::vector<double> M0(n_rf, 0.0);
    std::vector<double> M1(n_rf * 3, 0.0);
    std::vector<double> M2(n_rf * 9, 0.0);

    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {

      int j = jlist[jj] & NEIGHMASK;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      double r = sqrt(rsq);

      if (r >= cutoff) continue;

      std::vector<double> fmu;
      compute_radial_functions(r,
                               type[i]-1,
                               type[j]-1,
                               fmu);

      for (int mu = 0; mu < n_rf; mu++) {

        double f = fmu[mu];

        // M0
        M0[mu] += f;

        // M1
        M1[mu*3 + 0] += f * dx;
        M1[mu*3 + 1] += f * dy;
        M1[mu*3 + 2] += f * dz;

        // M2
        if (max_nu >= 2) {
          M2[mu*9 + 0] += f * dx * dx;
          M2[mu*9 + 1] += f * dx * dy;
          M2[mu*9 + 2] += f * dx * dz;
          M2[mu*9 + 3] += f * dy * dx;
          M2[mu*9 + 4] += f * dy * dy;
          M2[mu*9 + 5] += f * dy * dz;
          M2[mu*9 + 6] += f * dz * dx;
          M2[mu*9 + 7] += f * dz * dy;
          M2[mu*9 + 8] += f * dz * dz;
        }
      }
    }

    int k = 0;

    for (int s = 0; s < (int)species.size(); s++)
      data->descriptors[ii][k++] = ((type[i]-1) == s);

    for (size_t b = 0; b < basis_specs.size(); b++) {

      const BasisSpec &spec = basis_specs[b];

      double val = 0.0;

      if (spec.type == NU0) {

        val = M0[spec.mu[0]];

      } else if (spec.type == NU1_DOT) {

        int a = spec.mu[0];
        int b2 = spec.mu[1];

        val =
          M1[a*3+0]*M1[b2*3+0] +
          M1[a*3+1]*M1[b2*3+1] +
          M1[a*3+2]*M1[b2*3+2];

      } else if (spec.type == NU2_FROB) {

        int a = spec.mu[0];
        int b2 = spec.mu[1];

        for (int q = 0; q < 9; q++)
          val += M2[a*9+q] * M2[b2*9+q];
      }

      data->descriptors[ii][k++] = val;
    }
  }
}


void MLIAPDescriptorMTP::compute_descriptor_gradients(MLIAPData *data)
{
  double **x = atom->x;
  int *type = atom->type;

  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;

  for (int ii = 0; ii < inum; ii++) {

    int i = ilist[ii];

    int *jlist = firstneigh[i];
    int jnum = numneigh[i];

    for (int jj = 0; jj < jnum; jj++) {

      int j = jlist[jj] & NEIGHMASK;

      double dx = x[j][0] - x[i][0];
      double dy = x[j][1] - x[i][1];
      double dz = x[j][2] - x[i][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      double r = sqrt(rsq);

      if (r >= cutoff) continue;

      double rinv = 1.0 / (r + 1e-12);

      double ex = dx * rinv;
      double ey = dy * rinv;
      double ez = dz * rinv;

      // radial functions + derivatives
      std::vector<double> fmu, dfmu;
      fmu.resize(n_rf);
      dfmu.resize(n_rf);

      for (int mu = 0; mu < n_rf; mu++) {

        double f = 0.0;
        double df = 0.0;

        for (int n = 0; n < n_radial; n++) {

          double c =
            radial_coeffs[mu][type[i]-1][type[j]-1][n];

          double Tn = 0.0; // assume cached or recomputed consistently
          double dTn = 0.0;

          f  += c * Tn;
          df += c * dTn;
        }

        fmu[mu] = f;
        dfmu[mu] = df;
      }

      int k = species.size();

      // ---------------- NU0 ----------------
      for (int mu = 0; mu < n_rf; mu++) {

        double df = dfmu[mu];

        data->graddesc[ii][k][0] += df * ex;
        data->graddesc[ii][k][1] += df * ey;
        data->graddesc[ii][k][2] += df * ez;

        k++;
      }

      // ---------------- NU1 ----------------
      for (int mu = 0; mu < n_rf; mu++) {

        double f = fmu[mu];
        double df = dfmu[mu];

        for (int a = 0; a < 3; a++) {
          for (int b = 0; b < 3; b++) {

            double ra = (a==0?dx:(a==1?dy:dz));
            double rb = (b==0?dx:(b==1?dy:dz));

            double delta = (a==b);

            data->graddesc[ii][k][b] +=
              df * ra * rb * rinv + f * delta;
          }
        }

        k++;
      }

      // ---------------- NU2 ----------------
      if (max_nu >= 2) {

        for (int mu = 0; mu < n_rf; mu++) {

          double f = fmu[mu];
          double df = dfmu[mu];

          for (int a = 0; a < 3; a++) {
            for (int b = 0; b < 3; b++) {
              for (int c = 0; c < 3; c++) {

                double ra = (a==0?dx:(a==1?dy:dz));
                double rb = (b==0?dx:(b==1?dy:dz));
                double rc = (c==0?dx:(c==1?dz:dz));

                double term =
                  df * ra * rb * rc * rinv +
                  f * (
                    (a==b ? rc : 0.0) +
                    (a==c ? rb : 0.0) +
                    (b==c ? ra : 0.0)
                  );

                data->graddesc[ii][k][c] += term;
              }
            }
          }

          k++;
        }
      }
    }
  }
}


/* ---------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double MLIAPDescriptorMTP::memory_usage()
{
  double bytes = MLIAPDescriptor::memory_usage();

  return bytes;
}

#endif
