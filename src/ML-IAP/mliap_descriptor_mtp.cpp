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

//#ifdef MLIAP_MTP

#include "mliap_descriptor_mtp.h"

#include "atom.h"
#include "error.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "pair_mliap.h"
#include "mliap_data.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

MLIAPDescriptorMTP::MLIAPDescriptorMTP(LAMMPS *_lmp) :
    Pointers(_lmp), MLIAPDescriptor(_lmp)
{
  // TODO replace hard-coded vals with read param file method
  cutoff = 5.0;
  rmin = 0.8;

  n_radial = 12;
  n_rf = 8;

  max_nu = 2;
  max_level = 24;

  nelements = 2;
  elements = new char *[nelements];
  elements[0] = utils::strdup("Hf");
  elements[1] = utils::strdup("O");

  cutsq = new double*[nelements];
  for (int i = 0; i < nelements; i++) {
    cutsq[i] = new double[nelements];
    for (int j = 0; j < nelements; j++) {
      cutsq[i][j] = cutoff * cutoff;
    }
  }

  radelem = new double[nelements];
  wjelem  = new double[nelements];

  for (int i = 0; i < nelements; i++) {
    radelem[i] = cutoff;
    wjelem[i]  = 1.0;
  }

  cutmax = cutoff;

  radial_coeffs = {{{{-0.2447, -0.0318,  0.0754,  0.1395, -0.1484,  0.0208,  0.0844,
           -0.1241,  0.0925,  0.0688, -0.0540,  0.0526},
          { 0.0042,  0.0917, -0.0252, -0.0610,  0.1305,  0.0653, -0.0369,
            0.1262, -0.1013,  0.1202, -0.0585,  0.0264}},
         {{-0.1521,  0.4008,  0.2365, -0.1499,  0.1519, -0.0173, -0.0042,
           -0.0246, -0.0036, -0.0513,  0.0146, -0.0260},
          { 0.1476, -0.3381, -0.1169,  0.1768, -0.0987, -0.0911,  0.1166,
           -0.0076,  0.0092,  0.0027, -0.0258,  0.0099}}},
        {{{-0.0352, -0.1239,  0.0984,  0.0361,  0.0329, -0.0730,  0.0708,
            0.0022, -0.0166,  0.0733, -0.0199,  0.0212},
          { 0.1127, -0.1117, -0.0670,  0.2263, -0.2338,  0.1436,  0.1020,
           -0.1920,  0.1989, -0.1239,  0.0523, -0.0205}},
         {{ 0.2011, -0.2627, -0.1469,  0.1648, -0.1849, -0.0698,  0.1653,
           -0.1533, -0.0114,  0.0688, -0.0374,  0.0207},
          { 0.0781, -0.1464,  0.1385, -0.0886, -0.0648, -0.0479, -0.0042,
            0.0007, -0.0623,  0.0068, -0.0163,  0.0040}}},
        {{{-0.2271, -0.0410,  0.1535, -0.0994,  0.0804,  0.0542,  0.0874,
            0.0205, -0.0566,  0.1701, -0.1366,  0.0880},
          { 0.1981, -0.1798, -0.0212,  0.0405,  0.0269, -0.1858,  0.0614,
            0.0035, -0.0844,  0.0912, -0.0525,  0.0290}},
         {{-0.1807, -0.2523, -0.0034,  0.0007, -0.1189,  0.0408, -0.0159,
           -0.0510, -0.0035,  0.0372, -0.0217,  0.0072},
          { 0.1311,  0.1413, -0.1593, -0.1606,  0.0378, -0.1025, -0.2056,
            0.1364, -0.1094, -0.0013, -0.0206, -0.0373}}},
        {{{ 0.2233,  0.0307, -0.1252,  0.0097,  0.1242, -0.0192, -0.1376,
            0.0655,  0.0555, -0.0627,  0.0618, -0.0513},
          { 0.0849,  0.0838, -0.1443,  0.1232, -0.2335,  0.1126, -0.0903,
           -0.0397, -0.0785,  0.0484, -0.0513,  0.0045}},
         {{-0.0511,  0.3114,  0.0510, -0.1860,  0.2855, -0.0076, -0.1646,
            0.2708, -0.2459,  0.1174, -0.0331, -0.0036},
          {-0.0429, -0.0498,  0.0480, -0.0121,  0.0943, -0.1320,  0.0387,
            0.0977, -0.0346,  0.0483, -0.0472,  0.0121}}},
        {{{ 0.0017,  0.0191,  0.0900, -0.0645,  0.0645,  0.1803, -0.0989,
            0.1618,  0.0903,  0.0585,  0.1584, -0.0665},
          {-0.1189, -0.0177, -0.1610, -0.1198,  0.2687, -0.0911,  0.1409,
            0.1545, -0.1616,  0.0156,  0.0058, -0.0927}},
         {{-0.0644, -0.0048,  0.1127,  0.0167, -0.1665, -0.1081,  0.1935,
            0.0259, -0.1781,  0.0779,  0.0277, -0.0143},
          { 0.1150, -0.0781, -0.1805,  0.0807, -0.0628,  0.0251, -0.0194,
           -0.1175, -0.0741,  0.0766,  0.0182,  0.0037}}},
        {{{-0.0388, -0.3264,  0.0051,  0.2516, -0.0125,  0.0237, -0.0725,
            0.0110, -0.0127, -0.0871,  0.1302, -0.0701},
          { 0.0675,  0.1019,  0.2805, -0.3129, -0.2140,  0.1077, -0.0609,
            0.0519, -0.0702,  0.0035,  0.0008, -0.0151}},
         {{ 0.4596, -0.1280, -0.3715, -0.1221,  0.3463, -0.2010,  0.0017,
           -0.1269, -0.0982,  0.0573, -0.0513, -0.1273},
          { 0.0752,  0.0196, -0.1749, -0.1342,  0.2093, -0.0124, -0.2006,
           -0.0512,  0.1997,  0.0291,  0.0679, -0.0592}}},
        {{{ 0.3244,  0.1113, -0.0616,  0.0581,  0.0030, -0.0107,  0.0733,
           -0.0252,  0.0375, -0.0201,  0.0007,  0.0576},
          {-0.4145,  0.2638,  0.0613, -0.3144, -0.1169, -0.0404, -0.0933,
            0.0047, -0.0482,  0.0934, -0.0588,  0.0761}},
         {{-0.1471,  0.1471,  0.1179, -0.0543,  0.0930,  0.2975,  0.0950,
           -0.0192,  0.0035,  0.1687, -0.1024, -0.0020},
          { 0.0780, -0.2967,  0.0461,  0.1159, -0.0736, -0.0665,  0.3035,
           -0.2316, -0.0116,  0.1327, -0.0472,  0.0801}}},
        {{{ 0.0424,  0.1429, -0.0733, -0.1524, -0.1965,  0.0758,  0.2900,
           -0.0539, -0.0053, -0.0039,  0.2369, -0.0459},
          { 0.2965, -0.1077, -0.2609,  0.0010, -0.1916,  0.0761,  0.1841,
            0.0561, -0.0155,  0.0884, -0.0855,  0.0201}},
         {{ 0.0552,  0.1000,  0.1373,  0.0681, -0.1462, -0.0236,  0.1402,
           -0.2282, -0.1106,  0.1163, -0.0921, -0.0582},
          {-0.1235,  0.3442, -0.0397, -0.3588,  0.0730,  0.0977, -0.1698,
            0.0307,  0.0226,  0.0568,  0.0417, -0.0255}}}};

  build_basis_index();
  ndescriptors = nelements + basis_specs.size();
}

/*
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

  // sanity checks
  if (nelements == 0)
    error->all(FLERR, "No species defined in MTP descriptor");

  if (n_rf <= 0)
    error->all(FLERR, "Invalid n_rf in MTP descriptor");
}
*/

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
      if (2 * mu0 + 2 * (2 * mu1 + 4) <= max_level) {
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

void MLIAPDescriptorMTP::chebyshev_basis(
    double r,
    std::vector<double>& basis,
    std::vector<double>& dbasisdr)
{
    basis.resize(n_radial);
    dbasisdr.resize(n_radial);

    double denom = cutoff - rmin + 1e-10;

    double x = (2.0*r - rmin - cutoff)/denom;
    x = std::clamp(x, -1.0, 1.0);

    double dxdr = 2.0/denom;

    double fc = cutoff_function(r);

    double dfcdr = 0.0;
    if (r < cutoff) {
        double tmp = (cutoff-r)/(cutoff-rmin);
        if (tmp > 0.0)
            dfcdr = -2.0*tmp/(cutoff-rmin);
    }

    std::vector<double> T(n_radial);
    std::vector<double> dTdx(n_radial);

    T[0] = 1.0;
    dTdx[0] = 0.0;

    if (n_radial >= 2) {
        T[1] = x;
        dTdx[1] = 1.0;
    }

    for (int n=2; n<n_radial; n++) {
        T[n] = 2*x*T[n-1] - T[n-2];

        dTdx[n] =
            2*T[n-1]
          + 2*x*dTdx[n-1]
          - dTdx[n-2];
    }

    for (int n=0; n<n_radial; n++) {
        basis[n] =
            T[n] * fc;

        dbasisdr[n] =
            dTdx[n]*dxdr*fc
          + T[n]*dfcdr;
    }
}

void MLIAPDescriptorMTP::compute_radial_functions(
  double r,
  int itype,
  int jtype,
  std::vector<double> &fmu)
{
  std::vector<double> T;
  std::vector<double> dTdr;
  chebyshev_basis(r, T, dTdr);

  fmu.resize(n_rf);

  for (int mu = 0; mu < n_rf; mu++) {

      double f = 0.0;

      for (int n = 0; n < n_radial; n++) {
          f += radial_coeffs[mu][itype][jtype][n] * T[n];
      }

      fmu[mu] = f;
  }
}


void MLIAPDescriptorMTP::compute_descriptors(MLIAPData *data)
{
  int *type = atom->type;

  int nlistatoms = data->nlistatoms;

  int *iatoms = data->iatoms;
  int *numneighs = data->numneighs;

  int *pair_i = data->pair_i;
  int *jatoms = data->jatoms;

  double **rij = data->rij;

  int pair_index = 0;

  for (int ii = 0; ii < nlistatoms; ii++) {

    int i = iatoms[ii];

    std::vector<double> M0(n_rf, 0.0);
    std::vector<double> M1(n_rf * 3, 0.0);
    std::vector<double> M2;
    if (max_nu >= 2) {
      M2.resize(n_rf * 9, 0.0);
    }

    int jnum = numneighs[ii];

    for (int jj = 0; jj < jnum; jj++, pair_index++) {

      int j = jatoms[pair_index];

      double dx = rij[pair_index][0];
      double dy = rij[pair_index][1];
      double dz = rij[pair_index][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      double r = sqrt(rsq);

      if (r >= cutoff)
        continue;

      std::vector<double> fmu;

      compute_radial_functions(
        r,
        type[i]-1,
        type[j]-1,
        fmu);

      for (int mu = 0; mu < n_rf; mu++) {

        double f = fmu[mu];

        // -------------------------------------------------
        // M0
        // -------------------------------------------------

        M0[mu] += f;

        // -------------------------------------------------
        // M1
        // -------------------------------------------------

        M1[mu*3 + 0] += f * dx;
        M1[mu*3 + 1] += f * dy;
        M1[mu*3 + 2] += f * dz;

        // -------------------------------------------------
        // M2
        // -------------------------------------------------

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

    // -----------------------------------------------------
    // Build invariant descriptor vector
    // -----------------------------------------------------

    int k = 0;

    // Species one-hot
    for (int s = 0; s < nelements; s++) {

      data->descriptors[ii][k++] =
        ((type[i]-1) == s) ? 1.0 : 0.0;
    }

    // Invariant basis functions
    for (size_t b = 0; b < basis_specs.size(); b++) {
      const BasisSpec &spec = basis_specs[b];
      double val = 0.0;

      if (spec.type == NU0) {
        val = M0[spec.mu[0]];
      } else if (spec.type == NU1_DOT) {
        int a = spec.mu[0];
        int b2 = spec.mu[1];

        val =
          M1[a*3 + 0] * M1[b2*3 + 0] +
          M1[a*3 + 1] * M1[b2*3 + 1] +
          M1[a*3 + 2] * M1[b2*3 + 2];

      } else if (spec.type == NU2_FROB) {
        int a = spec.mu[0];
        int b2 = spec.mu[1];

        for (int q = 0; q < 9; q++) {
          val += M2[a*9 + q] * M2[b2*9 + q];
        }
      } else if (spec.type == NU0_X_NU1SQ) {
        int mu0 = spec.mu[0];
        int mu1 = spec.mu[1];

        double norm2 =
            M1[mu1*3 + 0] * M1[mu1*3 + 0] +
            M1[mu1*3 + 1] * M1[mu1*3 + 1] +
            M1[mu1*3 + 2] * M1[mu1*3 + 2];

        val = M0[mu0] * norm2;
      } else if (spec.type == NU0_X_NU1_NU1) {
        int mu0 = spec.mu[0];
        int mu1 = spec.mu[1];
        int mu2 = spec.mu[2];

        double dot =
            M1[mu1*3 + 0] * M1[mu2*3 + 0] +
            M1[mu1*3 + 1] * M1[mu2*3 + 1] +
            M1[mu1*3 + 2] * M1[mu2*3 + 2];

        val = M0[mu0] * dot;
      } else if (spec.type == V_T_V) {
        int mu1 = spec.mu[0];
        int mu2 = spec.mu[1];
        int mu3 = spec.mu[2];

        double Tv[3];

        Tv[0] =
            M1[mu1*3 + 0] * M2[mu2*9 + 0] +
            M1[mu1*3 + 1] * M2[mu2*9 + 3] +
            M1[mu1*3 + 2] * M2[mu2*9 + 6];

        Tv[1] =
            M1[mu1*3 + 0] * M2[mu2*9 + 1] +
            M1[mu1*3 + 1] * M2[mu2*9 + 4] +
            M1[mu1*3 + 2] * M2[mu2*9 + 7];

        Tv[2] =
            M1[mu1*3 + 0] * M2[mu2*9 + 2] +
            M1[mu1*3 + 1] * M2[mu2*9 + 5] +
            M1[mu1*3 + 2] * M2[mu2*9 + 8];

        val =
            Tv[0] * M1[mu3*3 + 0] +
            Tv[1] * M1[mu3*3 + 1] +
            Tv[2] * M1[mu3*3 + 2];
      } else if (spec.type == NU0_X_NU0) {
        int mu0 = spec.mu[0];
        int mu1 = spec.mu[1];
        val = M0[mu0] * M0[mu1];
      } else if (spec.type == NU0_SQ) {
        int mu0 = spec.mu[0];
        val = M0[mu0] * M0[mu0];
      } else {
        printf("Basis type: %d\n", spec.type);
        error->all(FLERR, "Unknown MTP basis function type");
      }

      data->descriptors[ii][k++] = val;
    }
  }
}


void MLIAPDescriptorMTP::compute_descriptor_gradients(
  MLIAPData *data)
{
  int *type = atom->type;
  int nlistatoms = data->nlistatoms;
  int *iatoms = data->iatoms;
  int *numneighs = data->numneighs;
  int *jatoms = data->jatoms;
  double **rij = data->rij;

  int pair_index = 0;

  for (int ii = 0; ii < nlistatoms; ii++) {
    int i = iatoms[ii];
    int jnum = numneighs[ii];

    // -------------------------------------------------------
    // First pass:
    // build full moment tensors for atom i
    // -------------------------------------------------------

    std::vector<double> M0(n_rf, 0.0);
    std::vector<double> M1(n_rf * 3, 0.0);
    std::vector<double> M2(n_rf * 9, 0.0);

    int pair_start = pair_index;

    for (int jj = 0; jj < jnum; jj++, pair_index++) {

      int j = jatoms[pair_index];

      double dx = rij[pair_index][0];
      double dy = rij[pair_index][1];
      double dz = rij[pair_index][2];

      double r = sqrt(dx*dx + dy*dy + dz*dz);

      if (r >= cutoff)
        continue;

      // ---------------------------------------------
      // Compute Chebyshev basis + radial functions
      // ---------------------------------------------

      std::vector<double> T(n_radial, 0.0);

      double x =
        (2.0 * r - rmin - cutoff) /
        (cutoff - rmin + 1e-10);

      if (x > 1.0) x = 1.0;
      if (x < -1.0) x = -1.0;

      double fc = cutoff_function(r);

      T[0] = 1.0;

      if (n_radial >= 2)
        T[1] = x;

      for (int n = 2; n < n_radial; n++)
        T[n] = 2.0 * x * T[n-1] - T[n-2];

      std::vector<double> fmu(n_rf, 0.0);

      for (int mu = 0; mu < n_rf; mu++) {

        double f = 0.0;

        for (int n = 0; n < n_radial; n++) {

          f +=
            radial_coeffs[mu][type[i]-1][type[j]-1][n]
            * T[n];
        }

        fmu[mu] = f;

        // -----------------------------------------
        // M0
        // -----------------------------------------

        M0[mu] += f;

        // -----------------------------------------
        // M1
        // -----------------------------------------

        M1[mu*3 + 0] += f * dx;
        M1[mu*3 + 1] += f * dy;
        M1[mu*3 + 2] += f * dz;

        // -----------------------------------------
        // M2
        // -----------------------------------------

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

    // -------------------------------------------------------
    // Second pass:
    // descriptor derivatives
    // -------------------------------------------------------

    pair_index = pair_start;

    for (int jj = 0; jj < jnum; jj++, pair_index++) {

      int j = jatoms[pair_index];

      double dx = rij[pair_index][0];
      double dy = rij[pair_index][1];
      double dz = rij[pair_index][2];

      double rsq = dx*dx + dy*dy + dz*dz;
      double r = sqrt(rsq);

      if (r >= cutoff)
        continue;

      double rinv = 1.0 / (r + 1e-20);

      double ex = dx * rinv;
      double ey = dy * rinv;
      double ez = dz * rinv;

      // ---------------------------------------------------
      // Chebyshev basis + derivatives
      // ---------------------------------------------------

      std::vector<double> basis;
      std::vector<double> dbasisdr;

      chebyshev_basis(r, basis, dbasisdr);

      std::vector<double> fmu(n_rf, 0.0);
      std::vector<double> dfmu(n_rf, 0.0);

      for (int mu = 0; mu < n_rf; mu++) {

        double f = 0.0;
        double df = 0.0;

        for (int n = 0; n < n_radial; n++) {

          double c =
            radial_coeffs[mu][type[i]-1][type[j]-1][n];

          f += c * basis[n];
          df += c * dbasisdr[n];
        }

        fmu[mu] = f;
        dfmu[mu] = df;
      }

      // ---------------------------------------------------
      // Zero gradients
      // ---------------------------------------------------

      for (int k = 0; k < ndescriptors; k++) {

        data->graddesc[pair_index][k][0] = 0.0;
        data->graddesc[pair_index][k][1] = 0.0;
        data->graddesc[pair_index][k][2] = 0.0;
      }

      int k = nelements;

      // ===================================================
      // Descriptor derivatives
      // ===================================================

      for (size_t b = 0; b < basis_specs.size(); b++, k++) {

        const BasisSpec &spec = basis_specs[b];

        double gx = 0.0;
        double gy = 0.0;
        double gz = 0.0;

        // -------------------------------------------------
        // NU0
        // -------------------------------------------------

        if (spec.type == NU0) {

          int mu = spec.mu[0];

          double df = dfmu[mu];

          gx = df * ex;
          gy = df * ey;
          gz = df * ez;
        }

        // -------------------------------------------------
        // NU1_DOT
        // D = M1[a] dot M1[b]
        // -------------------------------------------------

        else if (spec.type == NU1_DOT) {

          int a = spec.mu[0];
          int b2 = spec.mu[1];

          double va[3] = {
            M1[a*3 + 0],
            M1[a*3 + 1],
            M1[a*3 + 2]
          };

          double vb[3] = {
            M1[b2*3 + 0],
            M1[b2*3 + 1],
            M1[b2*3 + 2]
          };

          double ra[3] = {dx, dy, dz};

          for (int c = 0; c < 3; c++) {

            double dMa =
              dfmu[a] * ra[c] * ex
              + (c == 0 ? fmu[a] : 0.0);

            double dMb =
              dfmu[b2] * ra[c] * ex
              + (c == 0 ? fmu[b2] : 0.0);

            if (c == 1) {

              dMa =
                dfmu[a] * ra[c] * ey
                + fmu[a];

              dMb =
                dfmu[b2] * ra[c] * ey
                + fmu[b2];
            }

            if (c == 2) {

              dMa =
                dfmu[a] * ra[c] * ez
                + fmu[a];

              dMb =
                dfmu[b2] * ra[c] * ez
                + fmu[b2];
            }

            double g =
              dMa * vb[c]
              + va[c] * dMb;

            if (c == 0) gx += g;
            if (c == 1) gy += g;
            if (c == 2) gz += g;
          }
        }

        // -------------------------------------------------
        // NU2_FROB
        // D = M2[a] : M2[b]
        // -------------------------------------------------

        else if (spec.type == NU2_FROB) {

          int a = spec.mu[0];
          int b2 = spec.mu[1];

          double rvec[3] = {dx, dy, dz};

          for (int alpha = 0; alpha < 3; alpha++) {

            double grad = 0.0;

            for (int p = 0; p < 3; p++) {
              for (int q = 0; q < 3; q++) {

                int idx = p*3 + q;

                double Mab =
                  M2[b2*9 + idx];

                double Maa =
                  M2[a*9 + idx];

                double rp = rvec[p];
                double rq = rvec[q];

                double rc =
                  (alpha == 0 ? ex :
                   alpha == 1 ? ey : ez);

                double dA =
                  dfmu[a] * rp * rq * rc;

                double dB =
                  dfmu[b2] * rp * rq * rc;

                if (alpha == p)
                  dA += fmu[a] * rq;

                if (alpha == q)
                  dA += fmu[a] * rp;

                if (alpha == p)
                  dB += fmu[b2] * rq;

                if (alpha == q)
                  dB += fmu[b2] * rp;

                grad +=
                  dA * Mab
                  + Maa * dB;
              }
            }

            if (alpha == 0) gx = grad;
            if (alpha == 1) gy = grad;
            if (alpha == 2) gz = grad;
          }
        }

        data->graddesc[pair_index][k][0] = gx;
        data->graddesc[pair_index][k][1] = gy;
        data->graddesc[pair_index][k][2] = gz;
      }
    }
  }
}

void MLIAPDescriptorMTP::init() {}
void MLIAPDescriptorMTP::compute_forces(class MLIAPData *) {}
void MLIAPDescriptorMTP::compute_force_gradients(class MLIAPData *) {}


//#endif
