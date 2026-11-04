//#ifdef MLIAP_MTP

#include "mliap_descriptor_mtp_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "mliap_data_kokkos.h"
#include "pair_mliap.h"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>

using namespace LAMMPS_NS;

#include <type_traits>

/* ---------------------------------------------------------------------- */

template <class DeviceType>
MLIAPDescriptorMTPKokkos<DeviceType>::MLIAPDescriptorMTPKokkos(LAMMPS *lmp)
  : Pointers(lmp),
    MLIAPDescriptorMTP(lmp),
    MLIAPDescriptorKokkos<DeviceType>(lmp, this)
{}

template <class DeviceType>
void test_mtp_conversion(MLIAPDescriptorMTPKokkos<DeviceType> *p)
{
  auto *m = static_cast<MLIAPDescriptorMTP *>(p);
  auto *d = static_cast<MLIAPDescriptor *>(m);
}

template <class DeviceType>
void MLIAPDescriptorMTPKokkos<DeviceType>::init()
{
  MLIAPDescriptorMTP::init();

  k_radial_coeffs =
      Kokkos::View<double ****, DeviceType>("mtp_radial_coeffs",
                                             n_rf, nelements, nelements,
                                             n_radial);

  k_specs_type =
      Kokkos::View<int *, DeviceType>("mtp_specs_type", basis_specs.size());

  k_specs_mu =
      Kokkos::View<int **, DeviceType>("mtp_specs_mu", basis_specs.size(), 3);

  auto h_radial_coeffs =
      Kokkos::create_mirror_view(k_radial_coeffs);

  for (int mu = 0; mu < n_rf; ++mu)
    for (int itype = 0; itype < nelements; ++itype)
      for (int jtype = 0; jtype < nelements; ++jtype)
        for (int n = 0; n < n_radial; ++n)
          h_radial_coeffs(mu, itype, jtype, n) =
              radial_coeffs[mu][itype][jtype][n];

  Kokkos::deep_copy(k_radial_coeffs, h_radial_coeffs);

  auto h_specs_type = Kokkos::create_mirror_view(k_specs_type);
  auto h_specs_mu = Kokkos::create_mirror_view(k_specs_mu);

  for (std::size_t b = 0; b < basis_specs.size(); ++b) {
    h_specs_type(b) = basis_specs[b].type;
    h_specs_mu(b, 0) = basis_specs[b].mu[0];
    h_specs_mu(b, 1) = basis_specs[b].mu[1];
    h_specs_mu(b, 2) = basis_specs[b].mu[2];
  }

  Kokkos::deep_copy(k_specs_type, h_specs_type);
  Kokkos::deep_copy(k_specs_mu, h_specs_mu);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
MLIAPDescriptorMTPKokkos<DeviceType>::~MLIAPDescriptorMTPKokkos()
{
  k_radial_coeffs = decltype(k_radial_coeffs)();
  k_specs_type = decltype(k_specs_type)();
  k_specs_mu = decltype(k_specs_mu)();
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
double MLIAPDescriptorMTPKokkosDeviceCutoff(double r, double cutoff,
                                             double rmin)
{
  if (r >= cutoff)
    return 0.0;

  double x = (cutoff - r) / (cutoff - rmin);
  if (x < 0.0)
    x = 0.0;

  return x * x;
}

/* ---------------------------------------------------------------------- */

KOKKOS_INLINE_FUNCTION
void MLIAPDescriptorMTPKokkosChebyshev(
    double r, double cutoff, double rmin, int n_radial,
    double *basis, double *dbasisdr)
{
  const double denom = cutoff - rmin + 1.0e-10;

  double x = (2.0 * r - rmin - cutoff) / denom;
  if (x < -1.0)
    x = -1.0;
  else if (x > 1.0)
    x = 1.0;

  const double dxdr = 2.0 / denom;
  const double fc =
      MLIAPDescriptorMTPKokkosDeviceCutoff(r, cutoff, rmin);

  double dfcdr = 0.0;
  if (r < cutoff) {
    const double tmp = (cutoff - r) / (cutoff - rmin);
    if (tmp > 0.0)
      dfcdr = -2.0 * tmp / (cutoff - rmin);
  }

  double T[MLIAPDescriptorMTP::MAX_N_RADIAL];
  double dTdx[MLIAPDescriptorMTP::MAX_N_RADIAL];

  T[0] = 1.0;
  dTdx[0] = 0.0;

  if (n_radial >= 2) {
    T[1] = x;
    dTdx[1] = 1.0;
  }

  for (int n = 2; n < n_radial; ++n) {
    T[n] = 2.0 * x * T[n - 1] - T[n - 2];
    dTdx[n] =
        2.0 * T[n - 1] + 2.0 * x * dTdx[n - 1] - dTdx[n - 2];
  }

  for (int n = 0; n < n_radial; ++n) {
    basis[n] = T[n] * fc;
    dbasisdr[n] = dTdx[n] * dxdr * fc + T[n] * dfcdr;
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDescriptorMTPKokkos<DeviceType>::compute_descriptors(
    class MLIAPData *data_)
{
  auto data = static_cast<MLIAPDataKokkos<DeviceType> *>(data_);

  const int nlistatoms = data->nlistatoms;
  const int nrf = n_rf;
  const int nelem = nelements;
  const int maxnu = max_nu;
  const int nbasis = static_cast<int>(basis_specs.size());
  const int ndesc = ndescriptors;

  auto d_iatoms = data->k_iatoms.template view<DeviceType>();
  auto d_numneighs = data->k_numneighs.template view<DeviceType>();
  auto d_jatoms = data->k_jatoms.template view<DeviceType>();
  auto d_ij = data->k_ij.template view<DeviceType>();
  auto d_rij = data->k_rij.template view<DeviceType>();
  auto d_type = atomKK->k_type.template view<DeviceType>();
  auto d_desc = data->k_descriptors.template view<DeviceType>();

  auto d_coeff = k_radial_coeffs;
  auto d_specs_type = k_specs_type;
  auto d_specs_mu = k_specs_mu;

  const double d_cutoff = cutoff;
  const double d_rmin = rmin;
  const int d_nradial = n_radial;

  Kokkos::parallel_for(
      "MLIAPDescriptorMTPKokkos::compute_descriptors",
      Kokkos::RangePolicy<DeviceType>(0, nlistatoms),
      KOKKOS_LAMBDA(const int ii) {
        double m0[MAX_N_RF] = {};
        double m1[MAX_N_RF * 3] = {};
        double m2[MAX_N_RF * 9] = {};

        const int i = d_iatoms(ii);
        const int jnum = d_numneighs(ii);
        int ij = d_ij(ii);

        const int itype = d_type(i) - 1;

        for (int jj = 0; jj < jnum; ++jj, ++ij) {
          const int j = d_jatoms(ij);
          const int jtype = d_type(j) - 1;

          const double dx = d_rij(ij, 0);
          const double dy = d_rij(ij, 1);
          const double dz = d_rij(ij, 2);
          const double r =
              Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

          if (r >= d_cutoff)
            continue;

          double basis[MAX_N_RADIAL];
          double dbasisdr[MAX_N_RADIAL];

          MLIAPDescriptorMTPKokkosChebyshev(
              r, d_cutoff, d_rmin, d_nradial, basis, dbasisdr);

          for (int mu = 0; mu < nrf; ++mu) {
            double f = 0.0;

            for (int n = 0; n < d_nradial; ++n)
              f += d_coeff(mu, itype, jtype, n) * basis[n];

            m0[mu] += f;

            m1[mu * 3 + 0] += f * dx;
            m1[mu * 3 + 1] += f * dy;
            m1[mu * 3 + 2] += f * dz;

            if (maxnu >= 2) {
              m2[mu * 9 + 0] += f * dx * dx;
              m2[mu * 9 + 1] += f * dx * dy;
              m2[mu * 9 + 2] += f * dx * dz;
              m2[mu * 9 + 3] += f * dy * dx;
              m2[mu * 9 + 4] += f * dy * dy;
              m2[mu * 9 + 5] += f * dy * dz;
              m2[mu * 9 + 6] += f * dz * dx;
              m2[mu * 9 + 7] += f * dz * dy;
              m2[mu * 9 + 8] += f * dz * dz;
            }
          }
        }

        int k = 0;

        for (int s = 0; s < nelem; ++s)
          d_desc(ii, k++) = (itype == s) ? 1.0 : 0.0;

        for (int b = 0; b < nbasis; ++b) {
          const int spec_type = d_specs_type(b);
          const int mu0 = d_specs_mu(b, 0);
          const int mu1 = d_specs_mu(b, 1);
          const int mu2 = d_specs_mu(b, 2);

          double val = 0.0;

          if (spec_type == NU0) {
            val = m0[mu0];

          } else if (spec_type == NU1_DOT) {
            val =
                m1[mu0 * 3 + 0] * m1[mu1 * 3 + 0] +
                m1[mu0 * 3 + 1] * m1[mu1 * 3 + 1] +
                m1[mu0 * 3 + 2] * m1[mu1 * 3 + 2];

          } else if (spec_type == NU2_FROB) {
            for (int q = 0; q < 9; ++q)
              val += m2[mu0 * 9 + q] * m2[mu1 * 9 + q];

          } else if (spec_type == NU0_X_NU1SQ) {
            const double norm2 =
                m1[mu1 * 3 + 0] * m1[mu1 * 3 + 0] +
                m1[mu1 * 3 + 1] * m1[mu1 * 3 + 1] +
                m1[mu1 * 3 + 2] * m1[mu1 * 3 + 2];

            val = m0[mu0] * norm2;

          } else if (spec_type == NU0_X_NU1_NU1) {
            const double dot =
                m1[mu1 * 3 + 0] * m1[mu2 * 3 + 0] +
                m1[mu1 * 3 + 1] * m1[mu2 * 3 + 1] +
                m1[mu1 * 3 + 2] * m1[mu2 * 3 + 2];

            val = m0[mu0] * dot;

          } else if (spec_type == V_T_V) {
            double Tv[3];

            Tv[0] =
                m1[mu0 * 3 + 0] * m2[mu1 * 9 + 0] +
                m1[mu0 * 3 + 1] * m2[mu1 * 9 + 3] +
                m1[mu0 * 3 + 2] * m2[mu1 * 9 + 6];

            Tv[1] =
                m1[mu0 * 3 + 0] * m2[mu1 * 9 + 1] +
                m1[mu0 * 3 + 1] * m2[mu1 * 9 + 4] +
                m1[mu0 * 3 + 2] * m2[mu1 * 9 + 7];

            Tv[2] =
                m1[mu0 * 3 + 0] * m2[mu1 * 9 + 2] +
                m1[mu0 * 3 + 1] * m2[mu1 * 9 + 5] +
                m1[mu0 * 3 + 2] * m2[mu1 * 9 + 8];

            val =
                Tv[0] * m1[mu2 * 3 + 0] +
                Tv[1] * m1[mu2 * 3 + 1] +
                Tv[2] * m1[mu2 * 3 + 2];

          } else if (spec_type == NU0_X_NU0) {
            val = m0[mu0] * m0[mu1];

          } else if (spec_type == NU0_SQ) {
            val = m0[mu0] * m0[mu0];
          }

          d_desc(ii, k++) = val;
        }
      });

  Kokkos::fence();

  Kokkos::deep_copy(data->k_descriptors.view_host(), d_desc);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDescriptorMTPKokkos<DeviceType>::compute_forces(
    class MLIAPData *data_)
{
  auto data = static_cast<MLIAPDataKokkos<DeviceType> *>(data_);
  compute_forces_from_coeffs(data, data->k_betas);
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDescriptorMTPKokkos<DeviceType>::compute_forces_from_coeffs(
    MLIAPDataKokkos<DeviceType> *data,
    DAT::tdual_double_2d_lr &coeffs,
    typename AT::t_double_1d phi)
{
  const int nlistatoms = data->nlistatoms;
  const int nelem = nelements;
  const int maxnu = max_nu;
  const int nbasis = static_cast<int>(basis_specs.size());
  const int ndesc = ndescriptors;

  auto d_iatoms = data->k_iatoms.template view<DeviceType>();
  auto d_numneighs = data->k_numneighs.template view<DeviceType>();
  auto d_jatoms = data->k_jatoms.template view<DeviceType>();
  auto d_ij = data->k_ij.template view<DeviceType>();
  auto d_rij = data->k_rij.template view<DeviceType>();
  auto d_type = atomKK->k_type.template view<DeviceType>();

  auto d_beta = data->k_betas.template view<DeviceType>();
  auto d_f = atomKK->k_f.template view<DeviceType>();

  auto d_coeff = k_radial_coeffs;
  auto d_specs_type = k_specs_type;
  auto d_specs_mu = k_specs_mu;

  const double d_cutoff = cutoff;
  const double d_rmin = rmin;
  const int d_nradial = n_radial;
  const int d_n_rf = MAX_N_RF;

  const int vflag = data->vflag;
  const int vflag_global = data->pairmliap->vflag_global;
  const int vflag_atom = data->pairmliap->vflag_atom;

  auto d_vatom =
      data->k_pairmliap->k_vatom.template view<DeviceType>();

  Kokkos::View<double[6], DeviceType> virial("mtp_virial");

  if (vflag_atom) {
    data->k_pairmliap->k_vatom.modify_host();
    data->k_pairmliap->k_vatom.template sync<DeviceType>();
  }

  auto d_force_coeffs = coeffs.template view<DeviceType>();

  /*
   * The force contribution for a central atom is accumulated independently
   * for each neighbor-list atom.  The final i/j updates are atomic because
   * an atom can occur in multiple central-atom neighbor lists.
   */
  Kokkos::parallel_for(
      "MLIAPDescriptorMTPKokkos::compute_forces",
      Kokkos::RangePolicy<DeviceType>(0, nlistatoms),
      KOKKOS_LAMBDA(const int ii) {
        double m0[d_n_rf] = {};
        double m1[d_n_rf * 3] = {};
        double m2[d_n_rf * 9] = {};

        const int i = d_iatoms(ii);
        const int jnum = d_numneighs(ii);
        const int itype = d_type(i) - 1;

        int ij0 = d_ij(ii);

        /*
         * First construct the moment tensors for atom i.
         */
        for (int jj = 0; jj < jnum; ++jj) {
          const int ij = ij0 + jj;
          const int j = d_jatoms(ij);
          const int jtype = d_type(j) - 1;

          const double dx = d_rij(ij, 0);
          const double dy = d_rij(ij, 1);
          const double dz = d_rij(ij, 2);

          const double r =
              Kokkos::sqrt(dx * dx + dy * dy + dz * dz);

          if (r >= d_cutoff)
            continue;

          double basis[MAX_N_RADIAL];
          double dbasisdr[MAX_N_RADIAL];

          MLIAPDescriptorMTPKokkosChebyshev(
              r, d_cutoff, d_rmin, d_nradial,
              basis, dbasisdr);

          for (int mu = 0; mu < d_n_rf; ++mu) {
            double f = 0.0;

            for (int n = 0; n < d_nradial; ++n)
              f += d_coeff(mu, itype, jtype, n) * basis[n];

            m0[mu] += f;

            m1[mu * 3 + 0] += f * dx;
            m1[mu * 3 + 1] += f * dy;
            m1[mu * 3 + 2] += f * dz;

            if (maxnu >= 2) {
              m2[mu * 9 + 0] += f * dx * dx;
              m2[mu * 9 + 1] += f * dx * dy;
              m2[mu * 9 + 2] += f * dx * dz;
              m2[mu * 9 + 3] += f * dy * dx;
              m2[mu * 9 + 4] += f * dy * dy;
              m2[mu * 9 + 5] += f * dy * dz;
              m2[mu * 9 + 6] += f * dz * dx;
              m2[mu * 9 + 7] += f * dz * dy;
              m2[mu * 9 + 8] += f * dz * dz;
            }
          }
        }

        /*
         * Each descriptor is differentiated with respect to the current
         * neighbor vector r_ij.  This gives the pair force contribution
         * associated with this descriptor.
         */
        for (int jj = 0; jj < jnum; ++jj) {
          const int ij = ij0 + jj;
          const int j = d_jatoms(ij);
          const int jtype = d_type(j) - 1;

          const double dx = d_rij(ij, 0);
          const double dy = d_rij(ij, 1);
          const double dz = d_rij(ij, 2);

          const double rsq = dx * dx + dy * dy + dz * dz;
          const double r = Kokkos::sqrt(rsq);

          if (r >= d_cutoff)
            continue;

          const double rinv = 1.0 / (r + 1.0e-20);
          const double ex = dx * rinv;
          const double ey = dy * rinv;
          const double ez = dz * rinv;

          double basis[MAX_N_RADIAL];
          double dbasisdr[MAX_N_RADIAL];

          MLIAPDescriptorMTPKokkosChebyshev(
              r, d_cutoff, d_rmin, d_nradial,
              basis, dbasisdr);

          double fmu[MAX_N_RF];
          double dfmu[MAX_N_RF];

          for (int mu = 0; mu < d_n_rf; ++mu) {
            double f = 0.0;
            double df = 0.0;

            for (int n = 0; n < d_nradial; ++n) {
              const double c =
                  d_coeff(mu, itype, jtype, n);

              f += c * basis[n];
              df += c * dbasisdr[n];
            }

            fmu[mu] = f;
            dfmu[mu] = df;
          }

          double fij[3] = {0.0, 0.0, 0.0};

          for (int b = 0; b < nbasis; ++b) {
            const int spec_type = d_specs_type(b);
            const int mu0 = d_specs_mu(b, 0);
            const int mu1 = d_specs_mu(b, 1);
            const int mu2 = d_specs_mu(b, 2);

            double gx = 0.0;
            double gy = 0.0;
            double gz = 0.0;

            if (spec_type == NU0) {

              gx = dfmu[mu0] * ex;
              gy = dfmu[mu0] * ey;
              gz = dfmu[mu0] * ez;

            } else if (spec_type == NU1_DOT) {

              const double va[3] = {
                  m1[mu0 * 3 + 0],
                  m1[mu0 * 3 + 1],
                  m1[mu0 * 3 + 2]};

              const double vb[3] = {
                  m1[mu1 * 3 + 0],
                  m1[mu1 * 3 + 1],
                  m1[mu1 * 3 + 2]};

              const double rv[3] = {dx, dy, dz};
              const double ev[3] = {ex, ey, ez};

              double grad[3] = {0.0, 0.0, 0.0};

              for (int alpha = 0; alpha < 3; ++alpha) {
                for (int beta = 0; beta < 3; ++beta) {
                  const double dMa =
                      dfmu[mu0] * ev[alpha] * rv[beta] +
                      ((alpha == beta) ? fmu[mu0] : 0.0);

                  const double dMb =
                      dfmu[mu1] * ev[alpha] * rv[beta] +
                      ((alpha == beta) ? fmu[mu1] : 0.0);

                  grad[alpha] +=
                      dMa * vb[beta] + va[beta] * dMb;
                }
              }

              gx = grad[0];
              gy = grad[1];
              gz = grad[2];

            } else if (spec_type == NU2_FROB) {

              const double rv[3] = {dx, dy, dz};
              const double ev[3] = {ex, ey, ez};

              double grad[3] = {0.0, 0.0, 0.0};

              for (int alpha = 0; alpha < 3; ++alpha) {
                for (int p = 0; p < 3; ++p) {
                  for (int q = 0; q < 3; ++q) {
                    const int idx = p * 3 + q;

                    const double Mab =
                        m2[mu1 * 9 + idx];
                    const double Maa =
                        m2[mu0 * 9 + idx];

                    const double rp = rv[p];
                    const double rq = rv[q];

                    double dA =
                        dfmu[mu0] *
                        ev[alpha] * rp * rq;

                    double dB =
                        dfmu[mu1] *
                        ev[alpha] * rp * rq;

                    if (alpha == p) {
                      dA += fmu[mu0] * rq;
                      dB += fmu[mu1] * rq;
                    }

                    if (alpha == q) {
                      dA += fmu[mu0] * rp;
                      dB += fmu[mu1] * rp;
                    }

                    grad[alpha] +=
                        dA * Mab + Maa * dB;
                  }
                }
              }

              gx = grad[0];
              gy = grad[1];
              gz = grad[2];

            } else if (spec_type == NU0_X_NU1SQ) {

              const double v[3] = {
                  m1[mu1 * 3 + 0],
                  m1[mu1 * 3 + 1],
                  m1[mu1 * 3 + 2]};

              const double rv[3] = {dx, dy, dz};
              const double ev[3] = {ex, ey, ez};

              const double norm2 =
                  v[0] * v[0] +
                  v[1] * v[1] +
                  v[2] * v[2];

              double grad[3] = {0.0, 0.0, 0.0};

              for (int alpha = 0; alpha < 3; ++alpha) {
                const double dM0 =
                    dfmu[mu0] * ev[alpha];

                double dnorm2 = 0.0;

                for (int beta = 0; beta < 3; ++beta) {
                  const double dM1 =
                      dfmu[mu1] *
                          ev[alpha] * rv[beta] +
                      ((alpha == beta) ?
                           fmu[mu1] : 0.0);

                  dnorm2 +=
                      2.0 * v[beta] * dM1;
                }

                grad[alpha] =
                    dM0 * norm2 +
                    m0[mu0] * dnorm2;
              }

              gx = grad[0];
              gy = grad[1];
              gz = grad[2];

            } else if (spec_type == NU0_X_NU1_NU1) {

              const double A[3] = {
                  m1[mu1 * 3 + 0],
                  m1[mu1 * 3 + 1],
                  m1[mu1 * 3 + 2]};

              const double B[3] = {
                  m1[mu2 * 3 + 0],
                  m1[mu2 * 3 + 1],
                  m1[mu2 * 3 + 2]};

              const double rv[3] = {dx, dy, dz};
              const double ev[3] = {ex, ey, ez};

              const double dot =
                  A[0] * B[0] +
                  A[1] * B[1] +
                  A[2] * B[2];

              double grad[3] = {0.0, 0.0, 0.0};

              for (int alpha = 0; alpha < 3; ++alpha) {
                const double dM0 =
                    dfmu[mu0] * ev[alpha];

                double ddot = 0.0;

                for (int beta = 0; beta < 3; ++beta) {
                  const double dA =
                      dfmu[mu1] *
                          ev[alpha] * rv[beta] +
                      ((alpha == beta) ?
                           fmu[mu1] : 0.0);

                  const double dB =
                      dfmu[mu2] *
                          ev[alpha] * rv[beta] +
                      ((alpha == beta) ?
                           fmu[mu2] : 0.0);

                  ddot +=
                      dA * B[beta] +
                      A[beta] * dB;
                }

                grad[alpha] =
                    dM0 * dot +
                    m0[mu0] * ddot;
              }

              gx = grad[0];
              gy = grad[1];
              gz = grad[2];

            } else if (spec_type == V_T_V) {

              const double L[3] = {
                  m1[mu0 * 3 + 0],
                  m1[mu0 * 3 + 1],
                  m1[mu0 * 3 + 2]};

              const double R[3] = {
                  m1[mu2 * 3 + 0],
                  m1[mu2 * 3 + 1],
                  m1[mu2 * 3 + 2]};

              const double rv[3] = {dx, dy, dz};
              const double ev[3] = {ex, ey, ez};

              double grad[3] = {0.0, 0.0, 0.0};

              for (int alpha = 0; alpha < 3; ++alpha) {
                double g = 0.0;

                for (int p = 0; p < 3; ++p) {
                  for (int q = 0; q < 3; ++q) {
                    const int idx = p * 3 + q;

                    const double M =
                        m2[mu1 * 9 + idx];

                    double dL =
                        dfmu[mu0] *
                        ev[alpha] * rv[p];

                    if (alpha == p)
                      dL += fmu[mu0];

                    double dR =
                        dfmu[mu2] *
                        ev[alpha] * rv[q];

                    if (alpha == q)
                      dR += fmu[mu2];

                    double dM =
                        dfmu[mu1] *
                        ev[alpha] *
                        rv[p] * rv[q];

                    if (alpha == p)
                      dM += fmu[mu1] * rv[q];

                    if (alpha == q)
                      dM += fmu[mu1] * rv[p];

                    g +=
                        dL * M * R[q] +
                        L[p] * dM * R[q] +
                        L[p] * M * dR;
                  }
                }

                grad[alpha] = g;
              }

              gx = grad[0];
              gy = grad[1];
              gz = grad[2];

            } else if (spec_type == NU0_X_NU0) {

              gx =
                  dfmu[mu0] * ex * m0[mu1] +
                  m0[mu0] * dfmu[mu1] * ex;

              gy =
                  dfmu[mu0] * ey * m0[mu1] +
                  m0[mu0] * dfmu[mu1] * ey;

              gz =
                  dfmu[mu0] * ez * m0[mu1] +
                  m0[mu0] * dfmu[mu1] * ez;

            } else if (spec_type == NU0_SQ) {

              gx =
                  2.0 * m0[mu0] *
                  dfmu[mu0] * ex;

              gy =
                  2.0 * m0[mu0] *
                  dfmu[mu0] * ey;

              gz =
                  2.0 * m0[mu0] *
                  dfmu[mu0] * ez;
            }

            const double coeff = d_force_coeffs(ii, nelem + b);

            fij[0] += coeff * gx;
            fij[1] += coeff * gy;
            fij[2] += coeff * gz;
          }

          /*
           * fij is the derivative contribution with respect to r_ij.
           * The force on i is +fij and the force on j is -fij.
           */
          Kokkos::atomic_add(&d_f(i, 0), fij[0]);
          Kokkos::atomic_add(&d_f(i, 1), fij[1]);
          Kokkos::atomic_add(&d_f(i, 2), fij[2]);

          Kokkos::atomic_add(&d_f(j, 0), -fij[0]);
          Kokkos::atomic_add(&d_f(j, 1), -fij[1]);
          Kokkos::atomic_add(&d_f(j, 2), -fij[2]);

          if (vflag) {
            const double v0 = -dx * fij[0];
            const double v1 = -dy * fij[1];
            const double v2 = -dz * fij[2];
            const double v3 = -dx * fij[1];
            const double v4 = -dx * fij[2];
            const double v5 = -dy * fij[2];

            if (vflag_global) {
              Kokkos::atomic_add(&virial[0], v0);
              Kokkos::atomic_add(&virial[1], v1);
              Kokkos::atomic_add(&virial[2], v2);
              Kokkos::atomic_add(&virial[3], v3);
              Kokkos::atomic_add(&virial[4], v4);
              Kokkos::atomic_add(&virial[5], v5);
            }

            if (vflag_atom) {
              Kokkos::atomic_add(&d_vatom(i, 0), 0.5 * v0);
              Kokkos::atomic_add(&d_vatom(i, 1), 0.5 * v1);
              Kokkos::atomic_add(&d_vatom(i, 2), 0.5 * v2);
              Kokkos::atomic_add(&d_vatom(i, 3), 0.5 * v3);
              Kokkos::atomic_add(&d_vatom(i, 4), 0.5 * v4);
              Kokkos::atomic_add(&d_vatom(i, 5), 0.5 * v5);

              Kokkos::atomic_add(&d_vatom(j, 0), 0.5 * v0);
              Kokkos::atomic_add(&d_vatom(j, 1), 0.5 * v1);
              Kokkos::atomic_add(&d_vatom(j, 2), 0.5 * v2);
              Kokkos::atomic_add(&d_vatom(j, 3), 0.5 * v3);
              Kokkos::atomic_add(&d_vatom(j, 4), 0.5 * v4);
              Kokkos::atomic_add(&d_vatom(j, 5), 0.5 * v5);
            }
          }
        }
      });

  Kokkos::fence();

  if (vflag) {
    if (vflag_global) {
      Kokkos::View<double[6], LMPHostType> h_virial("mtp_h_virial");
      Kokkos::deep_copy(h_virial, virial);

      for (int i = 0; i < 6; ++i)
        data->k_pairmliap->virial[i] += h_virial[i];
    }

    if (vflag_atom) {
      data->k_pairmliap->k_vatom.template modify<DeviceType>();
      data->k_pairmliap->k_vatom.sync_host();
    }
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDescriptorMTPKokkos<DeviceType>::compute_descriptor_gradients(
    class MLIAPData *data_) {}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
void MLIAPDescriptorMTPKokkos<DeviceType>::compute_force_gradients(
    class MLIAPData *data_) {}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
double MLIAPDescriptorMTPKokkos<DeviceType>::memory_usage() {}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {

template class MLIAPDescriptorMTPKokkos<LMPDeviceType>;

#ifdef LMP_KOKKOS_GPU
template class MLIAPDescriptorMTPKokkos<LMPHostType>;
#endif

}

//#endif
