.. index:: fix baoab/molblock

fix baoab/molblock command
==========================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID baoab/molblock Tstart Tstop damp seed keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* baoab/molblock = style name of this fix command
* Tstart, Tstop = desired temperature at start/end of run (temperature units)
* damp = damping parameter (time units)
* seed = random number seed (positive integer)
* the *theta* and *refresh* keywords must be given
* keyword = *theta* or *refresh* or *heff_samples* or *gamma_soft* or *eps* or *inner* or *verbose*

  .. parsed-literal::

       *theta* value = dimensionless stiffness threshold, must be > 0: a
         molecule is treated as stiff if a vibration has frequency times
         timestep above *theta*
       *refresh* value = steps between force-constant refreshes, must be > 0
       *heff_samples* value = N, must be >= 0 (0 = measure the force
         constants by finite differences at every refresh; N > 0 = estimate
         them from the trajectory, averaged with weight 1/N per refresh)
       *gamma_soft* value = thermostat friction (1/time units), must be >= 0
       *eps* value = finite-difference displacement (distance units), must be > 0
       *inner* value = number of inner substeps per timestep, must be >= 0
         (0 selects the number automatically)
       *verbose* value = *yes* or *no*

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 all baoab/molblock 300.0 300.0 1000.0 12345 theta 0.2 refresh 10 &
         heff_samples 12 gamma_soft 0.001

Description
"""""""""""

.. versionadded:: TBD

This fix performs Langevin dynamics with a multiple-time-step integrator for
small, stiff, bonded molecules, such as flexible water. A molecule here is
an atom together with its bond and angle partners, at most 5 atoms in total.
At every refresh the fix assembles the full second-derivative matrix of each
molecule from central finite-difference force probes of each atom
(displacement *eps*, by default :math:`10^{-4}` times the neighbor skin). If
any internal vibration of a molecule is faster than the *theta* threshold
allows, every bond length and bond angle of that molecule is included in a
"fast" harmonic model, :math:`V_{\rm fast} = \frac{1}{2}(s - s_0)^T F (s -
s_0)`, where *s* holds the bond lengths and angles. The force constants *F*
are extracted from the measured second derivatives and forces (the Wilson GF
method), so they follow whatever force field or machine-learned potential is
in use, including the influence of neighboring molecules; no fast/slow split
of the force field itself has to be specified. The reference values
:math:`s_0` are set once, at the molecule's equilibrium position in its
environment at the first step, and are kept for the rest of the run.

A bond angle larger than 150 degrees at a probe refresh is treated as the
bend of a linear group (for example carbon dioxide).  A single angle cannot
describe such a bend, which can happen in two perpendicular directions and
has no well-defined direction of change at 180 degrees, so it is replaced
by the two components of the displacement of the middle atom from the line
through the two outer atoms.  These two coordinates share one force
constant and are not coupled to the bond lengths, and their reference
value is zero (all exact by symmetry for a linear equilibrium structure),
so the fast energy does not depend on how the two perpendicular directions
are chosen.

Each timestep then applies half a kick from the remaining ("slow") force,
the full force minus the fast model force, followed by several short
velocity-Verlet substeps that use only the fast model force, which needs no
force evaluations, and ends with another slow half kick. The Langevin
thermostat, with rate *gamma_soft* (default 1/*damp*), is applied once per
timestep in the middle of the substeps. Atoms that are not part of a stiff
molecule simply drift during the substeps. When the force constants are
refreshed, the timestep in progress completes with the old constants and the
new constants take effect at the start of the next timestep, so that every
timestep is a consistent splitting of the same total force. Only one force
evaluation is needed per timestep, as in plain velocity Verlet, plus the
curvature probes at each refresh.

How the force constants are updated after the first step depends on
*heff_samples*. With *heff_samples 0* (the default), they are measured again
from finite-difference probes every *refresh* steps, which gives the
instantaneous curvature at that moment but costs roughly as much as 15 to 20
ordinary timesteps per refresh. With *heff_samples* N > 0, the probes are
used only at the first step. Afterwards the force constants are estimated
from the trajectory itself, at no extra cost: from one timestep to the next,
the change in the force on the bond lengths and angles is very nearly the
force constants times the change in those coordinates, because the vibration
dominates both changes while the slow drift of the molecule's surroundings
largely cancels. These step-to-step changes are accumulated over each block
of *refresh* steps, blended into a running average in which each new block
has weight 1/N (with the probed values as the starting point), and the
resulting force constants are used from the next timestep on. This is a
thermally averaged effective curvature, in the same spirit as *heff_samples*
for the per-atom treatment of :doc:`fix baoab/tether <fix_baoab_tether>`. An
estimate that is not physically sensible (not positive definite) is
discarded and the previous force constants are kept. For flexible SPC/Fw
water, *refresh 10* with *heff_samples 12* gave force constants matching the
probed values to within their thermal fluctuation (about 1.5 percent), the
same accuracy in all checks listed below as probing every 10 steps, and
about twice the speed; a shorter memory (*heff_samples 4*) was stable but
conserved energy slightly less well.

The *inner* keyword sets the number of substeps.  With the
default of 0 the fix chooses, at every refresh, the smallest even number
(at least 2) that keeps the fastest fast-model frequency times the substep
length at or below 0.3.  For flexible SPC/Fw water in real units this gives
6 substeps at a 2 fs timestep.  In tests on flexible SPC/Fw water at 300 K,
a 2 fs timestep reproduced the temperature, the stretch and bend peaks of
the velocity spectrum, the radial distribution functions, and the diffusion
coefficient of a 0.5 fs velocity-Verlet reference within statistical
uncertainty; the librational band of the spectrum is about 10 percent
weaker relative to the stretch peak.  At 4 fs the peaks and the structure
are still reproduced, but the temperature reads about 5 K low and the
diffusion coefficient is about 12 percent low.  With the thermostat off, the
total energy drift at 1 and 2 fs is comparable to that of :doc:`run_style
respa <run_style>` with the bonded terms on the inner level, and somewhat
larger at 4 fs.  For flexible carbon dioxide (liquid, 250 K) and nitrogen
(liquid, 77 K), timesteps of 2, 3 and 4 fs reproduced the temperature, all
vibrational peaks (for carbon dioxide including the bend), the radial
distribution functions and the diffusion coefficient of a 0.5 fs
velocity-Verlet reference within the statistical uncertainty of 20 ps runs
with three random seeds.

The *verbose* keyword prints, at every refresh, the number of
molecules and fast coordinates (or, with *heff_samples* > 0, the number
of molecules whose force constants were updated or kept), the fastest
frequency times the timestep, and the number of substeps, together with
the force constants of one molecule at the first refresh and, with
*heff_samples* > 0, the average diagonal force constants.

This fix requires a molecular atom style and :doc:`newton <newton>` bond
setting *off* (the bond and angle lists of every atom must hold all bonds
and angles it takes part in); the timestep stays fixed.  Molecules
are found from the bond and angle lists, so the bond and angle styles must
be able to report their atoms to this fix; with a machine-learned potential
that supplies all forces, use *bond_style harmonic* and *angle_style
harmonic* with zero force constants rather than the *zero* styles.  Every
atom of a molecule must be linked to every other one by a bond or an angle
(water, diatomics, linear triatomics such as carbon dioxide, ammonia-like
molecules), and the bonds and angles of a molecule must be independent of
each other, which excludes molecules with rings; the fix stops with an error
otherwise.

This fix runs in parallel without any communication during the
substeps.  Each processor integrates its own copy of every molecule that
has at least one atom it owns, using ghost atoms for the other atoms of the
molecule, and keeps the result only for its owned atoms; velocities and
forces of ghost atoms are communicated once per timestep, before the
substeps.  The thermostat noise of each atom is generated from the random
seed, the atom ID and the timestep, so all copies of a molecule draw the
same numbers and results do not depend on the number of processors, apart
from round-off.  Each molecule's force constants and averages are stored
with its lowest-ID atom and move with it between processors.  All atoms of
a molecule must be within the ghost-atom cutoff of each other, which holds
automatically for small molecules with the usual pair-style cutoffs.

Restart, fix_modify, output, run start/stop, minimize info
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`; the force constants are measured again at the start of each
run.  None of the :doc:`fix_modify <fix_modify>` options apply.  This fix
computes no output quantities.  The *start* and *stop* keywords of the
:doc:`run <run>` command ramp the target temperature.  This fix is not
invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the TETHER package.  It is only enabled if LAMMPS
was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`fix baoab <fix_baoab>`,
:doc:`fix baoab/tether <fix_baoab_tether>`,
:doc:`fix baoab/hrespa <fix_baoab_hrespa>`,
:doc:`run_style respa <run_style>`

Default
"""""""

*heff_samples* = 0, *gamma_soft* = 1/*damp*, *eps* = :math:`10^{-4}` times
the neighbor skin, *inner* = 0, *verbose* = *no*.
