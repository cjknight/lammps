.. index:: fix baoab/tether

fix baoab/tether command
=========================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID baoab/tether Tstart Tstop damp seed keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* baoab/tether = style name of this fix command
* Tstart, Tstop = desired temperature at start/end of run (temperature units)
* damp = damping parameter (time units)
* seed = random number seed (positive integer)
* two or more keyword/value pairs must be appended
* keyword = *theta* or *refresh* or *eps*

  .. parsed-literal::

       *theta* value = dimensionless stiffness threshold, must be > 0
       *refresh* value = steps between curvature-block refresh, must be > 0
       *eps* value = finite-difference probe displacement (distance units),
         must be > 0 (optional; default is a small fraction of the
         neighbor skin)

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20
   fix 2 rest nve

.. versionadded:: TBD

Description
"""""""""""

This fix is a variant of :doc:`fix baoab <fix_baoab>` that integrates each
atom in the fix group as though it were tethered to a frozen harmonic
center for whichever of its three local vibrational eigenmodes are
currently stiff relative to the timestep, while every other mode of that
same atom, and every atom outside the group, is integrated exactly as
:doc:`fix baoab <fix_baoab>` would integrate it. It is the stage-1
(fixed stiff-atom group, no adaptive timestep, no mollifier) implementation
of the frozen-Hessian mollified integrator described in the project's
design notes; later stages will add dynamic stiff/soft reclassification,
adaptive timestep selection, and event handling on top of this core.

Every *refresh* steps, this fix recomputes a local, per-atom, block-diagonal
curvature estimate for every atom in the group: three finite-difference
force probes (a positive and a negative displacement along each Cartesian
direction, displacement magnitude *eps*) give a 3x3 matrix that is
symmetrized and diagonalized, giving three mode directions and angular
frequencies :math:`\omega_1,\omega_2,\omega_3` for that atom. The atom's
current position becomes the frozen center for all three modes until the
next refresh. A mode is classified stiff if :math:`\omega\,dt > \theta`
(*theta*); soft modes fall back to ordinary free drift.

Each timestep performs the same B-A-O-A-B splitting as
:doc:`fix baoab <fix_baoab>`, generalizing only the drift ("A") and
thermostat ("O") sub-steps for group atoms: stiff modes are drift-integrated
analytically about the frozen center as an undamped harmonic oscillator
of frequency :math:`\omega`, and receive the exact Ornstein-Uhlenbeck
thermostat kick in that mode's own coordinate rather than in Cartesian
space; soft modes and all other atoms are unaffected. The half-step
velocity kicks from conservative forces, at the start and end of the
timestep, are identical for every atom regardless of group membership.

Restrictions
""""""""""""

This fix is part of the TETHER package. It is only enabled if LAMMPS was
built with that package; that package's :doc:`fix baoab/tether
<fix_baoab_tether>` also requires the EXTRA-FIX package to be enabled, since
it derives from and reuses :doc:`fix baoab <fix_baoab>`. See the
:doc:`Build package <Build_package>` page for more info.

This fix currently supports a single MPI rank only.

Curvature-block refresh does not yet have restart support: after reading a
restart file, this fix re-acquires curvature blocks for the whole group on
the first subsequent timestep rather than resuming mid-refresh-interval.

Group membership is static for this fix: an atom's stiff/soft mode
classification is only ever recomputed at a refresh for atoms already in
the group when the fix was defined; this fix does not yet support dynamic
group reclassification of which atoms belong to the stiff set.

This fix cannot be combined with :doc:`fix shake <fix_shake>` or
:doc:`fix rattle <fix_shake>` (inherited restriction from
:doc:`fix baoab <fix_baoab>`).

Do not combine this fix with :doc:`fix nve <fix_nve>` or any other
time-integration fix on the same group of atoms.

Related commands
"""""""""""""""""

:doc:`fix baoab <fix_baoab>`,
:doc:`fix langevin <fix_langevin>`,
:doc:`fix nvt <fix_nh>`

Default
"""""""

*eps* defaults to :math:`10^{-4}` times the neighbor skin distance.
