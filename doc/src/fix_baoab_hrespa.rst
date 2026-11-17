.. index:: fix baoab/hrespa

fix baoab/hrespa command
========================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID baoab/hrespa Tstart Tstop damp seed keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command; group-ID must be *all*
* baoab/hrespa = style name of this fix command
* Tstart, Tstop = desired temperature at start/end of run (temperature units)
* damp = damping parameter (time units)
* seed = random number seed (positive integer)
* zero or more keyword/value pairs may be appended
* keyword = *inner* or *cut* or *eps* or *read* or *write*

  .. parsed-literal::

       *inner* value = number of inner substeps per timestep, must be >= 0
         (0 selects the number automatically)
       *cut* value = distance (distance units) beyond which atom pairs are
         not coupled in the harmonic model, must be > 0
       *eps* value = finite-difference displacement (distance units) for
         building the harmonic model, must be > 0
       *read* value = file to read the harmonic model from
       *write* value = file to write the harmonic model to

Examples
""""""""

.. code-block:: LAMMPS

   minimize 0.0 1.0e-6 2000 20000
   fix 1 all baoab/hrespa 300.0 300.0 1.0 12345 write hfo2.model
   run 100000

   fix 1 all baoab/hrespa 300.0 300.0 1.0e9 12345 read hfo2.model inner 0

Description
"""""""""""

.. versionadded:: TBD

This fix performs Langevin (or, with a very large *damp*,
microcanonical) dynamics with a two-level multiple-time-step integrator
for systems without a molecular split, such as crystals, glasses, or
oxides described by machine-learned potentials.  Its fast level is a
harmonic model of the complete force field,

.. math::

   \mathbf{F}_h(\mathbf{x}) = \mathbf{f}_\text{ref} - \mathbf{K}\,(\mathbf{x} - \mathbf{x}_\text{ref}) ,

where displacements are taken by the minimum-image convention.  Each
timestep starts and ends with a half kick by the difference between the
full force and :math:`\mathbf{F}_h`; in between, *inner* velocity Verlet
substeps integrate the atoms under :math:`\mathbf{F}_h` alone.  The
substeps need no force-field evaluations (one communication of positions
each), so the cost per timestep is about one force evaluation.  The
thermostat (with friction 1/*damp*) acts once per timestep, in the middle
of the substeps; a very large *damp* gives microcanonical dynamics.  With
*inner* 0 (the default) the number of substeps is chosen so that the
fastest harmonic frequency times the substep is at most 0.1.

Without *read*, the harmonic model is built at the start of the first
run from the current configuration: :math:`\mathbf{x}_\text{ref}` and
:math:`\mathbf{f}_\text{ref}` are the current positions and forces, and
:math:`\mathbf{K}` is obtained by central finite differences of the forces
(6 force evaluations per atom, displacement *eps*, default 0.001
Angstrom).  It is usually best to minimize the energy first.  Blocks of
:math:`\mathbf{K}` coupling atoms farther apart than *cut* at
:math:`\mathbf{x}_\text{ref}` are dropped, :math:`\mathbf{K}` is made
symmetric, and the diagonal blocks are set so that a rigid translation
costs no force.  *write* saves the model in a text file that *read* can
use for later runs; the model is kept for all subsequent runs in the
same input.  The file lists the number of atoms and of 3x3 blocks, one
line per atom with its ID, reference position and force, and one line
per block with the two atom IDs and the nine block entries (row by row).

The largest stable timestep is limited by a resonance of the slow forces
with the fastest vibrations (about one third of the shortest vibrational
period), so this fix is accurate when the anharmonic part of the forces
is small, e.g. a solid at moderate temperature.

Pair styles that set atom charges at every force evaluation (such as
latent-Ewald machine-learned potentials) are supported: the charges of
the actual configuration are restored after each finite-difference
evaluation.

Restart, fix_modify, output, run start/stop, minimize info
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>`; the harmonic model can be saved with *write* and read back
with *read*.  None of the :doc:`fix_modify <fix_modify>` options apply.
This fix computes no output quantities.  The *start* and *stop* keywords
of the :doc:`run <run>` command ramp the target temperature.  This fix
is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the TETHER package.  It is only enabled if LAMMPS
was built with that package.  See the :doc:`Build package
<Build_package>` page for more info.

The fix must be applied to group *all*.  It requires consecutive atom
IDs, :doc:`atom_modify map <atom_modify>`, and that all atoms coupled by
the model are within the ghost-atom cutoff (which holds for the default
*cut*).  The model is stored on every processor, which limits system
sizes to some ten thousand atoms.  The model is fixed for the whole run.

Related commands
""""""""""""""""

:doc:`fix baoab <fix_baoab>`,
:doc:`fix baoab/tether <fix_baoab_tether>`,
:doc:`run_style respa <run_style>`

Default
"""""""

*inner* = 0, *cut* = the pair style cutoff, *eps* = 0.001 Angstrom,
no *read* or *write* file.
