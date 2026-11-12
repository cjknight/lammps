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
* keyword = *theta* or *refresh* or *heff_samples* or *promote_every* or *settle_confirm* or *eps* or *mollify* or *newton_iters* or *newton_damp* or *solve_tol_rel* or *force_eval* or *adapt* or *c_acc* or *dt_min* or *dt_max* or *ke_rel* or *ke_rel_refresh* or *ke_rel_margin_blocks* or *gamma_quench* or *gamma_quench_calib_gain* or *gamma_quench_calib_max_step* or *gamma_quench_calib_confirm* or *gamma_quench_calib_deadband* or *gamma_quench_calib_ceiling* or *gamma_quench_calib_refresh* or *refresh_skin* or *refresh_skin_rel* or *drift_response* or *exile_substep* or *preshrink_exile* or *gamma_soft* or *eligibility* or *newton_res_length* or *recoil* or *molblock* or *molblock_inner* or *molblock_verbose* or *molblock_diag* or *event_log*

  .. parsed-literal::

       *theta* value = dimensionless stiffness threshold, must be > 0
       *refresh* value = steps between curvature-block refresh, must be > 0
       *heff_samples* value = number of consecutive scheduled curvature
         refreshes to average into one effective block, must be >= 0
         (optional; default 0, which keeps every refresh an independent,
         un-averaged instantaneous estimate)
       *promote_every* value = steps between extra dt-ladder promotion
         rechecks, must be >= 0 (optional; default 0, which disables this
         check and reproduces the behavior of promotion being re-evaluated
         only every *refresh* steps; only used if *adapt* is *yes*)
       *settle_confirm* value = number of consecutive fresh curvature
         refreshes that must agree the timestep may rise before an active
         *preshrink* floor (see *drift_response*) is released early, must
         be >= 0 (optional; default 0, which disables early release and
         always waits out the floor's full duration; only used if
         *adapt* is *yes*)
       *eps* value = finite-difference probe displacement (distance units),
         must be > 0 (optional; default is a small fraction of the
         neighbor skin)
       *mollify* value = *yes* or *no* (optional; default *no*)
       *newton_iters* value = minimum number of Newton solver iterations
         per timestep, must be >= 1 (optional; default 8; only used if
         *mollify* is *yes*)
       *newton_damp* value = initial Newton step damping factor, must be
         in the range (0,1] (optional; default 0.6; only used if
         *mollify* is *yes*)
       *solve_tol_rel* value = Newton convergence tolerance, relative to
         the thermal vibrational amplitude of an atom's stiffest mode,
         must be > 0 (optional; default 0.005; only used if *mollify*
         is *yes*)
       *force_eval* value = *local* or *full* (optional; default *local*;
         only used if *mollify* is *yes*; *full* additionally requires a
         single MPI process). Not currently recommended for production
         use; see the Description below.
       *adapt* value = *yes* or *no* (optional; default *no*)
       *c_acc* value = target accuracy for the fastest mode this fix is
         not treating analytically, dimensionless, must be > 0
         (optional; default 0.25; only used if *adapt* is *yes*)
       *dt_min* value = smallest timestep the adaptive controller may
         select (time units), must be > 0 (optional; default 0.02 times
         the timestep in effect when this fix was defined; only used if
         *adapt* is *yes*)
       *dt_max* value = largest timestep the adaptive controller may
         select (time units), must be > 0 (optional; default the
         timestep in effect when this fix was defined; only used if
         *adapt* is *yes*)
       *ke_rel* value = kinetic-energy over-excitation threshold for the
         event guard, in multiples of :math:`k_B T`, must be > 0, or
         *warmup* followed by a step count (optional; default 12.0; only
         used if *adapt* is *yes*; *warmup* additionally requires *adapt
         yes*). In testing on an H-in-Pd EAM system, every kinetic-energy
         guard trip at the default of 12.0 fell within a narrow 1.0-1.8x
         margin above the threshold, the signature of ordinary thermal
         fluctuations rather than a genuine over-excitation event; raising
         this value toward 25.0, by analogy to the fixed guard already used
         for over-excited harmonic energy (see the Description below),
         removed most of these spurious trips with no observed downside,
         and is a reasonable starting point to check for new production
         runs. See the Description below for what *warmup* does.
       *ke_rel_refresh* value = *freeze* or *refine* (optional; default
         *freeze*; only used if *ke_rel warmup* is given)
       *ke_rel_margin_blocks* value = target rarity, as 1-in-this-many,
         for the extrapolated *ke_rel* threshold, must be >= 1 (optional;
         default 1000; only used if *ke_rel warmup* is given)
       *gamma_quench* value = friction coefficient applied to an atom
         while it is cooling down after a demotion event (time units,
         inverse), must be > 0, or the literal word *auto*, optionally
         followed by *calib* and two step counts (optional; default 20.0;
         only used if *adapt* is *yes*; *auto* additionally requires
         *adapt yes*; *calib* additionally requires *gamma_quench auto*).
         See the Description below for what *auto* and *auto calib* do.
       *gamma_quench_calib_gain* value = exponent applied to the measured
         temperature ratio when committing a calibration update, must be
         > 0 (optional; default 0.5; only used if *gamma_quench auto
         calib* is given)
       *gamma_quench_calib_max_step* value = largest multiplicative change
         to *gamma_quench* allowed in a single calibration update, must be
         > 1.0 (optional; default 2.0; only used if *gamma_quench auto
         calib* is given)
       *gamma_quench_calib_confirm* value = number of consecutive
         measurement windows that must agree on the same direction before
         a calibration update commits, must be >= 1 (optional; default 3;
         only used if *gamma_quench auto calib* is given)
       *gamma_quench_calib_deadband* values = lo hi, the range of measured
         temperature ratio treated as "close enough" (no calibration
         evidence), lo must be in (0,1) and hi must be > 1.0 (optional;
         default 0.8 1.25; only used if *gamma_quench auto calib* is
         given)
       *gamma_quench_calib_ceiling* value = absolute upper bound the
         calibration loop may raise *gamma_quench* to, must be > 0
         (optional; default 10000.0; only used if *gamma_quench auto
         calib* is given)
       *gamma_quench_calib_refresh* value = *freeze* or *refine*
         (optional; default *refine*; only used if *gamma_quench auto
         calib* is given)
       *refresh_skin* value = excursion distance, beyond which an atom's
         drift away from its last curvature refresh triggers an early
         response (distance units), must be >= 0 (optional; default 0.0,
         which disables this check entirely; only used if *adapt* is
         *yes*; mutually exclusive with *refresh_skin_rel*)
       *refresh_skin_rel* value = dimensionless multiplier of each atom's
         own thermal ballistic displacement at *dt_max*, used in place of
         a single absolute excursion distance, must be >= 0 (optional;
         default 0.0, which disables this check entirely; only used if
         *adapt* is *yes*; mutually exclusive with *refresh_skin*)
       *drift_response* value = *retether* or *preshrink* (optional;
         default *retether*; only used if *refresh_skin* or
         *refresh_skin_rel* is > 0)
       *exile_substep* value = *yes* or *no* (optional; default *no*;
         only used if *adapt* is *yes*)
       *recoil* value = *yes* or *no* (optional; default *yes*)
       *preshrink_exile* value = *yes* or *no* (optional; default *no*;
         requires *exile_substep yes*; only used if *drift_response* is
         *preshrink*). Not currently recommended for production use; see
         the Description below.
       *gamma_soft* value = friction rate in 1/time for modes that are not
         clamped analytically (optional; default 1/*damp*)
       *eligibility* value = *per_mode* or *all_modes* or *bonded*
         (optional; default *per_mode*; only used if *adapt* is *yes*;
         *bonded* requires a molecular atom style with a bond style)
       *newton_res_length* value = *yes* or *no* (optional; default *no*)
       *molblock* value = *yes* or *no* (optional; default *no*; requires
         *mollify yes*, *adapt no*, and a molecular atom style)
       *molblock_inner* value = number of inner substeps per timestep,
         must be >= 0 (optional; default 0, which selects the number
         automatically; only used if *molblock* is *yes*)
       *molblock_verbose* value = *yes* or *no* (optional; default *no*;
         only used if *molblock* is *yes*)
       *molblock_diag* value = *yes* or *no* (optional; default *no*)
       *event_log* value = filename for a per-atom/per-event diagnostic
         log (optional; default none, which disables this log entirely)

Examples
""""""""

.. code-block:: LAMMPS

   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20
   fix 2 rest nve
   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20 &
         mollify yes newton_iters 8 newton_damp 0.6 solve_tol_rel 0.005
   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20 &
         mollify yes adapt yes c_acc 0.25 dt_min 0.0002 dt_max 0.01 &
         ke_rel 12.0 gamma_quench 20.0
   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20 &
         mollify yes adapt yes c_acc 0.25 dt_min 0.0002 dt_max 0.01 &
         ke_rel 12.0 gamma_quench 20.0 refresh_skin 0.1 &
         drift_response preshrink
   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20 &
         mollify yes adapt yes c_acc 0.25 dt_min 0.0002 dt_max 0.01 &
         ke_rel 25.0 gamma_quench auto refresh_skin 0.1 &
         drift_response preshrink
   fix 1 stiff baoab/tether 300.0 300.0 100.0 12345 theta 1.8 refresh 20 &
         mollify yes adapt yes c_acc 0.25 dt_min 0.0002 dt_max 0.01 &
         ke_rel warmup 2000 ke_rel_refresh refine gamma_quench auto &
         refresh_skin 0.1 drift_response preshrink
   fix 1 stiff baoab/tether 20.0 20.0 100.0 12345 theta 1.8 refresh 20 &
         mollify yes adapt yes c_acc 0.25 dt_min 0.0002 dt_max 0.01 &
         ke_rel 25.0 gamma_quench auto calib 2000 2000 &
         refresh_skin 0.1 drift_response preshrink event_log tether.log
   fix 1 all baoab/tether 300.0 300.0 1000.0 12345 theta 0.2 refresh 10 &
         mollify yes molblock yes heff_samples 12 gamma_soft 0.001

.. versionadded:: TBD

Description
"""""""""""

This fix is a variant of :doc:`fix baoab <fix_baoab>` that integrates each
atom in the fix group as though it were tethered to a harmonic center for
whichever of its three local vibrational eigenmodes are currently stiff
relative to the timestep, while every other mode of that same atom, and
every atom outside the group, is integrated exactly as :doc:`fix baoab
<fix_baoab>` would integrate it. It implements the frozen-Hessian
mollified integrator described in the project's design notes. By default
(*mollify no*) the harmonic center for each stiff mode is the atom's own
position at the last curvature refresh, held fixed until the next
refresh; the *mollify yes* keyword instead re-solves each stiff atom's
instantaneous equilibrium every timestep, as described further below.
Group membership is fixed (no dynamic stiff/soft reclassification) in
every mode. By default (*adapt no*) the timestep is also fixed, with no
event handling; the *adapt yes* keyword, described further below, adds
both adaptive timestep selection and event handling on top of this core.

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

----------

.. versionadded:: TBD

A single refresh's finite-difference curvature probe is an instantaneous
snapshot: for an atom whose local environment wobbles anharmonically
(light interstitial species caged by much heavier neighbors are the
typical case), consecutive refreshes can each diagonalize to a
meaningfully different estimate even though the atom's true, thermally
averaged local curvature is comparatively stable. Letting the working
estimate thrash with every refresh risks a stiff mode spuriously
reclassified as soft (or the reverse) between refreshes, purely from
probe noise rather than any real change in the atom's environment. The
*heff_samples* keyword addresses this by averaging the raw (symmetrized,
pre-diagonalization) curvature matrix over *heff_samples* consecutive
scheduled refreshes before using it: the matrix, not the diagonalized
frequencies, is accumulated, since averaging frequencies and eigenvectors
taken from different local eigenbases across an anharmonically wobbling
environment is not a meaningful operation, while averaging the underlying
matrix is (the same self-consistent-harmonic-approximation reasoning
behind thermally sampling an effective force constant). The working
estimate that actually governs stiff/soft classification is only
re-diagonalized at the first sample of each window (an ordinary
instantaneous estimate, exactly as every refresh was before this keyword
existed) and once more when the window closes to the full *heff_samples*-
sample average, which then stays frozen until the next window is armed;
samples in between accumulate silently without changing the working
estimate. This keeps the atom's stiff/soft classification and tethered
center stable for the whole averaging window instead of following every
intermediate noisy sample.

An emergency re-tether (triggered by *drift_response preshrink* or by the
kinetic-energy/drift excursion guards described below) always probes and
uses a fresh instantaneous estimate immediately, regardless of
*heff_samples*, since it needs the atom's current local curvature right
now rather than a stale mid-window average. When *heff_samples* is
greater than 0, that same emergency probe also arms a brand-new averaging
window, seeded with its own raw matrix as the window's first sample,
rather than leaving the working estimate permanently pinned on that one
instantaneous sample with no further averaging ever applied to it.
Leaving *heff_samples* at its default of 0 reproduces the original
behavior exactly: every refresh, scheduled or emergency, is an
independent, un-averaged instantaneous estimate.

This keyword is a direct port of a thermally-averaged effective curvature
technique already validated in this project's reference Python
implementation of this integration method, where it was essential for
stability on anharmonic rattler systems. Its C++ port, including the
re-arming behavior above, has not shown a measurable benefit in LAMMPS
testing on an H-in-Pd EAM system under the same *drift_response preshrink*
configuration documented elsewhere on this page: dt-occupancy, heavy-
subsystem temperature range, and guard/demotion event counts were
statistically indistinguishable from the *heff_samples* default of 0.

More importantly, short averaging windows were observed to be actively
unsafe on that same system: *heff_samples* values of 3 and 6 both caused
the run to lose atoms and abort (at steps 240 and 500 respectively) under
a configuration that runs stably at the default of 0 and that also runs
stably at *heff_samples* 12. This is consistent with the re-arming
behavior changing, not merely delaying, the exact sequence of curvature
estimates fed to an already-resonance-sensitive integrator on a chaotic
trajectory -- a difference anywhere early in the run propagates into a
different trajectory for the remainder of it, rather than being confined
to the moment the averaging window actually governs a decision. The
*heff_samples* 12 case surviving is consistent with a window long enough
that, on this system, it is essentially always re-armed by the next
emergency re-tether before it can close into a potentially stale
intermediate average. Treat short *heff_samples* windows (small relative
to how often emergency re-tethers recur for the atoms being tracked) as
an empirically identified risk, not just an unproven feature, on systems
using *drift_response preshrink*: verify against the standard diagnostic
suite (heavy-subsystem temperature trace, guard/demotion event counts,
dt-occupancy histogram, and simply whether the run completes) for any new
system or window size before relying on a particular *heff_samples*
setting.

----------

.. versionadded:: TBD

The *mollify* keyword selects the Newton mollifier. When set to *yes*,
the harmonic center for each stiff mode is not held fixed at the last
refresh position but is instead re-solved every timestep as that atom's
instantaneous constrained equilibrium, using the current positions of
its real pairwise neighbors as the effective cage. This is found with a
damped Newton iteration, warm-started from the atom's own previously
converged center, using only local force evaluations over that atom's
existing neighbor list rather than a global force recomputation per
iteration. *newton_iters* sets the minimum number of Newton iterations
performed each timestep. *newton_damp* sets the initial damping factor
applied to each Newton step; the damping is halved, down to a floor of
0.05, whenever an iteration's residual grows instead of shrinking.
*solve_tol_rel* sets the Newton convergence tolerance, relative to the
thermal vibrational amplitude of the atom's stiffest mode; values of
:math:`\omega\,dt` (*theta*) well below the 2.2 near-harmonic limit can
use the default, but values approaching that limit require a smaller
*solve_tol_rel* (of order 0.005 or tighter) to remain stable.

Enabling *mollify yes* also applies a force correction to each stiff
atom's neighbors, so that the neighbors feel the true gradient of the
mollified potential rather than the force evaluated at the stiff atom's
un-relaxed position. This fix also computes three per-atom self-evaluation
diagnostics each timestep (Newton non-convergence, a large jump ("flip")
in the solved center, and over-excited harmonic energy); they are always
available as output quantities (see below) for monitoring, but are only
acted on -- demoting the atom back to unmollified integration -- when
*adapt* is *yes*.

.. versionadded:: TBD

The *force_eval* keyword selects how the Newton solve above evaluates a
trial force. The default, *local*, is the behavior described above: a
local, no-communication force evaluation over the atom's existing
neighbor list. Setting this to *full* instead recomputes the force on
the trial atom from a genuine whole-system force evaluation (every pair,
bond, angle, dihedral, improper, and k-space contribution the simulation
has defined, not only the pairwise terms this fix's local evaluation
covers), for every Newton iteration. This is a correctness-validation
mode, not a production option: its cost scales with the total number of
atoms in the simulation rather than with one atom's neighbor count, and
it is restricted to a single MPI process (an error is raised at fix
creation if *force_eval full* is requested with more than one process).
It is intended for checking, on a small or serial system, how much of an
observed discrepancy between the local evaluation and an exact reference
comes from the local evaluation's neighbor-shell truncation, not for
everyday runs.

----------

.. versionadded:: TBD

The *adapt* keyword enables adaptive timestep selection together with an
event-handling mechanism that demotes individual atoms out of stiff-mode
treatment when they misbehave, and quenches them back to thermal
equilibrium afterward. It is only meaningful combined with *mollify yes*.

With *adapt yes*, the stiff/soft split for each atom's three eigenmodes
is no longer decided by comparing :math:`\omega\,dt` against the fixed
*theta* threshold. Instead, the first time this fix refreshes curvature
blocks under *adapt yes*, it collects every group atom's eigenfrequencies
on this MPI rank, sorts them, and looks for the largest multiplicative
gap in the upper half of that sorted list; the geometric mean of the two
frequencies straddling the gap becomes a global split frequency that is
held fixed for the remainder of the run. An atom's three modes are
treated as stiff only if all three are well above this split frequency;
otherwise every mode of that atom falls back to ordinary free drift, the
same as a soft mode in *adapt no* mode.

Also with *adapt yes*, at every curvature refresh this fix recomputes the
timestep from the fastest eigenmode it is not currently treating stiffly,
targeting a user-specified accuracy (*c_acc*), and clamped so the fastest
mode it does treat stiffly stays comfortably inside the harmonic
integrator's own stability limit. The resulting timestep is restricted to
a small set of powers of two of *dt_max* (never smaller than *dt_min*),
and changes only one step at a size at a time, shrinking immediately when
warranted but growing back only after a run of margin, so that it does
not chatter step to step. Changing the timestep this way is applied with
the same bookkeeping as the :doc:`fix dt/reset <fix_dt_reset>` command,
so elapsed simulation time, pair-style internal state, and other fixes
all stay consistent across the change.

Two further event checks run only when *adapt* is *yes*. First, the three
per-atom self-evaluation diagnostics described above (Newton
non-convergence, center "flip", over-excited harmonic energy) now trigger
an actual demotion when tripped: the offending atom's stiff modes are
dropped back to ordinary free drift for a cooldown period that grows,
capped, after repeated offenses from the same atom, and shrinks again
after a long enough period of good behavior. Second, every timestep, each
stiff atom's kinetic energy in its own stiff modes is compared against a
threshold (*ke_rel*, in multiples of :math:`k_B T`); an atom that exceeds
it is demoted the same way. In either case, an atom whose excess energy
looks like a genuine physical event rather than a numerical artifact is
also given a brief, strong-friction thermostat kick (*gamma_quench*)
while it cools down, so that legitimate kinetic energy does not linger in
its now-unconstrained modes.

----------

.. versionadded:: TBD

With *gamma_quench* set to *auto*, this fix determines the quench
friction coefficient itself instead of using a fixed literal value, and
keeps updating it for the entire run as the system evolves -- this is not
a one-time warm-up calibration. Every curvature refresh, *gamma_quench* is
set to the same global stiffest-mode frequency (the largest currently
stiff-classified eigenfrequency across every atom in the group) that
already governs the adaptive timestep itself, a critically-damped ansatz
tying the cooldown friction rate to whatever the system's own measured
dynamics currently justify rather than a hand-picked constant. The value
is free to rise or fall later in the run if the system's stiffest mode
changes (heating, a different atom becoming dominant, and so on). Because
this derivation is purely a function of each atom's own mass-normalized
curvature, it requires no pair-style- or species-specific tuning. When
*event_log* is also set, each time the derived value changes by more than
20% from the last value written, a *GAMMA_AUTO* entry records the new
value and the triggering frequency.

----------

.. versionadded:: TBD

Plain *auto* ties the quench friction to the stiffest mode's own
oscillation frequency, which is a critically-damped ansatz for that
*mode*, not a calibration against how hot the masked (stiff) subsystem
actually runs relative to its target temperature; in practice *auto* alone
can under-damp the masked subsystem by two to three orders of magnitude.
Appending *calib* followed by two step counts, *warmup* and *period*,
turns on a closed-loop controller that estimates the needed correction
from the run itself rather than from a hand-picked multiplier: for the
first *warmup* steps, *gamma_quench* follows plain *auto* exactly, while
this fix also accumulates, every timestep, the kinetic energy of every
masked mode on every group atom (excluding any step where the kinetic
guard above just fired, since that energy is being actively removed, not
representative of steady-state behavior). Every *period* steps after
*warmup* has elapsed, these accumulated sums are converted into an
effective temperature of the masked subsystem via equipartition and
compared, as a ratio, against the fix's own target temperature. A ratio
of 1.0 means the current *gamma_quench* is already correct; this is true
regardless of what that target temperature is or what system is being
simulated, which is what allows the same controller and the same default
settings to calibrate correctly across different temperatures and
different systems without being told in advance what value to converge
to.

Because any single measurement window can be dominated by ordinary
chaotic-trajectory noise rather than a genuine mismatch, a change is only
committed once *gamma_quench_calib_confirm* consecutive windows agree on
the same direction (too hot or too cold, outside the
*gamma_quench_calib_deadband* range); a window matching neither direction,
or disagreeing with the streak so far, resets the count. Once confirmed,
*gamma_quench* is multiplied by the measured ratio raised to
*gamma_quench_calib_gain* and clamped to at most
*gamma_quench_calib_max_step* in either direction, then clamped again to
lie between the plain *auto* value (a permanent floor -- *gamma_quench* is
never allowed to damp more weakly than critical damping for the current
stiffest mode) and *gamma_quench_calib_ceiling* (a generous safety rail,
not a target value). *gamma_quench_calib_gain*,
*gamma_quench_calib_max_step*, *gamma_quench_calib_confirm*, and
*gamma_quench_calib_deadband* govern only how cautiously this loop reacts
to its own measurement noise; none of them encode what the correct
damping value is for any particular system. With *gamma_quench_calib_refresh
refine* (the default), this process continues for the entire run, so the
calibration can continue tracking a system whose character changes over
time; with *freeze*, calibration stops updating after its first committed
change, similar in spirit to *ke_rel_refresh freeze*. When *event_log* is
set, each committed update is recorded as a *GAMMA_CALIB* entry with the
new value, the measured ratio and temperature, and the confirming streak.

*warmup* and *period* have no default: the right window lengths depend on
how many atoms are being tracked and how quickly the system's behavior can
change, and a hidden default here would simply reintroduce the kind of
unexamined constant this mechanism exists to avoid. Note that the number
of confirming windows needed to close a large initial mismatch can be
substantial -- each committed step is bounded by *gamma_quench_calib_max_step*
and each commit requires its own fresh *gamma_quench_calib_confirm*-window
streak -- so *period* should be chosen short enough, relative to the total
run length, for the controller to actually reach its target value within
the run being calibrated.

----------

.. versionadded:: TBD

Unlike *gamma_quench*, *ke_rel* is a rare-event guard threshold rather than
an ordinary dynamical property, and cannot be safely read off typical
thermal fluctuations: a guard threshold calibrated purely from an
ordinary-dynamics sample of "how hot does this atom normally get" will
always undershoot what the guard actually needs to tolerate, because that
sample essentially never contains the rare near-collision tail events a
guard exists to distinguish from. Setting *ke_rel* to *warmup* followed by
a step count instead fits the EXPONENTIAL DECAY RATE of that tail (a
Peaks-Over-Threshold fit with shape fixed at 0, matching the
:math:`\mathrm{Exponential}(\mathrm{mean}=0.5\,k_B T)` distribution a
single resolved harmonic mode's kinetic energy follows under
equipartition) and extrapolates it analytically to a target rarity,
rather than reading off whatever percentile of the warm-up sample happened
to be observed.

For the first *warmup* steps, this fix bins every group atom's
over-excitation ratio (the same quantity compared against *ke_rel* itself)
into a histogram and, once the window closes, fixes a threshold *u* at
its empirical 90th-percentile bin. From that point on, every sample
exceeding *u* updates two running totals: how many samples have exceeded
it, and by how much on average. These two numbers give a closed-form fit
for the tail's decay rate, which is then extrapolated to the ratio
expected to be exceeded only about once every *ke_rel_margin_blocks*
repetitions of the data collected so far (default 1000, matching the
roughly 1-in-1000 rarity target of the 99.9th-percentile approach this
replaces). The guard keeps using whatever *ke_rel* value was in effect
(hand-set or the 12.0 default) throughout this entire process, so the
over-excitation guard is never disabled or weakened while statistics are
being collected, and the fitted result is never allowed to end up below
that starting value.

*ke_rel_refresh* selects what happens once a fit is available, since the
right choice depends on the system: with *freeze* (the default), the fit
computed when *warmup* steps have elapsed is applied once and *ke_rel* is
then held fixed for the remainder of the run, with no further sampling
overhead. With *refine*, the running totals behind the tail fit are never
reset and continue accumulating for the rest of the run; every subsequent
curvature refresh re-derives the threshold from the now-larger sample and
raises *ke_rel* if the new fit exceeds the current value (never lowers
it). When *event_log* is also set, each such update is recorded as a
*KE_REL_CAL* entry with the new value and the fitted decay rate.

----------

.. versionadded:: TBD

The *refresh_skin* keyword adds an early-warning excursion guard for an
atom's soft (not currently stiff-treated) modes, which otherwise have no
analytic tether and are invisible to the kinetic-energy-based event checks
described above. Every timestep, while *refresh_skin* is greater than
zero, each stiff atom's minimum-imaged displacement from its position at
the last curvature refresh is compared against *refresh_skin*; exceeding
it triggers the response selected by *drift_response* before the kinetic
energy checks above run.

*refresh_skin_rel* is a dimensionless alternative to *refresh_skin*,
useful when the group spans species of very different mass: instead of
one fixed distance for every atom, each atom's own excursion threshold is
*refresh_skin_rel* times its own thermal ballistic displacement at
*dt_max*, :math:`\sqrt{k_B T / m_i}\ \mathrm{dt\_max}`. This threshold is
deliberately not based on the atom's own curvature: an atom with weak or
vanishing curvature in every mode (an unconfined rattler, exactly the
case this guard exists to catch) would otherwise produce an excursion
threshold that grows without bound and never trips. *refresh_skin* and
*refresh_skin_rel* are mutually exclusive.

With *drift_response* set to *preshrink* (the response validated for
production use), no per-atom action is taken immediately. Instead, at the
end of the timestep, this fix independently re-checks every candidate
atom's drift against a freshly probed curvature estimate and, if any atom
still needs it, lowers the timestep for the next step using the same
quantized-power-of-two ladder *adapt yes* already uses elsewhere, so the
atom's next step proceeds through the ordinary, unmodified integration
path at a size small enough to keep it valid. A short-lived floor prevents
the very next scheduled curvature refresh from relaxing the timestep back
up before the atom that justified the shrink has had a chance to clear the
encounter. With *drift_response* set to *retether* (the default, for
backward compatibility with prior verification runs), the triggering
atom instead has its curvature immediately re-probed at its current
position and is handed a local, sub-stepped Cartesian integration for the
remainder of the current timestep; this response has been superseded by
*preshrink* for production use, since it was found to inject spurious
kinetic energy when an atom's excursion invalidates its cached neighbor
list before a genuine close encounter is resolved.

----------

.. versionadded:: TBD

With *drift_response preshrink*, the short-lived floor described above
also blocks the ordinary, refresh-cadence timestep promotion described
under *adapt* from running until it expires, and that promotion only ever
raises the timestep one rung of the ladder at a time. When this floor
recurs much more often than the handful of refreshes needed to climb back
from the smallest rung to the largest, the timestep can become stuck near
*dt_min* indefinitely even once the encounter that triggered the floor has
long since cleared. The *promote_every* keyword addresses this without
changing the promotion decision itself: while greater than zero, this fix
re-evaluates the same one-rung-at-a-time promotion check every
*promote_every* steps (reusing the curvature estimate already cached from
the last refresh, so no extra curvature probing is done) instead of only
when a refresh is scheduled. Leaving it at its default of 0 reproduces the
original behavior exactly.

In testing on an H-in-Pd EAM system, this keyword did **not** meaningfully
raise the fraction of steps spent at *dt_max* at any setting tried (2, 5,
10, or 20): the timestep ceiling in that system was set by genuinely
volatile, near-continuously elevated curvature, not by the promotion
cadence, so closing the cadence gap had no effect on the outcome it was
meant to fix. A recheck interval of 1 (every step) additionally caused a
real stability regression in that same test -- guard-triggered kinetic
energy rescales rose by roughly 26x over the unmodified baseline, with a
correspondingly higher peak temperature in the light subsystem -- because
reusing one curvature snapshot to justify climbing multiple ladder rungs
across consecutive steps, with no fresh probe confirming each rung is
still safe for the atom's evolving post-encounter state, is not equivalent
to the implicit re-validation that a real refresh provides. A recheck
interval of 1 is therefore **not** recommended. If this keyword is used at
all, prefer a larger interval (5 or more) and verify against the standard
diagnostic suite (heavy-subsystem temperature trace, guard/demotion event
counts, dt-occupancy histogram) that it has not introduced new heating for
the system at hand; do not assume it will improve cruise fraction without
checking. When running on more than one MPI process this also adds one
small collective communication call per recheck.

----------

.. versionadded:: TBD

The *preshrink* floor described above always waits out a fixed duration
before allowing the ordinary one-rung-at-a-time promotion to resume, even
if the encounter that triggered it clears well before the floor expires.
The *settle_confirm* keyword lets that floor end early, without changing
the promotion decision itself or how the floor is triggered: while the
floor is active, this fix keeps counting how many consecutive, regularly
scheduled curvature refreshes in a row independently find that the
timestep is already eligible for ordinary promotion. Once that count
reaches *settle_confirm*, the floor is lifted immediately, and the
existing one-rung climb resumes starting with the refresh after next.
Every one of these checks uses a freshly recomputed curvature estimate,
never a cached or reused one from a *promote_every* recheck, and any
refresh that fails the check, or any fresh retrigger of the floor itself,
resets the count to zero. Leaving this keyword at its default of 0
disables early release entirely and reproduces the original fixed-duration
behavior exactly.

In testing on the same H-in-Pd EAM system used above, *settle_confirm 2*
gave a real, measurable benefit with no added instability: time spent at
the smallest timestep rung fell from 81% to 74% of the run, time spent at
the next rung up more than tripled, guard-triggered kinetic energy
rescales fell (by about 46%) rather than rose, and the heavy-subsystem
temperature trace stayed within its established stable range. This did
**not** increase the fraction of steps spent at the largest timestep,
*dt_max*, which stayed unchanged -- consistent with the finding under
*promote_every* above that this system's timestep ceiling is set by
genuinely volatile curvature rather than by how quickly the floor is
released. Settings of 5 or more showed no measurable effect at all on this
system: its curvature apparently never holds still for that many
consecutive refreshes in a row, which is itself informative about how
volatile this particular system is, not a flaw in the mechanism. As with
*promote_every*, verify against the standard diagnostic suite before
relying on any particular setting for a new system, and do not assume a
given value will help without checking.

----------

.. versionadded:: TBD

An atom that keeps re-triggering demotion is placed in a temporary
"exile": for a cooldown window after each such event, every one of its
local modes is treated as non-stiff regardless of what the cached
curvature says (see the *ke_rel* and *drift_response* discussion above).
Without *exile_substep*, an exiled atom still takes the ordinary shared
drift at this fix's single, global timestep, and its cached curvature is
still folded into the vote that sets that global timestep and into the
*preshrink* trigger described above -- so one atom already flagged as
chronically unstable, and already relegated to the weakest integration
path this fix offers, can keep the timestep small for every *other* atom
in the group as well, which runs against the purpose of giving a
troublesome atom its own separate treatment in the first place. With
*exile_substep yes*, an exiled atom instead integrates for the entire
exile window using the same local, sub-stepped, resonance-bounded
Cartesian integration this fix already uses for one step at a time during
certain guard responses, sized from the atom's own cached curvature and
independent of the shared timestep, and it is excluded from both the
global timestep vote and the *preshrink* trigger for as long as the exile
lasts. This does not change when exile starts, how long it lasts, or how
its cooldown decays -- only what the atom does while exiled, and whether
it is counted. Leaving this keyword at its default of *no* reproduces the
original behavior exactly.

In testing on the same H-in-Pd EAM system used above, *exile_substep yes*
left the system safe -- the heavy-subsystem temperature trace and total
energy stayed within their established stable ranges, with no sign of the
accumulating drift that would result from double-counting an exiled
atom's closing half-kick -- and produced a real, if modest, redistribution
of the dt-occupancy histogram toward higher rungs (time spent at the
second-smallest rung roughly quadrupled, and a small amount of time
reached a rung that was never populated at all without this keyword).
It did **not** raise the fraction of steps spent at *dt_max*, which was
already close to zero and stayed there, and the running count of demotion
events rose several-fold rather than fell. That rise is consistent with
what this keyword is meant to do, not a regression: once an already-exiled
atom's curvature can no longer hold the shared timestep down, that
timestep is freer to rise for the rest of the group, which in turn exposes
genuine resonance limits in other atoms that the previously depressed
timestep had been masking. As with *promote_every* and *settle_confirm*
above, this keyword does not by itself solve the cruise-fraction ceiling
this system has shown under every tuning tried so far; verify against the
standard diagnostic suite (heavy-subsystem temperature trace, guard/
demotion event counts, dt-occupancy histogram, and energy drift over a
single long exile episode specifically) before relying on it for a new
system.

----------

.. versionadded:: TBD

With *drift_response preshrink*, any single atom whose drift crosses its
own local threshold shrinks this fix's shared, global timestep for every
atom and locks it there for a cooldown window, even though the rest of
the group may have nothing wrong with it. *preshrink_exile* changes what
happens when that trigger fires: instead of shrinking the shared
timestep, the triggering atom alone is placed into the same exile
treatment described above for *exile_substep* (same fixed cooldown
duration as the kinetic-energy guard uses), leaving the shared timestep
free to serve the rest of the group. Requires *exile_substep yes*, since
it reuses that keyword's exile machinery rather than introducing a
second one; setting *preshrink_exile yes* without it is an error.
Leaving this keyword at its default of *no* reproduces the original
*preshrink* behavior exactly.

.. warning::

   *preshrink_exile* is **not currently recommended for production use**.
   In testing on the same H-in-Pd EAM system used above, it reproduced a
   known failure mode this fix's *preshrink* response was specifically
   designed to avoid: an earlier, same-step local-rescue response to this
   same drift trigger (now the *retether* setting of *drift_response*)
   was found during this fix's development to fire on nearly every step
   from ordinary thermal jitter, with each rescue's own approximation
   error compounding into the very close encounter it was meant to
   prevent. *preshrink_exile* routes the same predictive, pre-encounter
   drift trigger into a different mechanism -- a sustained exile window,
   already validated for the kinetic-energy guard's *reactive*,
   post-encounter trigger -- but an exiled atom receives no thermostat
   coupling while it integrates on its own, and the drift trigger fires
   before the close encounter it is warning about has resolved. The
   result was the same failure signature by a different route: within
   about 120 steps the heavy-subsystem temperature reached in excess of
   170,000 K and the run aborted on lost atoms, with *preshrink_exile*
   triggering roughly twenty times more often over that short span than
   the ordinary *preshrink* global-shrink response triggers across an
   entire stable 10000-step run of the same system. *exile_substep*
   itself is unaffected by this finding and remains safe on its own; the
   problem is specific to feeding it this particular, predictive trigger
   with no damping applied during the exile window.

----------

.. versionadded:: TBD

The *event_log* keyword writes a permanent, per-atom/per-event diagnostic
log to the given filename, one line per curvature refresh, guard trip,
demotion, or timestep preshrink. Each rank opens its own log file; when
running on more than one MPI process, the rank number is appended to the
filename so that each rank's local events land in a separate file. The
log is flushed to disk after every line, so it survives up to the moment
of a crash. Leaving *event_log* unset (the default) disables the log
entirely, with no effect on this fix's behavior or performance.

----------

.. versionchanged:: TBD

This fix detects the spectral gap and picks the timestep from the
combined atoms on every MPI rank: the per-rank local frequency lists are
gathered once, at the first refresh, and every rank runs the same
deterministic gap search against the merged result, and the accuracy-
limited timestep ceiling described above is reduced across ranks the same
way every step. See Restrictions below for the remaining, narrower
cross-*group* limitation this does not address.

----------

The *damp* argument and the *gamma_soft* keyword set the strength of the
Langevin thermostat in the run's own time units (*damp* is a relaxation
time, *gamma_soft* a rate). *gamma_soft* applies a baseline thermostat to
the modes that are not clamped analytically; if it is not given it defaults
to 1/*damp*, so a single physical relaxation time governs the whole fix.
Giving *gamma_soft 0* explicitly turns that baseline thermostat off. Because
these values are in time units, a number copied from one unit style to
another (for example 0.1 in metal units, i.e. picoseconds, used in real
units, i.e. femtoseconds) is wrong by orders of magnitude and would make the
thermostat re-randomize every velocity on every step, producing a stable
looking but physically meaningless trajectory. The fix therefore stops with
an error if either rate multiplied by the largest timestep (*dt_max* with
*adapt yes*, otherwise the current timestep) exceeds 1. A typical choice is
a *damp* of about 100 times the largest timestep.

The *eligibility* keyword selects which atoms may have a mode clamped
analytically when *adapt* is *yes*. With the default, *per_mode*, every
mode of every atom that exceeds the spectral split is clamped on its own,
whatever the atom's other modes are. With *all_modes*, an atom is eligible
only if all three of its modes exceed 0.6 times the spectral split. This
leaves an atom that is stiff in only one or two directions (for example a
hydrogen atom crossing between interstitial sites) entirely unresolved,
and because all atoms share one timestep, such an atom limits the timestep
for the whole system. In large systems some atom is almost always in that
state, so *all_modes* can lose most of the timestep gain. With *per_mode*
two further rules keep the timestep independent of system size. A larger
timestep is still only taken when every mode allows it, but the timestep
is reduced only when more than 1% of the modes that may be clamped no
longer allow it (the 99th percentile of their frequencies), not as soon as
the single fastest one does. The fastest mode among many atoms keeps
getting faster as the system grows, so without this rule a single rare
fast mode would reduce the timestep for all atoms. A mode that is too fast
for the current timestep (above *c_acc* divided by 0.85 times the
timestep) is clamped instead. Modes of atoms that cannot be clamped (for
example after a demotion) still reduce the timestep on their own. With
*eligibility bonded* only atoms that have at least one bond skip the
all-modes test. It requires a molecular atom style and a bond style, and
with *mollify yes* the usual *newton_bond off* requirement applies.

The *newton_res_length* keyword selects the units of the residual that the
damped-Newton center solve compares against its convergence tolerance and
against the *newton_tol_rel* guard. With *no* (default) the residual is the
projected force, which is compared against a thermal amplitude (a length);
this mismatch is harmless when the effective spring constant is of order one
(hydrogen in palladium, metal units) but is wrong by that spring constant in
general. With *yes* the residual is the Newton step length, and the
comparison is dimensionally consistent. Use *yes* for systems with stiff
bonded modes, such as flexible water in real units.

----------

.. versionadded:: TBD

The *molblock* keyword replaces the per-atom treatment described above with
a multiple time step integrator for small, stiff, bonded molecules, such as
flexible water.  A molecule here is an atom together with its bond and angle
partners, at most 5 atoms in total.  At every curvature refresh the fix
assembles the full second-derivative matrix of each molecule from the same
finite-difference probes it uses for the per-atom blocks.  If any internal
vibration of a molecule is faster than the *theta* threshold allows, every
bond length and bond angle of that molecule is included in a "fast" harmonic
model, :math:`V_{\rm fast} = \frac{1}{2}(s - s_0)^T F (s - s_0)`, where
*s* holds the bond lengths and angles.  The force constants *F* are
extracted from the measured second derivatives and forces (the Wilson GF
method), so they follow whatever force field or machine-learned potential is
in use, including the influence of neighboring molecules; no fast/slow split
of the force field itself has to be specified.  The reference values
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
force evaluations, and ends with another slow half kick.  The Langevin
thermostat, with rate *gamma_soft*, is applied once per timestep in the
middle of the substeps.  Atoms that are not part of a stiff molecule simply
drift during the substeps.  When the force constants are refreshed, the
timestep in progress completes with the old constants and the new constants
take effect at the start of the next timestep, so that every timestep is a
consistent splitting of the same total force.  Only one force evaluation is
needed per timestep, as in plain velocity Verlet, plus the curvature probes
at each refresh.

How the force constants are updated after the first step depends on
*heff_samples*.  With *heff_samples 0* (the default), they are measured
again from finite-difference probes every *refresh* steps, which gives
the instantaneous curvature at that moment but costs roughly as much as
15 to 20 ordinary timesteps per refresh.  With *heff_samples* N > 0, the
probes are used only at the first step.  Afterwards the force constants
are estimated from the trajectory itself, at no extra cost: from one
timestep to the next, the change in the force on the bond lengths and
angles is very nearly the force constants times the change in those
coordinates, because the vibration dominates both changes while the
slow drift of the molecule's surroundings largely cancels.  These
step-to-step changes are accumulated over each block of *refresh*
steps, blended into a running average in which each new block has
weight 1/N (with the probed values as the starting point), and the
resulting force constants are used from the next timestep on.  This is
a thermally averaged effective curvature, in the same spirit as
*heff_samples* for the per-atom treatment.  An estimate that is not
physically sensible (not positive definite) is discarded and the
previous force constants are kept.  For flexible SPC/Fw water, *refresh
10* with *heff_samples 12* gave force constants matching the probed
values to within their thermal fluctuation (about 1.5 percent), the
same accuracy in all checks listed below as probing every 10 steps, and
about twice the speed; a shorter memory (*heff_samples 4*) was stable
but conserved energy slightly less well.

The *molblock_inner* keyword sets the number of substeps.  With the
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

The *molblock_verbose* keyword prints, at every refresh, the number of
molecules and fast coordinates (or, with *heff_samples* > 0, the number
of molecules whose force constants were updated or kept), the fastest
frequency times the timestep, and the number of substeps, together with
the force constants of one molecule at the first refresh and, with
*heff_samples* > 0, the average diagonal force constants.

The *molblock_diag* keyword is a diagnostic that changes no dynamics.  At
every refresh it assembles each molecule's second-derivative matrix as
above and prints a summary of its internal and rigid-body frequencies.

*molblock* requires *mollify yes* (which provides the finite-difference
probes and the *newton_bond off* bond and angle storage it relies on), a
molecular atom style, and *adapt no*; the timestep stays fixed.  Molecules
are found from the bond and angle lists, so the bond and angle styles must
be able to report their atoms to this fix; with a machine-learned potential
that supplies all forces, use *bond_style harmonic* and *angle_style
harmonic* with zero force constants rather than the *zero* styles.  Every
atom of a molecule must be linked to every other one by a bond or an angle
(water, diatomics, linear triatomics such as carbon dioxide, ammonia-like
molecules), and the bonds and angles of a molecule must be independent of
each other, which excludes molecules with rings; the fix stops with an error
otherwise.

*molblock* runs in parallel without any communication during the
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
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>` (see the curvature-refresh restriction below).

.. versionchanged:: TBD

.. versionadded:: TBD

The *recoil* keyword controls a reduced-mass correction to the frequency
of every mode that is treated analytically.  Such a mode is extracted
from one atom's own curvature block with all neighbors held fixed, which
gives a frequency of sqrt(k/m) for that atom.  If the neighbor that
supplies the restoring force is free to move, the mode actually runs at
sqrt(k/m + k/M), where M is the neighbor's mass.  For each analytic mode
the fix therefore adds the neighbors' coupling to that mode divided by
the neighbor mass, using the response blocks it already measures.  The
correction is negligible for a light atom in a cage of heavy atoms (a
hydrogen atom in palladium) and about 3 percent in frequency for the
O-H stretch of water, where it moves the stretch peak of the
velocity spectrum from about 3500 to about 3610 inverse centimeters
(reference: about 3690).  Setting *recoil no* restores the uncorrected
frequencies.

.. versionchanged:: TBD

The adaptive timestep ladder now raises the timestep one rung only when the
estimated stable step exceeds 1.87 times the current rung.  Previously the
threshold was 1.25 times the rung, which lies inside the band where the
step is lowered again at the next rung, so the timestep could alternate
between two rungs at every refresh.

This fix computes a global vector of length 18, which can be accessed by
various :doc:`output commands <Howto_output>`. The values are, in order:
the running count of Newton non-convergence guard trips, the running
count of center "flip" guard trips, the running count of over-excited
harmonic energy guard trips, the running count of local partial-force
evaluations performed by the Newton solver, the running count of full
local force recomputations performed by this fix, the running count of
demotion events from any cause, the running count of demotion events
from the kinetic over-excitation guard specifically (a subset of the
previous value), and the running count of stored back-reaction neighbors
that could not be resolved on the calling rank because they had drifted
outside the current ghost cutoff since the last curvature refresh (see
Restrictions below; that contribution is silently dropped rather than
causing an error). The 4th and 5th entries are a proxy for the fix's
force-evaluation cost. All counts are summed over all processors, and the
frequency diagnostics report the maximum over all processors. All eight entries are 0 unless
*mollify* is *yes*; the 6th, 7th, and 8th are additionally 0 unless
*adapt* is also *yes* (the 8th can in principle occur with *mollify yes*
alone, but in practice only matters once *adapt* keeps a fixed curvature
block in play across enough steps for it to be exercised). The first
three and the 6th/7th entries are diagnostic only except when *adapt* is
*yes*, in which case the first three drive actual demotion events (see
above). The vector values calculated by this fix are "extensive".

The 9th through 12th entries are diagnostic counters for the stale-mask
checks and the teacher force calls.  The 13th through 16th entries report,
as of the most recent timestep selection, the highest frequency among
modes that are not treated analytically (this sets the adaptive timestep),
the LAMMPS atom type that owns it, the 99th percentile of the same
frequencies on this processor, and the highest frequency among modes that
are treated analytically.  Frequencies are in radians per unit of time
of the active unit style.  They are 0 unless *adapt* is *yes*.

.. versionadded:: TBD

The 17th and 18th entries are the wall-clock time in seconds spent in
the teacher force corrections since the fix was defined, and the part of
that time spent in the teacher pair style itself (including any copies
between host and GPU).  Both are 0 unless *teacher_every* is set.  At the
end of each run with a teacher, the fix also prints how the run's wall
time divides between the teacher calls and everything else.

This fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the TETHER package. It is only enabled if LAMMPS was
built with that package; that package's :doc:`fix baoab/tether
<fix_baoab_tether>` also requires the EXTRA-FIX package to be enabled, since
it derives from and reuses :doc:`fix baoab <fix_baoab>`. See the
:doc:`Build package <Build_package>` page for more info.

.. versionchanged:: TBD

This fix supports multiple MPI ranks: this fix's own private per-atom
clamped-center state is forwarded to ghost atoms, back-reaction
contributions written into ghost atoms' forces are returned to their
owning rank, and (with *adapt yes*) the spectral-gap split frequency and
the adaptive timestep ceiling are both computed from the combined atoms
on every rank rather than only the calling rank's own local atoms.

A curvature-block back-reaction neighbor recorded at the last refresh can
still drift outside the current ghost cutoff before the next refresh,
particularly under a small ghost cutoff/skin relative to how far an atom
moves in *refresh* steps; when this happens, that neighbor's contribution
is silently dropped for the remainder of the current refresh interval
rather than causing an error. The 8th entry of this fix's global vector
(above) counts how often this occurs, so it can be monitored; it is 0 for
a sufficiently generous cutoff/skin relative to *refresh* and the
system's own mobility.

With *adapt yes*, the accuracy-limited timestep ceiling (the
:math:`c_{\rm acc}/\omega_{\rm unres}` term) is still computed only from
this fix's own group's per-atom curvature (now combined correctly across
every rank that owns part of the group, but still not including atoms
outside the group), since the fix has no visibility into the curvature of
atoms outside its group (e.g. a heavier bath integrated by a separate,
ordinary time-integration fix on the rest of the system). Whenever this
group's own unresolved band sits at a higher frequency than that outside
bath, this fix will select a smaller timestep than an integrator with
full-system curvature visibility would choose for the same accuracy
target. This is expected and is a direct consequence of this fix's
halo-local, per-group design: it makes the selected timestep more
conservative (safe, but potentially smaller than optimal), never less
conservative.

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

.. versionadded:: TBD

This fix runs on the host (CPU).  With the :doc:`KOKKOS package
<Speed_kokkos>`, its main pair style must not be a KOKKOS style (except
*symmetrix/mace/kk*), because the fix's per-atom force evaluations read
data that KOKKOS pair styles keep on the GPU.  When using the "-sf kk"
command-line switch, define that pair style after a :doc:`suffix off
<suffix>` command.  The *teacher_style* may be a KOKKOS style: it takes
the active suffix like any :doc:`pair_style <pair_style>` command, so
with "-sf kk" a teacher style of *symmetrix/mace* becomes
*symmetrix/mace/kk* and runs on the GPU, with the positions sent to the
GPU and the forces copied back at every teacher call.  For example:

.. code-block:: LAMMPS

   suffix off
   pair_style eam/fs
   pair_coeff * * PdH.eam.fs H Pd
   suffix kk
   fix 1 all baoab/tether ... teacher_style 2 symmetrix/mace no_mpi_message_passing &
       teacher_coeff 5 * * model.json H Pd teacher_every 20

Related commands
"""""""""""""""""

:doc:`fix baoab <fix_baoab>`,
:doc:`fix langevin <fix_langevin>`,
:doc:`fix nvt <fix_nh>`

Default
"""""""

*heff_samples* defaults to 0 (every refresh is an instantaneous,
un-averaged estimate). *eps* defaults to :math:`10^{-4}` times the
neighbor skin distance. *mollify* defaults to *no*. *newton_iters* defaults to 8. *newton_damp*
defaults to 0.6. *solve_tol_rel* defaults to 0.005. *adapt* defaults to
*no*. *c_acc* defaults to 0.25. *dt_min* defaults to 0.02 times the
timestep in effect when this fix was defined. *dt_max* defaults to the
timestep in effect when this fix was defined. *ke_rel* defaults to 12.0,
with *warmup* disabled. *ke_rel_refresh* defaults to *freeze*.
*ke_rel_margin_blocks* defaults to 1000.
*gamma_quench* defaults to 20.0, with *auto calib* disabled.
*gamma_quench_calib_gain* defaults to 0.5. *gamma_quench_calib_max_step*
defaults to 2.0. *gamma_quench_calib_confirm* defaults to 3.
*gamma_quench_calib_deadband* defaults to 0.8 1.25.
*gamma_quench_calib_ceiling* defaults to 10000.0.
*gamma_quench_calib_refresh* defaults to *refine*.
*refresh_skin* defaults to 0.0.
*refresh_skin_rel* defaults to 0.0. *promote_every* defaults to 0.
*settle_confirm* defaults to 0. *drift_response* defaults to *retether*.
*exile_substep* defaults to *no*. *preshrink_exile* defaults to *no*. *recoil* defaults to *yes*.
*gamma_soft* defaults to 1/*damp*. *eligibility* defaults to *per_mode*. *newton_res_length* defaults to *no*.
*molblock* defaults to *no*. *molblock_inner* defaults to 0. *molblock_verbose* defaults to *no*.
*molblock_diag* defaults to *no*.
*event_log* defaults to unset (disabled).
