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
* keyword = *theta* or *refresh* or *promote_every* or *settle_confirm* or *eps* or *mollify* or *newton_iters* or *newton_damp* or *solve_tol_rel* or *adapt* or *c_acc* or *dt_min* or *dt_max* or *ke_rel* or *gamma_quench* or *refresh_skin* or *refresh_skin_rel* or *drift_response* or *event_log*

  .. parsed-literal::

       *theta* value = dimensionless stiffness threshold, must be > 0
       *refresh* value = steps between curvature-block refresh, must be > 0
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
         event guard, in multiples of :math:`k_B T`, must be > 0
         (optional; default 12.0; only used if *adapt* is *yes*). In
         testing on an H-in-Pd EAM system, every kinetic-energy guard trip
         at the default of 12.0 fell within a narrow 1.0-1.8x margin above
         the threshold, the signature of ordinary thermal fluctuations
         rather than a genuine over-excitation event; raising this value
         toward 25.0, by analogy to the fixed guard already used for
         over-excited harmonic energy (see the Description below), removed
         most of these spurious trips with no observed downside, and is a
         reasonable starting point to check for new production runs.
       *gamma_quench* value = friction coefficient applied to an atom
         while it is cooling down after a demotion event (time units,
         inverse), must be > 0 (optional; default 20.0; only used if
         *adapt* is *yes*)
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

Restart, fix_modify, output, run start/stop, minimize info
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

No information about this fix is written to :doc:`binary restart files
<restart>` (see the curvature-refresh restriction below).

.. versionchanged:: TBD

This fix computes a global vector of length 8, which can be accessed by
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
causing an error). The 4th and 5th entries are a per-rank, halo-local
proxy for the fix's force-evaluation cost. All eight entries are 0 unless
*mollify* is *yes*; the 6th, 7th, and 8th are additionally 0 unless
*adapt* is also *yes* (the 8th can in principle occur with *mollify yes*
alone, but in practice only matters once *adapt* keeps a fixed curvature
block in play across enough steps for it to be exercised). The first
three and the 6th/7th entries are diagnostic only except when *adapt* is
*yes*, in which case the first three drive actual demotion events (see
above). The vector values calculated by this fix are "extensive".

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

Related commands
"""""""""""""""""

:doc:`fix baoab <fix_baoab>`,
:doc:`fix langevin <fix_langevin>`,
:doc:`fix nvt <fix_nh>`

Default
"""""""

*eps* defaults to :math:`10^{-4}` times the neighbor skin distance.
*mollify* defaults to *no*. *newton_iters* defaults to 8. *newton_damp*
defaults to 0.6. *solve_tol_rel* defaults to 0.005. *adapt* defaults to
*no*. *c_acc* defaults to 0.25. *dt_min* defaults to 0.02 times the
timestep in effect when this fix was defined. *dt_max* defaults to the
timestep in effect when this fix was defined. *ke_rel* defaults to 12.0.
*gamma_quench* defaults to 20.0. *refresh_skin* defaults to 0.0.
*refresh_skin_rel* defaults to 0.0. *promote_every* defaults to 0.
*settle_confirm* defaults to 0. *drift_response* defaults to *retether*.
*event_log* defaults to unset (disabled).
