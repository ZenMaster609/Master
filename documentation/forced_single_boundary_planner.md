# Forced Single-Boundary Planner

This document describes the forced single-boundary planner as a standalone method.

The idea is simple:

- we do not require a left/right cone pairing
- we trust one boundary that looks geometrically consistent
- we assume a nominal track width
- we offset inward from that one boundary to create a driving line
- we then stabilize that line over time so it does not jump around every frame

The important part is not just the geometry. The planner is built around **trust**:

- trust in individual cone tracks
- trust in a chosen boundary chain
- trust in the currently stored driving line

## 1. What the planner is trying to do

The planner assumes that if one side of the track is reliable, that is enough to keep driving.

So instead of trying to reconstruct the whole corridor from both sides, it does this:

1. Keep a short-term memory of cones over time.
2. Ignore cones that still look uncertain.
3. Build a smooth ordered chain along one side of the track.
4. Shift that chain inward by half the expected track width.
5. Smooth and validate the result.
6. Blend it with a stored line from previous frames so the car sees a stable trajectory instead of a flickering one.

That makes the planner robust when one side of the track is weak, missing, or temporarily noisy.

## 2. How cone trust is built before planning

The planner does not work directly on one raw frame of detections. It first relies on a cone-memory stage that keeps track of cones over time.

### 2.1 Repeated sightings matter

Each cone track gains confidence when it is seen again and loses confidence when it is missed.

In the current implementation:

- on a successful update, confidence is increased
- on a missed update, confidence is multiplied down by 10%
- a new track starts with moderate confidence, especially if lidar contributed
- a track becomes confirmed only after 3 hits
- track position itself is also low-pass filtered over time, with lidar updates trusted more strongly than camera updates

So a cone that has been seen many times is trusted more than a cone that appeared once.

This is the first reason the planner becomes stable: unreliable cones are filtered out before they can steer the path.

### 2.2 Tentative, confirmed, and stale cones

The tracker effectively keeps cones in three categories:

- `tentative`: too new to trust yet
- `confirmed`: repeatedly observed and currently trusted
- `stale`: previously trusted, but not seen very recently

For planning:

- tentative cones are effectively ignored
- confirmed cones are fully usable
- stale cones are still allowed for a short time, because briefly losing sight of a good cone should not immediately destroy the plan

In the current setup:

- a cone becomes stale after about 0.6 s without an update
- stale cones may still be passed to the planner for about 1.5 s

That means the planner is allowed to “coast” through short perception dropouts.

### 2.3 Color trust also has memory

The tracker does not treat cone color as a one-frame truth either.

Instead it keeps a running class belief:

- every new camera classification nudges the stored color belief
- old color evidence decays slowly
- the current label does not switch immediately when a new label appears
- a new label must beat the current label by a margin before the tracker changes its mind

So if a cone has looked blue for a while, one uncertain yellow frame will usually not flip it.

This is important because the planner needs a consistent idea of which side of the track a cone belongs to.

### 2.4 What actually reaches the planner

By the time the planner sees cones, each cone already carries:

- a position that has been smoothed over time
- a track confidence
- a color belief that has temporal hysteresis
- a seen-count history

Then the planner applies another confidence gate on top:

- only cones above the planning confidence threshold are allowed in
- tentative tracks are explicitly forced to zero confidence, so they never enter planning

In the current setup the planning confidence threshold is `0.3`.

The position smoothing matters too. In the current setup:

- lidar pulls a stored cone position about 25% toward the new observation on each update
- camera pulls it about 15%

So even before the planner starts building a boundary, cone positions are already being stabilized over time.

## 3. Which cones the single-boundary planner uses

The single-boundary planner only wants cones that can actually define one side of the track.

So before building a path, it rejects cones that are:

- too far away
- too far behind the car
- below the confidence threshold
- not confidently assigned to a left or right boundary color

One subtle but important point:

- ambiguous cones can be useful in other reasoning stages
- but the pure single-boundary chain itself is built from cones that already look like a specific side

So the actual side-chain geometry is built from cones we already trust to belong to one boundary.

Another practical detail in the current implementation:

- there is still an early gate requiring at least 4 confidently colored cones in the scene before planning starts at all
- once that gate is passed, the chosen single boundary itself can be as short as 2 cones in forced mode

So the entry gate is stricter than the final chain-length requirement.

## 4. How one boundary chain is built

Once the usable cones are selected, the planner tries to turn all cones on one side into an ordered boundary chain.

### 4.1 Seed selection

The first cone in the chain is chosen to be:

- in front of the vehicle
- as near as possible in forward distance
- and as close to the centerline region as possible among those near-forward candidates

This gives the chain a sensible starting point near the car.

### 4.2 Greedy chain growth

After that, the chain is extended one cone at a time.

A candidate next cone is accepted only if it satisfies geometric rules such as:

- it must represent forward progress
- it must not be too close or too far from the current cone
- it must not require a heading change that is too sharp
- it should not be “shadowed” by a nearer cone in nearly the same direction

So the planner is not simply sorting by x-position. It is trying to grow a boundary that looks like a plausible continuous wall of cones.

This is the second place where trust appears:

- a cone may be individually real
- but if using it would make the boundary zig-zag, reverse direction, or jump too far, it is not trusted as part of the chain

## 5. How the planner chooses which side to trust

Forced single-boundary planning still looks at both sides if both are available.

It then picks the better side using a simple preference order:

1. larger forward extent
2. more cones
3. smoother average turning behavior

So the chosen side is the one that looks most useful as a guide for the road ahead.

This is important:

- the planner is not trying to be fair to both sides
- it is trying to find the one side that currently gives the most trustworthy forward structure

## 6. How one boundary becomes a driving line

After one side is selected, the planner creates the driving line by offsetting inward from that boundary.

### 6.1 Local tangent estimation

At every boundary point, the planner estimates the local direction of the boundary:

- forward difference at the start
- backward difference at the end
- centered difference in the middle

That gives a local tangent direction along the cone chain.

### 6.2 Inward normal

From that tangent, the planner computes the inward normal:

- for the left boundary, inward points to the right
- for the right boundary, inward points to the left

### 6.3 Width prior

The boundary is then shifted inward by **half of the expected track width**.

In the current setup:

- nominal track width starts at `3.6 m`
- so the single-boundary offset is initially `1.8 m`

This matters conceptually:

- a midpoint planner measures the center from both sides
- a single-boundary planner cannot do that
- so it must rely on a width prior

In pure forced single-boundary operation, that width is mostly a remembered assumption rather than a freshly measured quantity.

So the centerline is really:

`trusted boundary + trusted inward offset`

## 7. How the raw line is cleaned up

The raw offset curve is not sent directly to the controller.

It is first regularized:

- a short moving average smooths local jitter
- the path is resampled to even spacing
- the total planning length is capped

In the current setup:

- smoothing window: 3 points
- resampling spacing: `0.5 m`
- maximum planned length: `30 m`

This turns a cone-to-cone construction into a controller-friendly trajectory.

## 8. How fresh path candidates are judged

The planner does not trust every newly generated line just because the geometry process succeeded.

A fresh candidate is checked for basic plausibility:

- enough points
- enough forward extent
- finite geometry
- no self-crossing
- no sudden heading flip at the start
- no excessive local kink
- no excessive near-field jump compared with the previous trusted path

In forced mode the rules are intentionally relaxed so the system keeps producing a path rather than giving up too easily.

In the current forced setup the planner can accept as little as:

- 2 points
- about `1 m` of forward extent

And the near-field continuity threshold is deliberately very loose compared with normal operation.

That is consistent with the design goal of forced single-boundary mode:

- keep driving from one side if there is any reasonable guide left

There is also a second, softer acceptance layer after this:

- even if the strict validator is unhappy, a raw single-boundary path can still be allowed to refresh the stored trajectory if it is long enough, finite, and does not jump too far from the currently trusted line

So the planner distinguishes between:

- “good enough to call fully valid”
- and “not perfect, but still safe enough to keep the stored line up to date”

## 9. The stored midline: where trajectory trust really lives

The most important stability mechanism is not the raw boundary offset. It is the **stored midline buffer**.

This is the planner’s memory of “the line I currently believe in”.

Every new frame produces a candidate line, but that candidate does not automatically replace the stored one.

### 9.1 A new line must not jump too far

Before the stored line is updated, the new candidate is compared against it over the forward horizon.

If the new line jumps too far sideways from the stored line, the update is rejected.

In the current setup:

- if the jump exceeds about `1.0 m`, the update is rejected

This protects against:

- one bad frame
- a temporary cone swap
- a wrong cone suddenly entering the chain
- a momentary collapse of the chosen boundary

### 9.2 The stored line has its own confidence

The planner keeps a confidence value for the stored line itself.

This confidence rises when good updates keep arriving and falls when they do not.

In the current setup:

- every accepted update adds `0.25`
- every rejected or missing update subtracts `0.10`
- if confidence falls below `0.20`, the stored line is discarded
- if the stored line is not refreshed for about `2.5 s`, it is discarded

So the path memory is not eternal. It survives only while recent evidence keeps supporting it.

This is exactly the behavior you described:

- when the planner has seen a stable line for a while, it becomes willing to keep trusting it
- when evidence stops supporting that line, trust decays and eventually the line is dropped

## 10. Why some parts of the line are harder to change than others

This is one of the most deliberate parts of the design.

When a new candidate is accepted, the planner does **not** simply overwrite the stored line point by point.

Instead it resamples both the stored line and the new candidate at fixed stations ahead of the car, then blends them differently depending on distance ahead.

### 10.1 Immediate control handoff zone

Very near the vehicle, the planner uses the fresh candidate directly.

In the current setup:

- for roughly the first `1.5 m` ahead, the fresh candidate is taken as-is

This is done so the controller is not forced to follow an outdated line right at the vehicle.

### 10.2 Near zone: strongly resistant to change

Just beyond the handoff zone, the stored line is very stubborn.

In the current setup, roughly from `1.5 m` to `4 m` ahead:

- only about `6%` of the requested lateral change is accepted per update
- and even that is clipped to at most `10 cm`

So if a new frame tries to move this part of the line sideways, the planner only allows a tiny correction.

### 10.3 Mid zone: moderate resistance

Further ahead, the planner loosens up.

Roughly from `4 m` to `12 m` ahead:

- about `18%` of the lateral correction is accepted per update
- clipped to at most `20 cm`

### 10.4 Far zone: most willing to adapt

Farther out, the planner is more willing to reshape the line.

Beyond about `12 m`:

- about `35%` of the lateral correction is accepted per update
- clipped to at most `40 cm`

### 10.5 Why this makes sense

This gives the path a very intentional behavior:

- closest to the car, the line must be controllable right now
- slightly farther ahead, the line should not wobble just because cone detections moved around
- farther out, the line can adapt more because that part of the future path is naturally more uncertain

So the planner is not equally stiff everywhere.

It is **spatially selective**:

- some regions are conservative
- some regions are adaptable

That is why the line can feel “hard to move” in some places after being stable for a while.

## 11. The planner preserves forward progress but stabilizes sideways shape

Another subtle design choice:

- the planner mostly keeps the fresh candidate’s forward progression
- but it damps the **lateral** movement of the stored line

So the line is allowed to keep advancing naturally down the track, while sideways changes are treated with suspicion.

This is important because blindly smoothing both directions would freeze the shape too much and can make the controller chase an outdated heading.

## 12. Anchoring near the car

After blending, the planner still does one more stabilization step near the vehicle:

- points slightly behind the car are dropped
- the first path point is forced to sit at the vehicle origin
- the first short section ahead is tapered toward zero lateral offset

In practice, this means:

- the planner refuses to let the path start a noticeable distance to the left or right of the car
- the controller always receives a path that begins in a physically sensible place

Without this anchor, even a good stored line can become awkward for steering if its first point drifts sideways.

## 13. What happens when the fresh candidate is bad

If the current frame does not produce a trustworthy update, the planner does not immediately give up.

It can keep publishing the last valid stored line for a limited amount of time.

This is the hold behavior.

In the current setup:

- the last valid line may be held for about `2.5 s`
- once the planner has entered hold mode, it requires 2 consecutive clean frames before it stops holding and goes fully fresh again

This prevents rapid switching between:

- “use old line”
- “use new line”
- “use old line”
- “use new line”

That hysteresis is another form of trust: the planner demands repeated evidence before it fully believes that the situation has stabilized again.

## 14. How the planner avoids getting stuck in old beliefs

A path memory can become too conservative if it rejects every new shape forever.

To prevent that, the planner has a recovery rule:

- if the jump-rejection test rejects several consecutive updates, the stored path is cleared

In the current setup:

- after 3 consecutive jump rejections, the stored line is reset

This is important because it prevents the system from becoming trapped by yesterday’s geometry when the track evidence has genuinely changed.

So the planner balances two opposite goals:

- do not jump too easily
- do not cling forever to an outdated line

## 15. What “confidence” means in this planner

There are really three different confidence layers:

### 15.1 Cone confidence

This answers:

- is this cone track real enough to use?
- has it been seen enough times?
- has it survived long enough to stop being tentative?

### 15.2 Boundary confidence

This answers:

- do these cones form a believable one-sided boundary?
- do they progress forward?
- are they spaced sensibly?
- does the chain avoid unreasonable turns and jumps?

### 15.3 Trajectory confidence

This answers:

- does the new line agree enough with the stored line to update it?
- has the stored line been refreshed often enough recently?
- should the planner trust the current memory more than one noisy frame?

That last layer is what makes the system feel stable over time.

## 16. What the forced single-boundary planner does not rely on

This planner does **not** need:

- cone pairings across the track
- midpoint construction between left and right cones
- live width estimation from paired boundaries

Its core assumption is narrower and simpler:

- one reliable side of the track
- one reasonable width prior
- one temporally stabilized stored driving line

## 17. Practical summary

The forced single-boundary planner works because it combines:

- repeated-observation trust for cones
- geometric consistency checks for building one boundary
- a nominal inward offset to create the driving line
- strong temporal memory for the final trajectory

So the planner is not just “offset a boundary and hope”.

It is really:

1. trust cones only after repeated evidence
2. trust a boundary only if its geometry is coherent
3. trust a new line only if it agrees with recent history
4. allow the stored line to change slowly in the places where sudden change is dangerous
5. let go of old beliefs when evidence keeps disagreeing for long enough

That is the central principle behind the forced single-boundary mode.
