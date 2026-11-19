# M5 — Temporal Behaviour Estimation (Part B)

**Document:** `docs/m5_temporal_behaviour_estimation.md`  
**Milestone:** M5  
**System:** Vision-Based First Contact  
**Status:** Design specification — implementation pending approval  
**Depends on:** M3 Single-Person Tracking, M4 Frame-Level Readiness Perception  
**Coordinate system:** Image-space coordinates from the RGB camera

---

## 1. Purpose

M5 implements temporal behaviour estimation for the Vision-Based First Contact system.

M3 provides a persistent person track and timestamped bounding-box observations. M4 provides frame-level observations describing person presence, bounding-box validity, face quality, interaction-zone membership, and approximate head orientation.

These observations describe what is visible in an individual frame. They do not establish how a person is behaving over time.

M5 addresses this gap by maintaining a short history of observations and estimating:

1. Image-space movement and trajectory.
2. Direction of movement.
3. Whether a person is approaching the interaction zone.
4. Image-space velocity and displacement.
5. Whether a person is stationary or moving.
6. Whether the person has maintained the required temporal conditions for two seconds.
7. The reliability and uncertainty of these temporal estimates.

M5 is a temporal estimation component, not a final first-contact decision-maker.

The output will be consumed by M6, which will combine frame-level and temporal information to determine overall interaction readiness.

### 1.1 Engineering approach

M5 will initially use deterministic, geometry-based and timestamp-based algorithms.

No additional deep-learning model is required.

The implementation will reuse:
- M3 bounding-box tracking and track identity.
- M4 interaction-zone geometry and frame-level observations.
- Timestamped observations from the perception pipeline.

The first implementation will prioritize:
- Deterministic behaviour.
- Explicit uncertainty handling.
- Configurable thresholds.
- Low memory consumption.
- Reproducible testing.
- Interpretability of every temporal decision.

An LSTM, GRU, Transformer, optical-flow model, or other learned temporal model is outside the initial M5 scope.

---

## 2. Scope and boundaries

### 2.1 In scope

| Component | Responsibility |
|---|---|
| Observation history | Maintain timestamped bounding-box centre observations |
| Movement estimation | Estimate image-space velocity and displacement |
| Trajectory estimation | Maintain recent image-space trajectory |
| Direction estimation | Estimate the direction of movement |
| Approach estimation | Determine whether movement is directed towards the interaction zone |
| Stationary estimation | Determine whether the person has stopped moving |
| Temporal smoothing | Reduce sensitivity to frame-level measurement noise |
| Stability timer | Measure continuous satisfaction of temporal conditions |
| Uncertainty management | Represent insufficient, unreliable, or discontinuous observations |
| Evaluation | Test temporal estimates using controlled and labelled sequences |

### 2.2 Explicitly out of scope

M5 must not implement or modify:

- Person detection or YOLO inference.
- Bounding-box smoothing owned by M2.
- Kalman tracking, IoU association, or track identity management owned by M3.
- Face detection or face quality estimation owned by M4.
- Face recognition or identity recognition.
- Body-pose estimation.
- Gaze estimation.
- Emotion or expression recognition.
- Metric distance estimation in metres.
- Physical velocity estimation in metres per second.
- Multi-person tracking or target selection.
- Final first-contact readiness decisions.
- Robot speech, movement, or interaction behaviour.
- Learned temporal models.

M5 must not treat a predicted Kalman position as an actual visual observation.

### 2.3 Coordinate-system limitation

All movement estimates are based on image-space coordinates.

A movement of 20 pixels does not represent a fixed physical distance. The same physical movement can produce different pixel displacements depending on the person's distance from the camera, camera geometry, and perspective.

Therefore:

- Velocity is expressed in pixels per second.
- Displacement is expressed in pixels.
- Trajectory is an image-space trajectory.
- Approach means movement towards an image-space interaction-zone reference point.
- Stationary means sufficiently small image-space movement under the configured thresholds.

M5 does not establish that a person is physically within a particular distance of the robot.

Laptop-camera experiments are suitable for prototyping and algorithm verification, but do not validate metric distance or deployment performance on the final robot camera.

---

## 3. Dependencies and input contract

### 3.1 M3 input

M5 consumes observations from the currently active M3 track.

The minimum required information is:

| Field | Description |
|---|---|
| `track_id` | Persistent identifier of the tracked person |
| `timestamp` | Monotonic observation timestamp |
| `bbox` | Current observed bounding box |
| `tracking_status` | Whether the track is currently supported by a valid observation |
| `observation_source` | Whether the position is based on an actual detection or prediction, if exposed by M3 |

M5 accepts only positions derived from actual observed detections.

Predicted Kalman positions must not enter the motion-history buffer.

If the existing M3 interface does not expose a suitable observed-versus-predicted indicator, the integration must use the existing tracking status or introduce the smallest necessary interface adaptation. M5 must not infer observation validity merely from the presence of a predicted bounding box.

### 3.2 M4 input

M5 consumes the relevant M4 observations for the same person and frame.

The five M4 observation categories are:

1. Person presence.
2. Bounding-box validity.
3. Face visibility and quality.
4. Interaction-zone membership.
5. Robot-facing head orientation.

Each observation follows the M4 contract:

```python
{
    "value": ...,
    "confidence": ...,
    "status": ...
}
```

The status values are:

- `VALID`
- `UNCERTAIN`
- `INVALID`

The value `UNKNOWN` is an observation value, not an observation status.

M5 must preserve this distinction.

An observation with status `INVALID` or value `UNKNOWN` must not be interpreted as a valid negative observation.

### 3.3 M5 internal observation

M5 uses the centre of the current observed bounding box:

\[
c_t = (x_t,y_t)
\]

For bounding box:

\[
B_t=(x_1,y_1,x_2,y_2)
\]

the centre is:

\[
x_t=\frac{x_1+x_2}{2}
\]

\[
y_t=\frac{y_1+y_2}{2}
\]

The normalized coordinates are optional internal representations only if required by the existing architecture. The selected algorithm uses the bounding-box centre; it does not use the bottom-centre or foot position.

A conceptual internal observation is:

```python
{
    "timestamp": 123.456,
    "track_id": 1,
    "center_x": 640.0,
    "center_y": 350.0,
    "observed": True,
    "m4_observations": {...}
}
```

This is a conceptual schema. The final implementation should follow existing repository conventions rather than introduce unnecessary duplicate representations.

### 3.4 Timestamp contract

All temporal calculations must use monotonic timestamps.

The system must not use wall-clock time for velocity, displacement, observation gaps, or stability duration.

For two observations:

\[
\Delta t=t_2-t_1
\]

The interval must be strictly positive.

Duplicate or non-monotonic timestamps must be rejected as invalid temporal input.

---

## 4. Design decisions and alternatives considered

This section records the decisions established during M5 requirements discussions. These choices are frozen for the initial implementation unless explicitly reopened.

### Q1 — Position representation

**Selected: A — Bounding-box centre only.**

\[
c_t=(x_t,y_t)
\]

Alternatives:

| Option | Representation | Reason for not selecting |
|---|---|---|
| A | Bounding-box centre | Selected |
| B | Bounding-box bottom-centre | Depends more strongly on visible lower-body geometry and foot visibility |
| C | Pose-based reference point | Requires pose estimation and introduces another model dependency |
 
**Reasoning:** The bounding-box centre is directly available from M3, does not depend on visible feet, and is consistent with the position representation already used by the tracking state.

No pose model or additional visual landmark is required.

### Q2 — Approach definition

**Selected: C — Combine movement direction and decreasing distance to the interaction zone.**

An approach should satisfy both:

1. Movement is directed towards the interaction zone.
2. Distance to the zone reference point decreases over time.

Alternatives:

| Option | Approach definition | Reason for not selecting |
|---|---|---|
| A | Distance reduction only | Distance may decrease due to lateral movement or noisy measurements without sufficiently directed movement |
| B | Direction towards zone only | Direction can point towards the zone even when the person is not actually getting closer |
| C | Both conditions | Selected because it requires complementary evidence |

**Reasoning:** Combining directional evidence and distance-trend evidence reduces false approach detections.

### Q2.1 — Distance reference

**Selected: A — Interaction-zone centroid.**

Let \(z=(x_z,y_z)\) denote the centroid of the configured interaction-zone polygon.

Alternatives:

| Option | Reference | Reason for not selecting |
|---|---|---|
| A | Polygon centroid | Selected as a simple, fixed reference |
| B | Nearest point on polygon boundary | More complex geometric distance calculation |
| C | Distance to polygon itself | Distance becomes zero anywhere inside the polygon, which is less useful for measuring progress towards a fixed target |

**Reasoning:** The centroid provides one stable reference for both directional and distance-trend calculations.

The selected approach is an image-space approximation. It does not claim that the centroid is the physically correct approach point for every possible polygon shape.

### Q3 — Stationary definition

**Selected: C — Combine velocity and displacement thresholds.**

Both conditions must hold.

\[
\text{STATIONARY}_t =
(v_t < \tau_v)
\land
(D_t < \tau_D)
\]

Alternatives:

| Option | Definition | Reason for not selecting |
|---|---|---|
| A | Velocity threshold only | A low instantaneous estimate can occur despite substantial movement over a longer period |
| B | Displacement threshold only | Net displacement can be small even when a person moves back and forth |
| C | Both velocity and displacement | Selected to combine local movement rate and accumulated position change |

**Reasoning:** The two tests provide complementary evidence and reduce dependence on any single noisy measurement.

### Q3.1 — Velocity threshold

**Selected: 20 pixels per second.**

\[
\tau_v=20\text{ px/s}
\]

Alternatives considered:

| Option | Threshold |
|---|---:|
| A | 5 px/s |
| B | 10 px/s |
| C | 15 px/s |
| D | 20 px/s — selected |

**Reasoning:** The selected threshold is more tolerant of small bounding-box fluctuations than the stricter alternatives.

It is an initial engineering parameter, not an empirically established universal stationary threshold.

### Q3.2 — Displacement threshold

**Selected: 20 pixels over one second.**

\[
\tau_D=20\text{ px}
\]

Alternatives considered:

| Option | Threshold over one second |
|---|---:|
| A | 5 px |
| B | 10 px |
| C | 15 px |
| D | 20 px — selected |

**Reasoning:** A 20-pixel allowance provides a practical initial tolerance for small image-space variations while still requiring the person's centre to remain relatively stable.

This threshold must be evaluated using actual recorded sequences.

### Q4 — Tracking loss and unreliable tracking

**Selected: A — Immediate reset.**

Alternatives:

| Option | Behaviour | Reason for not selecting |
|---|---|---|
| A | Clear temporal history immediately | Selected; simplest and most conservative |
| B | Retain history during a short grace period | Adds recovery state and stale-history management |
| C | Continue using Kalman predictions | Can mistake predicted motion for observed human behaviour |

**Selected behaviour:**

- Clear observation history.
- Reset velocity, displacement, and trajectory estimates.
- Reset approach confirmation.
- Reset stationary state.
- Reset the stability timer.
- Set affected temporal outputs to `UNKNOWN` with an appropriate reliability status.
- Do not use predicted Kalman positions to continue temporal estimation.
- Reinitialize after fresh observations resume.

**Reasoning:** M5 should estimate behaviour from observed evidence, not extrapolated positions. Immediate reset also avoids additional recovery-state complexity.

### Q5 — Stability timing

**Selected: A — Continuous stability with noise tolerance.**

The person must satisfy the required temporal conditions continuously for two seconds, subject to a limited cumulative tolerance for short, valid measurement fluctuations.

Alternatives:

| Option | Timer behaviour | Reason for not selecting |
|---|---|---|
| A | Continuous stability with tolerance | Selected |
| B | Pause and resume | Can preserve accumulated evidence across interruptions |
| C | Accumulate qualifying time across an observation window | Does not enforce continuous stability |

**Reasoning:** First-contact behaviour should be conservative. A person who becomes stationary, moves away, and becomes stationary again should not automatically be treated as having maintained one continuous stable period.

### Q5.1 — Noise tolerance mechanism

**Selected: A — Cumulative tolerance.**

\[
T_{\text{tolerance}}=200\text{ ms}
\]

Alternatives:

| Option | Behaviour | Reason for not selecting |
|---|---|---|
| A | Cumulative violation allowance | Selected; tolerates several brief fluctuations |
| B | Allow one continuous violation up to a limit | Less tolerant of multiple small fluctuations |

**Selected behaviour:**

- The stability attempt has a cumulative violation budget of 200 ms.
- Brief valid-observation threshold excursions consume that budget.
- The timer continues during tolerated excursions.
- When cumulative violation duration exceeds 200 ms, the attempt resets.
- Tracking loss and required `UNKNOWN` observations reset immediately; they do not consume the tolerance budget.

**Reasoning:** Multiple small fluctuations can occur due to detector and tracker noise. Cumulative tolerance handles these without turning the timer into a pause-and-resume mechanism.

The maximum tolerated overcount in elapsed stability time is bounded by the configured cumulative allowance.

### Q6 — Initial observation requirement

**Selected: B — Five consecutive valid observations.**

Alternatives:

| Option | Consecutive observations | Approximate duration at 30 FPS |
|---|---:|---:|
| A | 3 | 100 ms |
| B | 5 — selected | Approximately 133–167 ms, depending on the duration convention |
| C | 10 | Approximately 300–333 ms |
| D | 15 | Approximately 467–500 ms |

**Reasoning:** Five observations provide a small initialization buffer without imposing a long delay.

M3's three-frame track confirmation and M5's five-observation initialization are separate requirements:

- M3 establishes track confirmation.
- M5 establishes sufficient temporal evidence for motion estimation.

M5 must not assume that an M3-confirmed track is automatically ready for motion estimation.

### Q7 — Temporal history window

**Selected: B — One second.**

Alternatives:

| Option | History window | Reason for not selecting |
|---|---|---|
| A | 0.5 seconds | More responsive but more sensitive to measurement noise |
| B | 1 second — selected | Balance between smoothing and responsiveness |
| C | 2 seconds | Smoother but slower to react to genuine behavioural changes |

**Reasoning:** A one-second window is suitable for the initial image-space motion estimator and is consistent with the one-second stationary displacement test.

### Q8 — Maximum observation gap

**Selected: B — 200 ms.**

Alternatives:

| Option | Maximum gap |
|---|---:|
| A | 100 ms |
| B | 200 ms — selected |
| C | 500 ms |

**Reasoning:** A 200 ms gap is more tolerant of occasional processing delays than 100 ms, while avoiding motion estimates over long periods without visual evidence.

At 30 FPS, one nominal frame interval is approximately 33.3 ms. A 200 ms gap corresponds to roughly six frame intervals.

A gap exceeding this limit is a temporal discontinuity. The current motion estimate must be invalidated and the five-observation initialization requirement must be applied again.

### Q9 — Approach confirmation duration

**Selected: C — 500 ms.**

Alternatives:

| Option | Confirmation duration | Reason for not selecting |
|---|---:|---|
| A | Immediate | Sensitive to transient movement |
| B | 300 ms | Less temporal evidence |
| C | 500 ms — selected | Balanced confirmation period |
| D | 1 second | Slower to identify approach |

**Reasoning:** Approach is a behavioural estimate, not a single-frame event. Requiring 500 ms of sustained evidence reduces false positives due to temporary direction changes or bounding-box jitter.

### Q10 — Temporal output reliability

**Selected: Three reliability statuses.**

- `VALID`
- `UNCERTAIN`
- `INVALID`

The estimated behaviour value is represented separately. `UNKNOWN` is a possible value when a meaningful estimate cannot be produced.

**Reasoning:** The system must distinguish what it estimates from how reliable that estimate is.

A negative behaviour estimate is not equivalent to an unavailable estimate.

For example:

- `NOT_APPROACHING` with `VALID` means sufficient evidence indicates no approach.
- `UNKNOWN` with `UNCERTAIN` means evidence exists but is insufficient or ambiguous.
- `UNKNOWN` with `INVALID` means the estimate cannot be computed.

---

## 5. Mathematical formulation

### 5.1 Notation

| Symbol | Meaning |
|---|---|
| \(t\) | Current monotonic timestamp |
| \(t_i\) | Timestamp of observation \(i\) |
| \(c_i\) | Bounding-box centre at observation \(i\) |
| \(x_i,y_i\) | Horizontal and vertical centre coordinates |
| \(z\) | Interaction-zone centroid |
| \(d_i\) | Distance from person centre to zone centroid |
| \(W\) | Motion-estimation history window |
| \(v_t\) | Estimated image-space velocity vector |
| \(\|v_t\|\) | Velocity magnitude |
| \(D_t\) | Displacement over the selected window |
| \(\tau_v\) | Stationary velocity threshold |
| \(\tau_D\) | Stationary displacement threshold |
| \(\tau_\theta\) | Minimum approach-direction cosine |
| \(\tau_a\) | Minimum approach closing-speed threshold |
| \(T_a\) | Approach confirmation duration |
| \(T_s\) | Required stability duration |
| \(T_{\text{tol}}\) | Cumulative stability violation tolerance |

### 5.2 Interaction-zone reference point

Let the interaction zone be a polygon with vertices:

\[
P=\{p_1,p_2,\ldots,p_n\}
\]

where:

\[
p_i=(x_i^p,y_i^p)
\]

The polygon centroid is:

\[
z=(x_z,y_z)
\]

For a polygon with signed area:

\[
A=\frac{1}{2}
\sum_{i=1}^{n}
(x_i^p y_{i+1}^p-x_{i+1}^p y_i^p)
\]

the centroid coordinates are:

\[
x_z=
\frac{1}{6A}
\sum_{i=1}^{n}
(x_i^p+x_{i+1}^p)
(x_i^p y_{i+1}^p-x_{i+1}^p y_i^p)
\]

\[
y_z=
\frac{1}{6A}
\sum_{i=1}^{n}
(y_i^p+y_{i+1}^p)
(x_i^p y_{i+1}^p-x_{i+1}^p y_i^p)
\]

Indices wrap around at the final vertex.

If the existing M4 configuration stores normalized polygon coordinates, the centroid should be calculated in that same coordinate system, or transformed consistently into pixel coordinates.

The implementation must not mix normalized and pixel coordinates in the same distance calculation.

A degenerate polygon has no valid centroid and must produce an invalid configuration rather than a fabricated reference point.

### 5.3 Person position

For each accepted observed bounding box:

\[
B_t=(x_1,y_1,x_2,y_2)
\]

the bounding-box centre is:

\[
c_t=
\begin{bmatrix}
x_t\\
y_t
\end{bmatrix}
=
\begin{bmatrix}
(x_1+x_2)/2\\
(y_1+y_2)/2
\end{bmatrix}
\]

Only the current observed detection is accepted.

A predicted bounding box from M3 is not a valid input observation.

### 5.4 Observation history

Maintain a timestamp-ordered buffer:

\[
H_t=
\{(t_i,c_i)\}_{i=1}^{N}
\]

The buffer contains valid observed positions within the configured one-second window:

\[
t-W \leq t_i \leq t
\]

where:

\[
W=1\text{ second}
\]

The buffer must be bounded in time and memory.

At the nominal 30 FPS, a one-second window contains approximately 30 observations, subject to actual processing rate.

The system must use timestamps rather than assuming a fixed sampling interval.

### 5.5 Observation continuity

For consecutive observations:

\[
\Delta t_i=t_i-t_{i-1}
\]

The temporal estimator accepts the interval only if:

\[
0 < \Delta t_i \leq 0.2\text{ seconds}
\]

If:

\[
\Delta t_i>0.2
\]

the observation sequence is considered discontinuous.

The motion history is cleared, the current estimate becomes unavailable, and initialization begins again using fresh observations.

A non-positive interval is rejected as invalid timestamp input.

### 5.6 Velocity estimation

The raw finite-difference velocity between consecutive observations is:

\[
v_i^{raw}=
\frac{c_i-c_{i-1}}{t_i-t_{i-1}}
\]

Its magnitude is:

\[
s_i^{raw}=\|v_i^{raw}\|_2
\]

A single finite difference can be sensitive to bounding-box jitter. Therefore, the initial design should estimate the velocity vector over the one-second history using timestamp-aware linear regression.

For the observations in the active history window, estimate:

\[
x_i \approx a_x+b_x(t_i-\bar{t})
\]

\[
y_i \approx a_y+b_y(t_i-\bar{t})
\]

where:

\[
\bar{t}=\frac{1}{N}\sum_{i=1}^{N}t_i
\]

The least-squares slope for the horizontal coordinate is:

\[
b_x=
\frac{
\sum_i(t_i-\bar{t})(x_i-\bar{x})
}{
\sum_i(t_i-\bar{t})^2
}
\]

Similarly:

\[
b_y=
\frac{
\sum_i(t_i-\bar{t})(y_i-\bar{y})
}{
\sum_i(t_i-\bar{t})^2
}
\]

The estimated velocity vector is:

\[
v_t=
\begin{bmatrix}
b_x\\
b_y
\end{bmatrix}
\]

and its magnitude is:

\[
s_t=\sqrt{b_x^2+b_y^2}
\]

Units:

\[
s_t \in \text{pixels/second}
\]

This estimator uses actual timestamps and tolerates irregular frame intervals.

If the history does not contain enough distinct timestamps to estimate a slope, the velocity output must be `UNKNOWN`.

### 5.7 Displacement estimation

Let \(c_{start}\) be the first valid centre in the selected one-second comparison window and \(c_t\) be the latest centre.

The net displacement is:

\[
D_t=\|c_t-c_{start}\|_2
\]

In coordinates:

\[
D_t=
\sqrt{
(x_t-x_{start})^2+
(y_t-y_{start})^2
}
\]

The displacement threshold is:

\[
\tau_D=20\text{ pixels}
\]

The displacement window must cover approximately one second. If the history does not cover the required interval, the stationary estimate must remain unavailable rather than treating a shorter interval as equivalent.

### 5.8 Stationary predicate

The configured velocity threshold is:

\[
\tau_v=20\text{ px/s}
\]

The configured displacement threshold is:

\[
\tau_D=20\text{ px}
\]

The stationary predicate is:

\[
S_t=
(s_t<20)
\land
(D_t<20)
\]

The person is classified as stationary only if both conditions hold.

The moving predicate is:

\[
M_t=\neg S_t
\]

However, this complement is valid only when both estimates are available and reliable.

If either estimate is unavailable or unreliable, the output must be `UNKNOWN`, not automatically `MOVING`.

Boundary comparisons must be consistent and tested. The initial implementation uses strict less-than comparisons for the stationary thresholds.

### 5.9 Distance to interaction-zone centroid

The distance from the person's centre to the zone centroid is:

\[
d_t=\|z-c_t\|_2
\]

or:

\[
d_t=
\sqrt{
(x_z-x_t)^2+(y_z-y_t)^2
}
\]

The distance is measured in pixels.

### 5.10 Movement direction towards the zone

The vector from the person towards the zone centroid is:

\[
r_t=z-c_t
\]

The estimated movement direction is represented by \(v_t\).

The cosine similarity is:

\[
\cos(\theta_t)=
\frac{v_t^\top r_t}
{\|v_t\|_2\|r_t\|_2}
\]

Interpretation:

| Cosine value | Interpretation |
|---|---|
| Close to +1 | Movement is directed towards the centroid |
| Close to 0 | Movement is approximately perpendicular |
| Close to -1 | Movement is directed away from the centroid |

The cosine is undefined if either vector has zero magnitude. In that case, direction is `UNKNOWN`.

### 5.11 Distance trend

Estimate the trend in person-to-zone distance over the active history window.

For each observation:

\[
d_i=\|z-c_i\|_2
\]

Fit a timestamp-aware linear model:

\[
d_i \approx a_d+b_d(t_i-\bar{t})
\]

where:

\[
b_d=
\frac{
\sum_i(t_i-\bar{t})(d_i-\bar{d})
}{
\sum_i(t_i-\bar{t})^2
}
\]

Interpretation:

- \(b_d<0\): distance is decreasing.
- \(b_d\approx0\): distance is approximately stable.
- \(b_d>0\): distance is increasing.

The quantity \(-b_d\) is the estimated closing speed towards the centroid, expressed in pixels per second.

### 5.12 Approach predicate

Approach requires both directional and distance-trend evidence.

Define:

\[
A_t^{direction}=
\cos(\theta_t)>\tau_\theta
\]

and:

\[
A_t^{distance}=
b_d<-\tau_a
\]

The instantaneous approach evidence is:

\[
A_t=
A_t^{direction}
\land
A_t^{distance}
\]

where:

- \(\tau_\theta\) is the minimum acceptable direction cosine.
- \(\tau_a\) is the minimum closing-speed threshold.

Both thresholds must be configurable.

**Important:** The exact numerical values of \(\tau_\theta\) and \(\tau_a\) were not frozen during requirements discussions. They must be documented as provisional calibration parameters, not as user-approved decisions.

The implementation must avoid classifying tiny numerical fluctuations as approach. A configurable minimum speed and direction threshold are therefore required.

The approach output becomes `APPROACHING` only after valid approach evidence has persisted for:

\[
T_a=0.5\text{ seconds}
\]

Before that duration is satisfied, the output remains `UNKNOWN` or `UNCERTAIN`, depending on whether evidence is unavailable or merely not yet confirmed.

If valid observations establish that the person is not approaching, the output may become `NOT_APPROACHING`.

### 5.13 Stability timer

The required stability duration is:

\[
T_s=2\text{ seconds}
\]

The cumulative violation budget is:

\[
T_{\text{tol}}=0.2\text{ seconds}
\]

The timer is based on monotonic timestamps.

Define an eligibility predicate:

\[
E_t=
P_t
\land
B_t
\land
Z_t
\land
S_t
\]

where:

- \(P_t\): person presence is valid and detected.
- \(B_t\): bounding-box validity is valid.
- \(Z_t\): the person satisfies the configured interaction-zone condition.
- \(S_t\): stationary predicate is valid and true.

This is a temporal stability indicator, not a final readiness predicate.

M5 must not independently declare first-contact readiness.

Face quality and robot-facing orientation remain M4 observations and are available to M6. Their role in final readiness is outside M5's authority.

#### Timer accumulation

Let:

\[
T_{start}
\]

be the monotonic timestamp at which the current stability attempt starts.

The elapsed attempt time is:

\[
T_{elapsed}=t-T_{start}
\]

Let:

\[
T_{violation}
\]

be the cumulative duration of tolerated, valid-observation threshold excursions.

The attempt remains active while:

\[
T_{violation}\leq0.2
\]

and all hard-reset conditions remain absent.

The stability timer completes when:

\[
T_{elapsed}\geq2
\]

subject to the cumulative tolerance constraint.

A tolerated violation does not pause elapsed time. It consumes the violation budget while the continuous attempt remains active.

#### Hard reset conditions

The following immediately reset the stability attempt:

- M3 tracking loss.
- Track identity change.
- Required M4 observation becomes `UNKNOWN`.
- Required observation becomes invalid.
- Temporal observation gap exceeds 200 ms.
- Timestamp discontinuity.
- Confirmed movement or other violation exceeding the remaining cumulative tolerance budget.

The tolerance is intended for short, valid-observation fluctuations in threshold-based estimates. It is not a grace period for missing observations or tracking loss.

---

## 6. Algorithms and processing pipeline

### 6.1 High-level pipeline

```text
M3 tracked-person observation
          |
          v
Validate track identity and timestamp
          |
          v
Check observed vs predicted position
          |
          v
Validate M4 required observations
          |
          v
Compute bounding-box centre
          |
          v
Update bounded observation history
          |
          v
Check initialization requirement
          |
          v
Estimate image-space motion
          |
          +-----------------------------+
          |                             |
          v                             v
    Approach estimator           Stationary estimator
          |                             |
          v                             v
    Direction + distance          Velocity + displacement
    trend + confirmation                 |
          |                             |
          +-------------+---------------+
                        |
                        v
                Stability timer
                        |
                        v
            Temporal output and status
                        |
                        v
                       M6
```

### 6.2 Algorithm 1 — Observation ingestion

**Input:** M3 track observation and corresponding M4 observations.

**Output:** Accepted internal observation or a reset/invalid result.

Procedure:

1. Verify that the active track identity is valid.
2. Verify that the position comes from an actual observed detection.
3. Validate timestamp monotonicity.
4. Validate bounding-box coordinates and derive the centre.
5. Verify the required M4 observation contract.
6. Check the interval from the previous accepted observation.
7. If the interval exceeds 200 ms, invalidate temporal continuity and restart initialization.
8. Otherwise append the observation to the history buffer.
9. Remove observations older than the one-second history window.
10. Check whether five consecutive valid observations have been collected.

A predicted position must never be inserted into the observation history.

### 6.3 Algorithm 2 — Initialization

**Requirement:** Five consecutive valid observations.

Procedure:

1. Start with an empty history after a reset.
2. Accept valid observed centres with monotonic timestamps.
3. Increment the initialization count for each consecutive valid observation.
4. If a required observation is invalid, or tracking is lost, clear the count and history.
5. Once five consecutive valid observations are collected, mark motion estimation as initialized.
6. Do not emit a valid movement estimate until the estimator also has sufficient temporal coverage for the requested calculation.

The five-observation requirement is an initialization minimum, not a substitute for the one-second displacement window.

Consequently, stationary estimation may remain `UNKNOWN` until sufficient one-second history exists.

### 6.4 Algorithm 3 — Motion estimation

1. Select observations in the one-second time window.
2. Verify timestamp continuity.
3. Estimate horizontal and vertical velocity using timestamp-aware least squares.
4. Compute velocity magnitude.
5. Compute net displacement across the required one-second interval.
6. Return velocity and displacement with reliability status.
7. If the history is insufficient, return `UNKNOWN`.

No additional smoothing model is required in the first implementation. The regression-based estimator provides the initial temporal smoothing.

Any additional filtering must be justified by labelled sequence evaluation rather than added speculatively.

### 6.5 Algorithm 4 — Approach estimation

1. Compute the zone centroid from the M4 interaction-zone configuration.
2. Calculate distance from each observed person centre to the centroid.
3. Estimate the movement direction from the fitted velocity vector.
4. Calculate cosine similarity between movement direction and the direction towards the centroid.
5. Fit a distance-versus-time trend.
6. Evaluate the direction and closing-speed thresholds.
7. If both conditions hold, start or continue approach confirmation.
8. Require 500 ms of valid approach evidence.
9. Emit `APPROACHING` once confirmed.
10. If valid evidence indicates otherwise, emit `NOT_APPROACHING`.
11. If required evidence is insufficient, emit `UNKNOWN` with an appropriate status.
12. Reset confirmation on tracking loss or temporal discontinuity.

The approach confirmation timer must use timestamps, not frame counts.

### 6.6 Algorithm 5 — Stationary estimation

1. Obtain the velocity magnitude from the motion estimator.
2. Obtain the displacement over one second.
3. Verify both estimates are valid.
4. Evaluate:

\[
s_t<20
\]

and:

\[
D_t<20
\]

5. If both conditions hold, emit `STATIONARY`.
6. If reliable evidence establishes movement, emit `MOVING`.
7. If the evidence is unavailable or uncertain, emit `UNKNOWN` with the appropriate reliability status.

### 6.7 Algorithm 6 — Stability timer

1. Verify that the person has a valid track.
2. Verify required M4 person-presence and bounding-box observations.
3. Verify that the person is inside the interaction zone.
4. Verify a valid stationary estimate.
5. If the eligibility predicate becomes true, start the stability attempt.
6. Measure elapsed time using a monotonic clock.
7. For a brief valid threshold excursion, add its duration to cumulative violation time.
8. Continue the attempt if cumulative violation time remains within 200 ms.
9. Reset immediately on tracking loss, required `UNKNOWN`/invalid observation, or temporal discontinuity.
10. Reset when cumulative violation time exceeds 200 ms.
11. Emit `COMPLETE` when elapsed time reaches two seconds and the attempt has not been reset.

The timer must not be started or completed based only on a frame count.

---

## 7. Configuration parameters

All operational thresholds must be configurable using the existing project configuration conventions.

The following table distinguishes frozen decisions from provisional parameters.

| Parameter | Initial value | Decision status | Description |
|---|---:|---|---|
| `history_window_seconds` | 1.0 | Frozen | Motion history window |
| `max_observation_gap_seconds` | 0.2 | Frozen | Maximum consecutive observation gap |
| `initial_valid_observations` | 5 | Frozen | Initialization requirement |
| `stationary_velocity_threshold_px_s` | 20.0 | Frozen | Maximum velocity for stationary classification |
| `stationary_displacement_threshold_px` | 20.0 | Frozen | Maximum displacement over one second |
| `stationary_displacement_window_seconds` | 1.0 | Frozen | Displacement comparison window |
| `approach_confirmation_seconds` | 0.5 | Frozen | Required approach evidence duration |
| `stability_duration_seconds` | 2.0 | Frozen | Required temporal stability |
| `stability_violation_tolerance_seconds` | 0.2 | Frozen | Cumulative tolerated violation duration |
| `approach_min_direction_cosine` | To calibrate | Provisional | Minimum directional agreement |
| `approach_min_closing_speed_px_s` | To calibrate | Provisional | Minimum rate of distance reduction |
| `velocity_estimator` | Timestamp-aware linear regression | Design choice | Motion estimation method |

The implementation must not silently introduce arbitrary numerical values for the two provisional approach thresholds.

If defaults are required to run experiments, they must be explicitly marked provisional and recorded in the experiment configuration.

### 7.1 Configuration validation

Validate configuration during component initialization.

Reject:
- Non-positive history windows.
- Non-positive observation-gap limits.
- Non-positive initialization counts.
- Negative velocity or displacement thresholds.
- Non-positive stability duration.
- Negative violation tolerance.
- Invalid approach cosine values outside the mathematically valid range.
- Negative closing-speed thresholds.
- Invalid interaction-zone geometry.
- A degenerate zone polygon or undefined centroid.

Also check consistency:

- The displacement window must be supported by the history window.
- The maximum observation gap must be compatible with the intended sampling rate.
- The approach confirmation duration must be achievable under the configured observation rate.
- The stability tolerance must not exceed the stability duration.

---

## 8. Temporal output contract

M5 must expose behaviour values and reliability separately.

### 8.1 Approach output

Possible values:

- `APPROACHING`
- `NOT_APPROACHING`
- `UNKNOWN`

### 8.2 Movement output

Possible values:

- `MOVING`
- `STATIONARY`
- `UNKNOWN`

### 8.3 Stability output

Possible values:

- `NOT_STARTED`
- `ACCUMULATING`
- `COMPLETE`
- `RESET`
- `UNKNOWN`

### 8.4 Reliability status

Every temporal observation must carry one of:

- `VALID`
- `UNCERTAIN`
- `INVALID`

Interpretation:

| Status | Meaning |
|---|---|
| `VALID` | Sufficient reliable observations support the estimate |
| `UNCERTAIN` | Observations exist, but the estimate is ambiguous or not sufficiently confirmed |
| `INVALID` | The estimate cannot be computed due to invalid input, discontinuity, or unavailable required evidence |

The value `UNKNOWN` is not a reliability status.

### 8.5 Conceptual output schema

```python
{
    "timestamp": 123.456,
    "track_id": 1,

    "approach": {
        "value": "APPROACHING",
        "confidence": None,
        "status": "VALID"
    },

    "movement": {
        "value": "STATIONARY",
        "confidence": None,
        "status": "VALID"
    },

    "motion": {
        "velocity_px_s": 8.4,
        "displacement_px": 6.2,
        "distance_to_zone_centroid_px": 125.0,
        "direction_cosine": None
    },

    "stability": {
        "value": "ACCUMULATING",
        "elapsed_seconds": 1.25,
        "violation_seconds": 0.05,
        "status": "VALID"
    }
}
```

This schema is illustrative. It does not require a particular class hierarchy or serialization format.

No artificial probability or confidence score should be generated for deterministic geometric predicates.

If confidence is not genuinely available, use `None`.

---

## 9. State transitions

The implementation should use explicit state transitions rather than relying on implicit combinations of booleans.

### 9.1 Initialization state

States:

- `UNINITIALIZED`
- `COLLECTING`
- `READY`
- `RESET`

Transitions:

| Current state | Event | Next state |
|---|---|---|
| UNINITIALIZED | First valid observation | COLLECTING |
| COLLECTING | Additional consecutive valid observation | COLLECTING |
| COLLECTING | Five valid observations reached | READY |
| COLLECTING | Invalid observation or tracking loss | RESET |
| READY | Tracking loss | RESET |
| READY | Observation gap exceeds 200 ms | RESET |
| RESET | Fresh valid observation | COLLECTING |

After reset, motion estimation must not reuse old observations.

### 9.2 Approach state

States:

- `UNKNOWN`
- `CANDIDATE`
- `APPROACHING`
- `NOT_APPROACHING`

Transitions:

| Current state | Event | Next state |
|---|---|---|
| UNKNOWN | Sufficient motion evidence available | CANDIDATE or NOT_APPROACHING |
| CANDIDATE | Direction and distance conditions persist | CANDIDATE |
| CANDIDATE | Evidence persists for 500 ms | APPROACHING |
| CANDIDATE | Valid evidence contradicts approach | NOT_APPROACHING |
| APPROACHING | Reliable approach evidence continues | APPROACHING |
| APPROACHING | Reliable evidence indicates approach has stopped or reversed | NOT_APPROACHING or CANDIDATE, according to evidence |
| Any state | Tracking loss or temporal discontinuity | UNKNOWN |

The implementation must avoid immediate state oscillation caused by small numerical fluctuations. The 500 ms confirmation applies to entering `APPROACHING`. Any additional exit hysteresis must be separately configurable and justified by tests; it is not a frozen decision.

### 9.3 Movement state

States:

- `UNKNOWN`
- `MOVING`
- `STATIONARY`

Transitions:

| Current state | Event | Next state |
|---|---|---|
| UNKNOWN | Valid motion estimates available | MOVING or STATIONARY |
| MOVING | Both stationary conditions hold | STATIONARY |
| STATIONARY | Reliable stationary condition fails | MOVING |
| Any state | Insufficient history or tracking loss | UNKNOWN |

The implementation may use temporal smoothing to reduce boundary oscillation, but must not silently alter the frozen thresholds.

### 9.4 Stability state

States:

- `NOT_STARTED`
- `ACCUMULATING`
- `COMPLETE`
- `RESET`
- `UNKNOWN`

Transitions:

| Current state | Event | Next state |
|---|---|---|
| NOT_STARTED | Eligibility predicate becomes true | ACCUMULATING |
| ACCUMULATING | Eligibility remains true | ACCUMULATING |
| ACCUMULATING | Brief valid threshold excursion within budget | ACCUMULATING |
| ACCUMULATING | Elapsed duration reaches 2 seconds | COMPLETE |
| ACCUMULATING | Cumulative violation exceeds 200 ms | RESET |
| ACCUMULATING | Tracking loss | RESET |
| ACCUMULATING | Required observation becomes UNKNOWN/invalid | RESET |
| ACCUMULATING | Temporal discontinuity | RESET |
| COMPLETE | New stability attempt is required after conditions cease | NOT_STARTED or RESET |
| RESET | Fresh eligible evidence begins | ACCUMULATING |

A completed timer is an event associated with one stability attempt. It must not be repeatedly emitted as a new completion on every subsequent frame.

---

## 10. Testing strategy

M5 requires more than unit tests.

Unlike M4, whose acceptance criteria were explicitly based on unit tests, M5 must be evaluated using controlled and labelled temporal sequences because its purpose is to infer behaviour over time.

Testing is divided into four levels.

### 10.1 Unit tests

#### Observation-history tests

- Correct bounding-box centre calculation.
- Timestamp ordering.
- Duplicate timestamp rejection.
- Non-monotonic timestamp rejection.
- Correct one-second window eviction.
- Bounded history size.
- Five-observation initialization.
- Initialization reset on invalid observation.
- Predicted positions excluded from history.
- Track identity change resets history.
- Observation gap greater than 200 ms resets temporal continuity.

#### Motion-estimation tests

- Stationary synthetic trajectory.
- Constant horizontal motion.
- Constant vertical motion.
- Diagonal motion.
- Slow movement below the stationary threshold.
- Movement above the stationary threshold.
- Irregular timestamp intervals.
- Noisy bounding-box centres.
- Zero-duration interval.
- Insufficient history.
- Displacement exactly at the threshold.
- Velocity exactly at the threshold.

#### Approach tests

- Person moves directly towards zone centroid.
- Person moves directly away.
- Person moves perpendicular to the centroid direction.
- Person moves diagonally towards the centroid.
- Person remains stationary.
- Distance decreases but direction condition fails.
- Direction points towards zone but distance does not decrease.
- Approach conditions persist for less than 500 ms.
- Approach conditions persist for at least 500 ms.
- Tracking loss during approach confirmation.
- Temporal discontinuity during approach confirmation.
- Zero velocity and undefined direction.
- Person reaches the centroid or crosses the reference point.

#### Stationary tests

- Velocity below threshold and displacement below threshold.
- Velocity above threshold and displacement below threshold.
- Velocity below threshold and displacement above threshold.
- Both thresholds exceeded.
- Oscillatory movement with small net displacement.
- Bounding-box jitter around the threshold.
- Incomplete one-second displacement window.
- Unknown motion estimate.

#### Stability timer tests

- Eligibility begins and remains true for two seconds.
- Stability completes at the correct timestamp.
- One brief violation within the tolerance budget.
- Several brief violations whose cumulative duration remains within budget.
- Cumulative violation exceeds 200 ms.
- Tracking loss resets immediately.
- Required M4 observation becomes UNKNOWN.
- Required M4 observation becomes INVALID.
- Observation gap exceeds 200 ms.
- Timer is unaffected by nominal FPS variation.
- Timer is not driven by frame count.
- Timer cannot complete from predicted positions.
- A completed attempt is not emitted repeatedly.

### 10.2 Integration tests

Test the interfaces between M3, M4, and M5.

Verify:

1. M3 track observations are correctly consumed.
2. Only actual observed positions enter M5 history.
3. M4 observations are associated with the same person and frame.
4. M4 observation status and value semantics are preserved.
5. M3 tracking loss triggers immediate M5 reset.
6. M5 output remains independent of M6 decision logic.
7. Existing M1–M4 tests continue to pass.
8. Existing configuration and logging conventions are preserved.

Integration tests must not modify the frozen M3 algorithm or expand M4 scope.

### 10.3 Controlled and labelled sequence evaluation

Construct a reproducible set of sequences with known behaviour labels.

At minimum, include:

| Sequence | Ground-truth behaviour | Purpose |
|---|---|---|
| S1 | Person stationary in view | Stationary false-positive/negative evaluation |
| S2 | Person approaches interaction zone | Approach detection |
| S3 | Person moves away from interaction zone | Direction discrimination |
| S4 | Person walks laterally | Reject non-approach movement |
| S5 | Person passes near the zone | Reject incidental proximity |
| S6 | Person approaches and stops | Combined approach and stationary transition |
| S7 | Person moves slowly | Evaluate stationary threshold |
| S8 | Person oscillates in place | Evaluate displacement and velocity combination |
| S9 | Bounding-box jitter while person is still | Evaluate noise tolerance |
| S10 | Temporary detection loss | Verify immediate reset |
| S11 | Long observation gap | Verify temporal discontinuity |
| S12 | Person enters and exits the interaction zone | Verify zone-related stability reset |
| S13 | Person remains stable for two seconds | Evaluate stability timer |
| S14 | Person has brief threshold fluctuations | Evaluate cumulative tolerance |
| S15 | Person has repeated movement violations | Verify timer reset |

Each sequence should have:
- Ground-truth start and end timestamps.
- Behaviour labels.
- Relevant person-track identity.
- Expected transition points.
- Camera resolution and frame rate.
- Recorded configuration.
- Any known annotation uncertainty.

The sequence evaluation should be repeatable using saved observations or recorded input sequences.

Synthetic sequences are useful for algorithmic correctness, but real recorded sequences are necessary to evaluate performance under detector and tracker noise.

### 10.4 Metrics

#### Approach detection

- Precision.
- Recall.
- F1-score.
- False-positive rate.
- False-negative rate.
- Detection latency relative to ground-truth approach onset.

#### Stationary detection

- Precision.
- Recall.
- F1-score.
- False stationary rate.
- False moving rate.
- Transition detection latency.

#### Stability timer

- Timer completion accuracy.
- False completion rate.
- Missed completion rate.
- Completion timestamp error.
- Number of false resets.
- Number of resets correctly triggered by tracking loss or uncertainty.

#### Runtime

Measure:
- Total M5 processing latency.
- Motion-estimation latency.
- Approach-estimation latency.
- Stationary-estimation latency.
- Stability-timer latency.
- Average FPS where meaningful.
- Memory usage where measurable.

Report latency percentiles:

\[
p50,\quad p95,\quad p99
\]

Do not invent performance targets before measuring the baseline implementation.

### 10.5 Reproducibility

Each experiment must record:

- Configuration file.
- Code version or Git revision.
- Input sequence identifier.
- Camera resolution.
- Nominal and observed FPS.
- Threshold values.
- Timestamp source.
- Evaluation labels.
- Evaluation metrics.

Use deterministic synthetic sequences and fixed timestamps for timer and state-transition unit tests.

---

## 11. Demo requirements

The M5 demo must demonstrate temporal behaviour over a sequence, not merely display a single-frame estimate.

### 11.1 Expected demonstration

A typical successful sequence:

1. A person appears in the camera view.
2. M3 establishes and confirms the track.
3. M5 collects five consecutive valid observations.
4. M5 begins estimating image-space motion.
5. The person moves towards the interaction-zone centroid.
6. The approach indicator becomes `APPROACHING` after 500 ms of valid approach evidence.
7. The person slows down and stops.
8. The stationary indicator becomes `STATIONARY` when both stationary conditions hold.
9. The person remains inside the configured interaction region.
10. The stability timer begins.
11. The timer accumulates elapsed time while tolerating brief valid threshold fluctuations.
12. After two seconds, the timer emits `COMPLETE`.

The demo must also show failure and reset cases.

### 11.2 Visual overlays

Display:

- Current observed person bounding box.
- Bounding-box centre.
- Interaction-zone polygon.
- Interaction-zone centroid.
- Line from person centre to zone centroid.
- Recent centre trajectory.
- Estimated velocity vector.
- Estimated velocity magnitude in px/s.
- Displacement over one second.
- Distance to zone centroid in pixels.
- Direction cosine.
- Distance trend or closing-speed estimate.
- Approach state and reliability.
- Movement/stationary state and reliability.
- Initialization count.
- Observation gap.
- Stability timer elapsed time.
- Cumulative violation time.
- Stability state.
- Reset reason when applicable.

### 11.3 Diagnostic output

The demo should make it possible to understand why an estimate was produced.

For example:

```text
Track ID: 1
Initialization: READY (5/5)

Position: (640.0, 350.0)
Velocity: 8.4 px/s
Displacement: 6.2 px / 1 s

Distance to zone centroid: 125.0 px
Direction cosine: 0.91
Distance trend: -18.2 px/s

Approach: APPROACHING [VALID]
Movement: STATIONARY [VALID]

Stability:
  State: ACCUMULATING
  Elapsed: 1.25 s / 2.00 s
  Violation: 0.05 s / 0.20 s
```

Values above are illustrative, not benchmark results.

### 11.4 Demo acceptance

The demo is accepted when:

- It shows all required temporal indicators.
- It displays the interaction-zone centroid and trajectory.
- It correctly demonstrates approach, non-approach, moving, and stationary cases.
- It shows the five-observation initialization.
- It demonstrates the two-second stability timer.
- It demonstrates cumulative tolerance.
- It demonstrates immediate reset on tracking loss.
- It displays `UNKNOWN` and reliability status when temporal evidence is unavailable.
- It does not use predicted Kalman positions as observed motion.
- It does not claim final first-contact readiness.

---

## 12. Performance and resource considerations

M5 should be lightweight relative to the YOLO and MediaPipe components.

The expected processing operations are:

- A small bounded observation buffer.
- Bounding-box centre calculation.
- Distance calculations.
- Linear regression over a short history.
- State-machine updates.
- Timestamp-based timers.

The memory footprint should remain small because M5 stores numeric observations rather than image frames.

For a nominal 30 FPS camera and a one-second history, approximately 30 position observations are expected per active track.

No image crops, video frames, face landmarks, or duplicate model outputs should be retained by M5 unless a separate requirement justifies them.

The initial implementation should prioritize correctness and measurement over premature optimization.

---

## 13. Failure handling

| Failure condition | Required behaviour |
|---|---|
| M3 track lost | Immediately clear history and temporal state |
| Track ID changes | Reset all state associated with the previous track |
| Predicted position only | Reject as a motion observation |
| Invalid bounding box | Reject observation; invalidate affected estimate |
| M4 required observation UNKNOWN | Do not infer a positive or negative behaviour from it |
| M4 required observation INVALID | Invalidate affected temporal estimate |
| Timestamp not increasing | Reject observation |
| Observation gap exceeds 200 ms | Clear temporal history and reinitialize |
| Insufficient observations | Return UNKNOWN |
| Insufficient one-second coverage | Do not produce a valid stationary estimate |
| Zero velocity vector | Direction is UNKNOWN |
| Degenerate interaction polygon | Fail configuration validation |
| Numerical regression failure | Return UNKNOWN/INVALID as appropriate; do not fabricate an estimate |

A failure in one temporal estimator should not unnecessarily invalidate independent estimators if their required inputs remain valid.

For example, an undefined direction vector may make approach estimation unavailable while a valid displacement estimate remains available.

---

## 14. Implementation structure

The implementation must first inspect the repository and follow its existing module boundaries and naming conventions.

The following logical responsibilities are proposed. They are not mandatory filenames.

| Responsibility | Description |
|---|---|
| Observation history | Bounded timestamped centre history |
| Motion estimator | Velocity, displacement, and trajectory |
| Approach estimator | Direction and distance-trend predicate |
| Stationary estimator | Combined velocity/displacement predicate |
| Stability timer | Timestamp-based continuous stability |
| Temporal estimator/orchestrator | Coordinates estimators and output contracts |

Do not create unnecessary abstractions or duplicate data structures.

Keep state local to the relevant component and ensure that reset behaviour is explicit and testable.

---

## 15. M5 exit criteria

M5 is complete only when all of the following are satisfied.

### Functional

- [ ] Timestamped bounding-box centre history implemented.
- [ ] Five-consecutive-observation initialization implemented.
- [ ] One-second bounded history implemented.
- [ ] 200 ms maximum observation gap enforced.
- [ ] Image-space velocity estimation implemented.
- [ ] One-second displacement estimation implemented.
- [ ] Movement direction estimation implemented.
- [ ] Zone-centroid distance estimation implemented.
- [ ] Distance-trend estimation implemented.
- [ ] Approach predicate implemented.
- [ ] 500 ms approach confirmation implemented.
- [ ] Stationary predicate implemented with both thresholds.
- [ ] Two-second stability timer implemented.
- [ ] 200 ms cumulative tolerance implemented.
- [ ] Immediate reset on tracking loss implemented.
- [ ] Uncertainty and reliability status implemented.
- [ ] No predicted Kalman positions used as observed measurements.

### Testing

- [ ] Unit tests pass.
- [ ] Integration tests pass.
- [ ] Deterministic timer tests pass.
- [ ] State-transition tests pass.
- [ ] Controlled/labelled sequence evaluation completed.
- [ ] False-positive and false-negative measurements reported.
- [ ] Latency and memory measurements recorded where feasible.
- [ ] Existing M1–M4 tests remain unaffected.

### Demo

- [ ] Approach is visibly demonstrated.
- [ ] Stationary behaviour is visibly demonstrated.
- [ ] Non-approach movement is demonstrated.
- [ ] Stability timing is demonstrated.
- [ ] Noise tolerance is demonstrated.
- [ ] Tracking-loss reset is demonstrated.
- [ ] UNKNOWN and reliability states are visible.

### Documentation

- [ ] Configuration documented.
- [ ] Mathematical definitions documented.
- [ ] State transitions documented.
- [ ] Evaluation results documented.
- [ ] Known limitations documented.
- [ ] Implementation report updated after implementation and actual testing.

---

## 16. Known limitations and future considerations

1. **Perspective dependence:** Image-space movement is not equivalent to physical movement.
2. **Centroid approximation:** The polygon centroid may not correspond to the most natural physical approach point for every interaction-zone shape.
3. **Bounding-box jitter:** Changes in detector output can create apparent movement even when the person is stationary.
4. **Partial visibility:** The bounding-box centre may shift when a person is partially occluded.
5. **Camera dependence:** Thresholds may need recalibration when moving from laptop camera to robot camera.
6. **No learned behaviour model:** The initial implementation uses deterministic geometry and temporal rules.
7. **Single-person scope:** M5 estimates behaviour for the M3 active track; it does not manage multiple candidates.
8. **Threshold calibration:** The selected values are engineering starting points and require evaluation on labelled sequences.
9. **No physical distance:** A pixel-based interaction-zone predicate must not be interpreted as a physical proximity measurement.
10. **No final readiness decision:** M5 produces temporal evidence for M6 rather than declaring the robot ready to initiate contact.

Future work may consider improved camera calibration, perspective-aware movement estimation, alternative zone-distance geometry, and learned temporal models. Such changes require separate design approval and must not be silently introduced into the initial M5 implementation.

---

## 17. Final design summary

M5 uses timestamped bounding-box centres from the M3 observed track and relevant M4 frame-level observations.

Its initial algorithm is based on:

- One-second image-space observation history.
- Five consecutive valid observations for initialization.
- Timestamp-aware linear regression for velocity and distance trends.
- A 200 ms maximum observation gap.
- Approach determined by movement direction towards the zone centroid and decreasing distance to that centroid.
- 500 ms approach confirmation.
- Stationary detection using both a 20 px/s velocity threshold and a 20 px displacement threshold over one second.
- Immediate temporal reset on tracking loss.
- Two-second continuous stability timing with 200 ms cumulative tolerance.
- Explicit `VALID`, `UNCERTAIN`, and `INVALID` reliability statuses.
- `UNKNOWN` values whenever evidence is insufficient or unavailable.

The implementation is deterministic, image-space based, configurable, and testable.

**M5 provides temporal behavioural evidence. M6 remains responsible for combining that evidence with M4 frame-level observations to determine first-contact readiness.**

**Implementation status:** Not started. This document defines the approved design target; implementation begins only after the M4 correction pass has been reviewed and approved.