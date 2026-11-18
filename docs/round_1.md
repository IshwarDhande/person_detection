# Round 1 --- Hardware and Environment: Questions, Options, Selected Answers, and Rationale

**Project:** Vision-Based First Contact for a Tabletop Social Robot\
**Document:** `round_1.md`\
**Status:** Initial hardware and environment requirements recorded;
hardware details pending confirmation\
**Purpose:** Preserve the Round 1 questions, the options presented, the
user's selected answers, and the engineering implications of those
decisions.

> **Interpretation note:** The "Reasoning and implications" sections
> explain the engineering consequences of the selected configuration.
> They are not necessarily verbatim statements supplied by the user.
> Unknown or unconfirmed details are explicitly marked as pending rather
> than assumed.

------------------------------------------------------------------------

## 1. Summary of selected decisions

  -----------------------------------------------------------------------
  ID                      Topic                   Selected answer
  ----------------------- ----------------------- -----------------------
  Q1                      Robot configuration     Stationary tabletop
                                                  social robot

  Q2                      Camera mounting         Camera mounted on the
                                                  robot's head

  Q3                      Camera model            Unknown / not provided

  Q4a                     Resolution              720p (1280 × 720)

  Q4b                     Frame rate              30 FPS

  Q4c                     Camera modality         RGB only

  Q5                      Operating environment   Office reception

  Q6                      Lighting and scene      Normal artificial
                          conditions              lighting, daylight, low
                                                  lighting, crowded
                                                  environment, people
                                                  carrying bags or
                                                  objects

  Q7                      Maximum detection       Unknown / not yet
                          distance                decided

  Q8                      Additional details      None supplied
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 2. Detailed question and answer record

## Q1. What is the robot configuration?

**Options offered:**

-   A. Stationary social robot placed on a reception table.
-   B. Mobile robot near the reception desk.
-   C. Humanoid robot with a head-mounted camera.
-   D. Stationary robot with an external camera.
-   E. Other / not finalized.

**Selected answer:** **A --- Stationary tabletop social robot.**

**Reasoning and implications:**

-   The robot remains at a fixed reception-table location rather than
    navigating through the environment.
-   The first-contact system can therefore focus on observing the
    reception area from a fixed robot location.
-   The robot's position and the camera-to-reception geometry are
    expected to remain relatively stable during operation.
-   The system does not need mobile-robot navigation or moving-camera
    localization as part of this project's initial scope.

**Implementation implications:**

-   Define the robot's physical reference point and the intended visitor
    interaction area.
-   Record the robot's location relative to the reception desk and
    floor.
-   Verify whether the robot or its head can move independently; this
    has not yet been specified.
-   If the camera moves with the head, camera pose may change even
    though the robot base remains stationary.

------------------------------------------------------------------------

## Q2. Where is the camera mounted?

**Options offered:**

-   A. On the robot's head, facing forward.
-   B. On the robot's body.
-   C. External camera on the reception desk.
-   D. Ceiling or elevated camera.
-   E. Multiple cameras.
-   F. Not finalized.

**Selected answer:** **A --- On the robot's head.**

The selected option described a forward-facing head-mounted camera.

**Reasoning and implications:**

-   The camera observes the scene from approximately the robot's
    viewpoint.
-   The camera's field of view determines which visitors can be detected
    and monitored.
-   The camera's mounting height, pitch, yaw, and field of view affect
    person localization and distance estimation.
-   If the robot's head moves, the camera viewpoint may also change.
    This must be checked before relying on fixed image-space interaction
    zones.

**Implementation implications:**

-   Measure the camera's mounting height and orientation.
-   Determine whether the head/camera is fixed or moves during
    operation.
-   Record the camera's field of view.
-   Calibrate the camera relative to the reception area before defining
    the interaction polygon.

------------------------------------------------------------------------

## Q3. What camera hardware are you using?

**Options offered:** Free-text camera model input; examples included
Intel RealSense D435i and Logitech webcam. Leaving the field blank was
allowed if the model was unknown.

**Selected answer:** **Unknown / not provided.**

**Reasoning and implications:**

-   The exact camera model has not yet been identified.
-   It is therefore not possible to confirm its lens, field of view,
    intrinsic parameters, exposure behaviour, or other hardware
    characteristics.
-   The selected RGB-only modality is recorded separately in Q4; the
    exact device still needs confirmation.

**Implementation implications:**

Before camera calibration and deployment, identify:

-   Manufacturer and model.
-   Lens and field of view.
-   Supported resolution and frame rates.
-   Camera intrinsics or calibration procedure.
-   Exposure and low-light performance.
-   Whether the camera provides any additional sensor streams that are
    not currently intended for use.

------------------------------------------------------------------------

## Q4. What are the camera specifications?

This question was divided into three specification fields.

### Q4a. Resolution

**Options offered:**

-   A. 720p.
-   B. 1080p.
-   C. 4K.
-   D. Unknown.

**Selected answer:** **A --- 720p (1280 × 720).**

**Reasoning and implications:**

-   720p establishes the initial image resolution for the perception
    pipeline.
-   It provides a concrete input size for implementation and performance
    testing.
-   The resolution may be sufficient for person detection at the
    intended reception distances, but this must be evaluated with the
    actual camera and scene.
-   Small or distant people, partial occlusion, and low-light conditions
    may reduce detection quality.

**Implementation implications:**

-   Use 1280 × 720 as the initial recorded input resolution.
-   Evaluate person-detection performance at the actual operating
    distances.
-   Avoid assuming that a model's performance at a different resolution
    transfers unchanged.

### Q4b. Frame rate

**Options offered:**

-   A. 15 FPS.
-   B. 30 FPS.
-   C. 60 FPS.
-   D. Unknown.

**Selected answer:** **B --- 30 FPS.**

**Reasoning and implications:**

-   30 FPS provides a nominal frame interval of approximately 33.3
    milliseconds.
-   This gives the system a temporal stream for estimating movement,
    approach behaviour, and stationary state.
-   Actual end-to-end latency will also depend on inference time,
    tracking, filtering, and event processing.
-   The camera's configured frame rate and the effective processed frame
    rate may differ.

**Implementation implications:**

-   Treat 30 FPS as the nominal camera input rate.
-   Measure the actual frame acquisition and processing rates.
-   Timestamp frames so that movement and two-second stability checks
    use elapsed time rather than assuming every frame is processed.

### Q4c. Camera modality / depth information

**Options offered:**

-   A. RGB only.
-   B. RGB + Depth (RGB-D).
-   C. Stereo camera.
-   D. Unknown.

**Selected answer:** **A --- RGB only.**

**Reasoning and implications:**

-   The initial perception system must operate using colour images
    without direct depth measurements.
-   Physical distance cannot be read directly from a depth stream.
-   Distance estimation, if required, must use calibrated scene
    geometry, monocular estimation, or another explicitly validated
    method.
-   Bounding-box size alone is not a reliable metric-distance
    measurement.

**Implementation implications:**

-   Begin with RGB person detection and tracking.
-   Investigate fixed-camera calibration and floor-plane geometry for
    approximate ground-point localization.
-   Validate any estimated distance against measured ground truth.
-   Keep distance uncertainty explicit in the first-contact decision
    logic.

------------------------------------------------------------------------

## Q5. Where will the robot operate?

**Options offered:**

-   A. Office reception.
-   B. Hospital reception.
-   C. Research laboratory.
-   D. Shopping mall / public space.
-   E. Controlled experimental environment.
-   F. Other.

**Selected answer:** **A --- Office reception.**

**Reasoning and implications:**

-   The system is intended for a reception setting where visitors
    approach a stationary robot.
-   The reception desk and visitor waiting/standing area provide a
    physical context for defining the interaction region.
-   The scene may include people who pass through, wait nearby, talk to
    others, or approach the robot.
-   The system must distinguish a visitor's approach from unrelated
    movement in the environment.

**Implementation implications:**

-   Define the reception layout and intended visitor standing area.
-   Identify common walking paths and likely passerby regions.
-   Capture representative videos of normal reception activity.
-   Calibrate the interaction area for the actual camera placement
    rather than relying on a generic image region.

------------------------------------------------------------------------

## Q6. What lighting and visibility conditions are expected?

**Options offered:** Multiple selections were allowed.

-   A. Normal artificial lighting.
-   B. Daylight.
-   C. Low lighting.
-   D. Backlighting / sunlight behind the person.
-   E. Partial occlusion.
-   F. Crowded environment.
-   G. People carrying bags or objects.

**Selected answers:**

-   Normal artificial lighting.
-   Daylight.
-   Low lighting.
-   Crowded environment.
-   People carrying bags or objects.

**Not selected:**

-   Backlighting / sunlight behind the person.
-   Partial occlusion.

**Reasoning and implications:**

-   The system is expected to operate across more than one ordinary
    lighting condition.
-   Low lighting may reduce image quality and person-detection
    confidence.
-   Crowded scenes introduce overlapping people and potential tracking
    ambiguity.
-   Bags and carried objects may obscure parts of a person's body or
    affect bounding-box and pose estimates.
-   Although partial occlusion was not selected as a separate expected
    condition in this question, Q1 subsequently selected partial person
    visibility as acceptable. Therefore, partial visibility should still
    be considered in the detection requirements.

**Implementation implications:**

-   Evaluate detection and tracking under artificial lighting, daylight,
    and low-light conditions.
-   Include crowded scenes and people carrying bags or objects in the
    test data.
-   Record lighting and crowd conditions for each evaluation sequence.
-   Do not assume that an unselected backlighting condition is covered
    by the current requirements; it may be added to the test matrix
    later if it occurs in the deployment environment.

------------------------------------------------------------------------

## Q7. What is the expected maximum detection distance?

**Options offered:**

-   A. 0--2 metres.
-   B. 2--4 metres.
-   C. 4--6 metres.
-   D. More than 6 metres.
-   E. Unknown / not decided.

**Selected answer:** **E --- Unknown / not decided.**

**Reasoning and implications:**

-   The maximum distance at which the robot must detect people has not
    yet been specified.
-   This is different from the 1.5 m near-distance target selected later
    in Round 2.
-   Maximum detection distance determines how early the system can
    observe an approaching person and establish approach history.
-   It also affects camera field-of-view requirements, person-detection
    performance, and the time available for tracking.

**Implementation implications:**

-   Measure the physical reception area and available camera view.
-   Identify the farthest point at which a visitor should enter the
    system's observation range.
-   Test person detection at increasing distances using the actual
    camera.
-   Finalize the required detection range before freezing the camera and
    detector configuration.

------------------------------------------------------------------------

## Q8. Additional hardware or environment details

**Prompt:** Provide any additional information about the robot, camera
position, or reception setup.

**Selected answer:** No additional details supplied.

**Status:** No further constraints recorded.

------------------------------------------------------------------------

# 3. Consolidated hardware and environment profile

The current Round 1 configuration is:

  -----------------------------------------------------------------------
  Category                            Recorded requirement
  ----------------------------------- -----------------------------------
  Robot                               Stationary tabletop social robot

  Camera mounting                     Robot head, forward-facing as
                                      specified by the selected option

  Camera model                        Unknown

  Resolution                          720p (1280 × 720)

  Nominal frame rate                  30 FPS

  Modality                            RGB only

  Environment                         Office reception

  Lighting                            Normal artificial lighting,
                                      daylight, low lighting

  Scene complexity                    Crowded environment; people
                                      carrying bags or objects

  Maximum detection distance          Unknown

  Camera mounting geometry            Not measured

  Camera field of view                Unknown

  Camera intrinsics                   Unknown

  Head/camera movement                Not specified
  -----------------------------------------------------------------------

## 4. Engineering consequences of the selected setup

### 4.1 Fixed robot, potentially variable camera pose

The tabletop robot is stationary, but the camera is mounted on its head.
If the head can rotate or tilt, the camera is not necessarily fixed in
the world frame.

Before using a static image-space interaction polygon, verify whether
the head remains in a fixed pose during first-contact observation.

If the camera moves, the system may need head-pose information, image
stabilization, or a way to transform observations into a stable
reference frame.

### 4.2 RGB-only distance estimation

The camera provides no direct depth stream in the selected
configuration.

Potential initial approaches include:

1.  Calibrated camera geometry and floor-plane homography, if the scene
    and camera geometry support it.
2.  Ground-contact-point localization and geometric distance estimation.
3.  Monocular depth estimation as an alternative to investigate and
    validate.

No distance-estimation method has yet been selected or validated.

### 4.3 Operating conditions

The test environment should reflect the selected conditions:

-   Artificial lighting.
-   Daylight.
-   Low lighting.
-   Crowded reception scenes.
-   People carrying bags or objects.

Partial visibility should also be tested because it was explicitly
accepted in the subsequent Round 2 person-visibility decision.

### 4.4 Detection range versus interaction distance

Two different quantities must remain separate:

-   **Maximum detection distance:** The farthest distance at which a
    person should be detected and monitored. This is unknown from Round
    1.
-   **Near/interaction distance:** The initial 1.5 m target selected
    during Round 2.

The 1.5 m interaction target does not establish the maximum detection
range.

------------------------------------------------------------------------

# 5. Open hardware and deployment questions

The following items remain unresolved and should be addressed before the
perception architecture is frozen.

  -----------------------------------------------------------------------
  ID                      Open question           Why it matters
  ----------------------- ----------------------- -----------------------
  H1                      What is the exact       Determines hardware
                          camera manufacturer and capabilities and
                          model?                  calibration procedure

  H2                      What is the camera's    Determines visible
                          horizontal and vertical reception coverage
                          field of view?          

  H3                      What is the camera      Needed for geometric
                          mounting height?        distance estimation

  H4                      What are the camera's   Needed to relate image
                          pitch and yaw angles?   coordinates to the
                                                  reception area

  H5                      Does the robot's head   Determines whether a
                          or camera move?         static image-space zone
                                                  is valid

  H6                      What is the camera's    Needed for geometric
                          intrinsic calibration?  estimation

  H7                      What is the maximum     Determines observation
                          required detection      range and detector
                          distance?               requirements

  H8                      What is the physical    Needed to define the
                          layout of the reception interaction polygon
                          area?                   

  H9                      Is backlighting         Not selected in Round
                          expected in deployment? 1; may still occur in
                                                  practice

  H10                     What is the actual      Needed for temporal
                          effective camera and    estimation and latency
                          processing FPS?         measurement
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 6. Round 1 status

  Area                              Status
  --------------------------------- ----------------------------
  Robot configuration               Selected
  Camera mounting location          Selected
  Camera model                      Unknown
  Resolution                        Selected: 720p
  Frame rate                        Selected: 30 FPS
  Camera modality                   Selected: RGB
  Operating environment             Selected: office reception
  Lighting and scene conditions     Selected
  Maximum detection distance        Unknown
  Camera geometry and calibration   Pending
  Physical reception layout         Pending

**Round 1 status:** Initial hardware and environment requirements
recorded. Camera identification, mounting geometry, calibration, and
maximum detection range remain open.

## 7. Relationship to subsequent requirements rounds

-   **Round 1 --- Hardware and environment:** Establishes the physical
    and sensing constraints.
-   **Round 2 --- Detection, proximity, and first-contact readiness:**
    Defines how the vision system decides that a person is ready for
    first contact.
-   **Round 3 --- Multi-person tracking and target selection:** Will
    define how the system maintains and selects a person when multiple
    people are visible.

**Scope boundary:** This document records Round 1 only. It does not
introduce or finalize Round 2 behavioural decisions or Round 3 tracking
policies.
