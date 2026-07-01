# Handoff Intent Data Collection README

This README is a step-by-step script for collecting the handoff intent dataset using the granular collection server.

The goal is to collect labeled RGB-D images and temporally paired Quest upper-body joints for a single-frame model that predicts whether a person is offering an object to the robot.


---

## 0. Usage Prequisites

To use the data collection procedure, you'll first need to clone this repo and install all dependencies. You will also need a realsense D435i depth camera, and a Meta Quest 3 AR headset capable of collecting joint data and sending it in the format expected here. Scripts for this are provided in the QuestScripts directory for this repo; note these were created for version 72 of the Quest SDK.

---

---

## 1. Dataset Design

Each sample belongs to one of four conditions, with one sample for each hand.

| Key | Binary label | Condition | Hand |
|---|---:|---|---|
| `1` | `handoff` | Facing camera and offering | Right |
| `2` | `handoff` | Facing camera and offering | Left |
| `3` | `handoff` | Facing away/angled but offering to robot | Right |
| `4` | `handoff` | Facing away/angled but offering to robot | Left |
| `5` | `not_handoff` | Holding object near torso, not offering | Right |
| `6` | `not_handoff` | Holding object near torso, not offering | Left |
| `7` | `not_handoff` | Displaying object to camera, not offering | Right |
| `8` | `not_handoff` | Displaying object to camera, not offering | Left |

For each participant:

```text
9 locations (in an even grid )× number_of_objects × 8 hotkey samples
```

---


## 2. Physical Setup

Use a D435i depth camera mounted on a Unitree Go2 camera or a the camera mounted at the same approximate height, pitch, and field of view as the robot camera.

Do not move the camera during a participant’s collection session.

The participant should hold each pose steady for 1–2 seconds before the collector presses the save key.

---

## 3. 9-Position Grid

Tape a 3×3 grid on the floor relative to the camera.

The first row is near, the second row is mid, and the third row is far.

| Location ID | Row | Column | Meaning |
|---|---|---|---|
| `N-L` | Near | Left | Near row, camera-left |
| `N-C` | Near | Center | Near row, centered |
| `N-R` | Near | Right | Near row, camera-right |
| `M-L` | Mid | Left | Mid row, camera-left |
| `M-C` | Mid | Center | Mid row, centered |
| `M-R` | Mid | Right | Mid row, camera-right |
| `F-L` | Far | Left | Far row, camera-left |
| `F-C` | Far | Center | Far row, centered |
| `F-R` | Far | Right | Far row, camera-right |

---

## 4. Object Set

Use a fixed set of objects across participants.

---

## 5. Condition Definitions

Use these definitions consistently.

### Key 1 / Key 2: Facing camera and offering

The participant faces the camera/robot. The object is clearly extended toward the robot as if offering it for the robot to take.

This is a positive handoff sample.

### Key 3 / Key 4: Facing away or angled but offering

The participant’s torso/head is angled away from the camera, but the object is still clearly offered toward the robot/camera.

This is a positive handoff sample.

Important: if the object is offered away from the robot, this is **not** a positive. The body can be angled away, but the offer target must still be the robot.

### Key 5 / Key 6: Holding near torso, not offering

The participant holds the object near their torso, chest, waist, or side. The object is in hand, but the arm is not presenting it to the robot.

This is a negative sample.

### Key 7 / Key 8: Displaying to camera, not offering

The participant shows or displays the object to the camera so it is visible, but does not present it as something the robot should take.

This is a negative sample.

This should look like “showing what the object is,” not “offering it for transfer.”

---

## 6. Startup Procedure

1. Confirm the RealSense camera is connected.
2. Confirm the Quest joint sender is running or ready to run. For our example, navigate to the "Unknown sources" page in the Quest applications. Then, open the app called "HandoffIntentDataColl"
3. Confirm the server machine and Quest are on the same network. 
4. Start the collection server. If the Quest and server are on the same network, you will see messages coming into the server.  

Example:

```bash
python -m uvicorn HandIntentDataCollServer_granular:app --host 0.0.0.0 --port 8000 --workers 1
```

When prompted, enter the participant ID.

Example:

```text
Enter participant ID before startup, e.g. P01:
P01
```

Alternatively, set the participant ID using an environment variable:

```bash
PARTICIPANT_ID=P01 python -m uvicorn HandIntentDataCollServer_granular:app --host 0.0.0.0 --port 8000 --workers 1
```

Do not begin collection until both frame and joints are being received.

---

## 7. Per-Participant Collection Script

Use this exact procedure for every participant.

### For each participant

1. Start the server with the correct participant ID.
2. Explain the four conditions to the participant.
3. Demonstrate each pose once.
4. Ask the participant to hold each pose still for 1–2 seconds before each save.
5. Work through the grid in the order below.

For each location:

1. Move the participant to the taped location.
2. For each object:
   1. Give the participant the object.
   2. Collect the 8 samples below.
   3. Mark the object-location block complete on the checklist.

### For each location × object block

Say this aloud or follow it silently as a checklist.

| Step | Collector key | Instruction to participant |
|---:|---|---|
| 1 | `1` | Face the camera and offer the object with your right hand. |
| 2 | `2` | Face the camera and offer the object with your left hand. |
| 3 | `3` | Angle your body away, but still offer the object to the camera/robot with your right hand. |
| 4 | `4` | Angle your body away, but still offer the object to the camera/robot with your left hand. |
| 5 | `5` | Hold the object near your torso with your right hand. Do not offer it. |
| 6 | `6` | Hold the object near your torso with your left hand. Do not offer it. |
| 7 | `7` | Display or show the object to the camera with your right hand, but do not offer it. |
| 8 | `8` | Display or show the object to the camera with your left hand, but do not offer it. |

After each keypress, wait for the server to print a successful save message.

If the wrong key is pressed or the pose was wrong, write down the sample ID from the terminal so it can be removed later.

---


## 8. What the Server Saves

Each saved sample folder contains:

```text
rgb.png
depth_z16.png
joints_raw.json
meta.json
```

The CSV index is:

```text
handoff_dataset/samples_granular.csv
```

The sample folder structure is:

```text
handoff_dataset/
  samples/
    <participant_id>/
      handoff/
        facing_camera_offering/
          right/
          left/
        facing_away_offering/
          right/
          left/
      not_handoff/
        holding_torso_not_offering/
          right/
          left/
        displaying_camera_not_offering/
          right/
          left/
```

The Quest joints are saved in:

```text
joints_raw.json
```

The Quest joint packet should contain named joints such as:

```text
head
left_shoulder
right_shoulder
left_upper_arm
right_upper_arm
left_forearm
right_forearm
left_hand
right_hand
```

The saved Quest joints are temporally paired with the RGB-D frame. They are not automatically transformed into the RealSense camera coordinate frame.

---

## 9. Quality Control After Each Participant

After finishing one participant, check the dataset before moving on.

### Expected sample count

```text
expected_samples = 9 × number_of_objects × 8
```

Examples:

| Objects | Expected samples for one participant |
|---:|---:|
| 4 | 288 |
| 5 | 360 |
| 6 | 432 |

### Check the CSV row count

Open:

```text
handoff_dataset/samples_granular.csv
```

Confirm that the participant has the expected number of rows.

### Check folder contents

Randomly open several `rgb.png` files from different conditions.

Confirm:

- image is not black or corrupted,
- participant is visible,
- object is visible when expected,
- condition matches folder/key,
- hand matches folder/key,
- camera position did not move.

### Check Quest joints

Open a few `joints_raw.json` files.

Confirm:

- named joints are present,
- values are not empty,
- validity flags are mostly true.

---

## 10. Common Mistakes to Avoid

| Mistake | Why it is a problem |
|---|---|
| Pressing during motion | Label may not match saved frame |
| Letting participant drift between grid marks | Location consistency is lost |
| Mixing up left and right hand | Label noise |
| Letting “displaying” become “offering” | Negative class becomes ambiguous |
| Letting “facing away offering” become “offering away from robot” | Positive label becomes wrong |
| Forgetting to track object/location externally | Hard to audit dataset later |
| Starting before Quest joints are received | Samples will fail or be missing joint data |
| Randomly splitting images for evaluation | Adjacent controlled samples can leak person/location/object bias |

---

## 11. Recommended Train/Validation/Test Split

Split by participant, not by image.

Example with 10 participants:

| Split | Participants |
|---|---|
| Train | P01–P07 |
| Validation | P08 |
| Test | P09–P10 |

Do not put images from the same participant in both train and test.

---

