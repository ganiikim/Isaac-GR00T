# RB5 + DG5F Real-Time GR00T Inference Guide

> **Internal research-lab documentation**  
> System: NVIDIA Isaac-GR00T N1.7 (fine-tuned) + RB5 (6 DoF) + DG5F (20 DoF) + Intel RealSense D405/D435  
> Environment: Ubuntu 22.04, ROS 2 Humble, separate GR00T `uv` environment and system Python client

## 1. Overview

This guide describes how to run the fine-tuned GR00T model with live camera images and robot joint states, and optionally execute its predicted actions on the RB5 arm and DG5F hand.

**Data flow**

```text
RealSense D405 (front) ───┐
RealSense D435 (second) ──┤
RB5 joint state (6 DoF) ──┼──> GR00T client ── ZMQ:5555 ──> GR00T server
DG5F joint state (20 DoF) ┘                                   │
                                                               ▼
                                              16-step predicted action chunk
                                                               │
                                      ┌────────────────────────┴──────────────────────┐
                                      ▼                                               ▼
                                RB5 joint targets                              DG5F joint targets
                                200 Hz servo stream                            100 Hz trajectory stream
                                interpolation + EMA + slew limit               joint limits + slew limit
```

The dataset and inference policy use **26 joint dimensions**: RB5 joints 0–5 and DG5F joints 6–25. The dataset frame rate is **30 Hz**, and the model returns **16 action steps** per inference. The client periodically replaces its current predicted plan with a newly inferred one.

**Important:** A 200 Hz command stream is not a 200 Hz model-inference rate. GR00T inference and command streaming operate at different rates.

## 2. Repository and files

Recommended files in the existing `Isaac-GR00T` repository:

```text
Isaac-GR00T/
├── RB5_TRAINING_GUIDE.md
├── RB5_INFERENCE_GUIDE.md         # This document
├── rb5_groot_inference.py         # Observation + prediction only (if retained)
├── rb5_groot_rb5_dg5f_continuous_v9_rb5_smooth.py
├── gr00t/eval/run_gr00t_server.py
└── checkpoints/rb5_grape_sort_10k/  # Local only; do not commit
```

The v9 filename is the current **working experiment version**, not a claim of formally validated robot safety. Preserve the working file before refactoring or renaming it.

## 3. Hardware and software

| Component | Configuration |
|---|---|
| Robot arm | RB5, 6 DoF |
| Dexterous hand | DG5F, right hand, 20 DoF |
| Front camera | Intel RealSense D405, serial `218622277644` |
| Second camera | Intel RealSense D435, serial `817512070767` |
| Camera streams | 640 × 480, 30 FPS |
| ROS | ROS 2 Humble |
| ROS workspace | `~/rb5_teleop` |
| GR00T repository | `~/Isaac-GR00T` |
| Model | `ganikim/rb5_grape_sort_10k` |
| Model server | ZMQ, port `5555` |
| ROS / hardware client Python | `/usr/bin/python3` |
| GR00T model Python | `uv run python` |

The live client uses the two cameras, `/rb5/joint_states`, and `/dg5f_right/dg5f_right_controller/state`. Both state streams are required by the current client even when robot actuation is disabled.

## 4. One-time environment checks

### 4.1 ROS environment

In **each new terminal that runs a ROS node or the ROS-based inference client**:

```bash
source /opt/ros/humble/setup.bash
source ~/rb5_teleop/install/setup.bash
```

Verify that packages are visible:

```bash
ros2 pkg prefix rb5_teleop
ros2 pkg prefix dg5f_driver
```

If either package is missing, check that the workspace was built and sourced. Do not blindly rebuild a working workspace during a robot-control session.

### 4.2 Model checkpoint

Check that the local fine-tuned checkpoint exists:

```bash
ls -lh ~/Isaac-GR00T/checkpoints/rb5_grape_sort_10k
```

If it is missing, download it from the approved Hugging Face repository:

```bash
cd ~/Isaac-GR00T
uv run hf download ganikim/rb5_grape_sort_10k \
  --local-dir ./checkpoints/rb5_grape_sort_10k
```

Do **not** commit model weights, datasets, captured camera videos, tokens, or local machine credentials to GitHub, including a Private repository.

### 4.3 Python environments

- Run the **GR00T server** with `uv run python` inside `~/Isaac-GR00T`.
- Run the **ROS + RealSense + rbpodo client** with `/usr/bin/python3`.
- Avoid using Conda `base` Python for the ROS client; it may not contain `rclpy`, `pyrealsense2`, or `rbpodo`.

## 5. Startup procedure — four terminals

> **Before hardware execution:** clear the robot workspace, verify the physical emergency-stop is accessible, confirm the correct robot/hand are connected, and make sure no other node is commanding the same hardware.

### Terminal 1 — DG5F driver

```bash
source /opt/ros/humble/setup.bash
source ~/rb5_teleop/install/setup.bash

ros2 launch dg5f_driver dg5f_right_driver.launch.py \
  delto_ip:=169.254.186.6 \
  delto_port:=502
```

The driver publishes the DG5F state and receives DG5F trajectory commands.

### Terminal 2 — RB5 state publisher

```bash
source /opt/ros/humble/setup.bash
source ~/rb5_teleop/install/setup.bash

ros2 run rb5_teleop rb5_state_node
```

This node publishes the RB5 joint/TCP state. It is **not** the GR00T model server.

### Terminal 3 — GR00T model server

```bash
cd ~/Isaac-GR00T

uv run python gr00t/eval/run_gr00t_server.py \
  --model-path ./checkpoints/rb5_grape_sort_10k \
  --embodiment-tag new_embodiment \
  --device cuda \
  --host 0.0.0.0 \
  --port 5555
```

Wait until the model has loaded and the server is listening on port `5555` before starting the client.

### Terminal 4 — Real-time inference and robot control (v9)

```bash
source /opt/ros/humble/setup.bash
source ~/rb5_teleop/install/setup.bash

cd ~/Isaac-GR00T

/usr/bin/python3 rb5_groot_rb5_dg5f_continuous_v9_rb5_smooth.py \
  --execute-rb5 \
  --execute-dg5f \
  --show-cameras \
  --max-cycles 0
```

**Warning: this command sends real motion commands to both the RB5 and DG5F.** `--max-cycles 0` means continuous operation until stopped or blocked. `--show-cameras` opens the client camera views.

### Inference without robot motion

To test the inference pipeline before enabling hardware control, **omit both execute flags**:

```bash
source /opt/ros/humble/setup.bash
source ~/rb5_teleop/install/setup.bash

cd ~/Isaac-GR00T

/usr/bin/python3 rb5_groot_rb5_dg5f_continuous_v9_rb5_smooth.py \
  --show-cameras \
  --max-cycles 10
```

Confirm that the v9 script prints predictions and does not enable either controller when the execute flags are absent. Both ROS state publishers and the model server still need to be running.

## 6. ROS topic verification

Use a separate ROS-sourced terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/rb5_teleop/install/setup.bash

ros2 topic list | grep -E 'rb5|dg5f'
ros2 topic echo /rb5/joint_states --once
ros2 topic echo /dg5f_right/dg5f_right_controller/state --once
```

Relevant topics:

| Topic | Expected type / role |
|---|---|
| `/rb5/joint_states` | `sensor_msgs/msg/JointState`, 6 joint positions in radians |
| `/rb5/tcp_state` | RB5 TCP state |
| `/dg5f_right/dg5f_right_controller/state` | `control_msgs/msg/JointTrajectoryControllerState`, 20 actual positions |
| `/dg5f_right/dg5f_right_controller/joint_trajectory` | `trajectory_msgs/msg/JointTrajectory`, hand commands |

The DG5F expected joint names are `rj_dg_1_1` through `rj_dg_5_4` in the controller's established order. Verify the name-to-index mapping before commanding the hand.

## 7. Model inputs and outputs

### Inputs

```text
video.front                       RGB image, [1, 1, 480, 640, 3]
video.second                      RGB image, [1, 1, 480, 640, 3]
state.rb5_arm                     RB5 joint angles, [1, 1, 6]
state.right_hand                  DG5F joint angles, [1, 1, 20]
language.human.task_description   Natural-language task description
```

Current task description:

```text
Sort one bunch of grapes by color: place green grapes on the right and red grapes on the left.
```

### Outputs

```text
RB5 predicted action             [1, 16, 6]
DG5F predicted action            [1, 16, 20]
```

The client receives **postprocessed absolute joint targets in radians**. Although the GR00T training processor uses a relative representation for RB5 internally, its output is converted back to absolute joint targets before being returned by the policy client. **Do not add the current joint state to the returned RB5 action again.**

The training demonstrations were driven using Cartesian servo motion, but the stored learning targets are **joint-space actions**. Therefore the predicted joint targets are executed with joint-space servo control (`move_servo_j`), not by passing them to `move_servo_l`.

## 8. Control implementation (v9)

### RB5

```text
GR00T 16-step action chunk
    -> trajectory interpolation
    -> 200 Hz servo target stream
    -> EMA smoothing (RB5_SMOOTH_ALPHA = 0.15 by default)
    -> per-joint command slew clamp (MAX_RB5_STEP_RAD = 0.03 rad/command)
    -> rbpodo move_servo_j
```

Current `move_servo_j` parameters:

```python
SERVO_T1 = 0.01
SERVO_T2 = 0.10
SERVO_GAIN = 0.2
SERVO_ALPHA = 0.2
```

The **software EMA alpha** is separate from `SERVO_ALPHA` in the RB5 SDK call. Raising `RB5_SMOOTH_ALPHA` (e.g., from `0.15` toward `0.30`) makes the software target filter more responsive and less smoothing-intensive. These values are experimental, not safety-certified settings. Change one parameter at a time, start with short low-risk motion, and check actual robot behavior and tracking error.

**Important:** A `0.03 rad/command` clamp at 200 Hz is a *per-command* bound, not a validated joint-velocity or acceleration limit. It does not, by itself, guarantee smoothness or safe physical motion.

### DG5F

- Command stream: **100 Hz** `JointTrajectory`.
- Per-joint target clamps use the configured DG5F joint ranges.
- Software slew limiter: **0.02 rad/command** in the current implementation.
- Watch for large `cmd-actual` tracking errors. A limited change in the *command* does not guarantee that the hand physically tracks it.

### Workspace and safety

The latest intended RB5 nominal Cartesian workspace values used during development were:

```text
X: 450–745 mm
Y: -70–420 mm
Z: 130–550 mm
Software tolerance: 5 mm
```

**Verify the values actually configured in the current script and on the robot controller before motion.** Earlier development versions used different limits. Workspace checks should remain hard stops; the RB5 command-jump guard was replaced by a slew clamp, but this is **not** a reason to disable workspace, controller-error, or other safety checks.

A software `SERVO STREAM BLOCKED` message is **not** the physical emergency stop. A stop in the RB5 stream does **not necessarily** stop the independently controlled DG5F. Keep the physical E-stop available and do not rely on Ctrl+C to cancel an already accepted motion command.

## 9. Shutdown

1. Bring the robot to a safe, stable state using the established hardware procedure.
2. Stop the inference/control client (Terminal 4) and confirm hardware motion has ceased.
3. Stop the GR00T server (Terminal 3).
4. Stop the RB5 state publisher and DG5F driver (Terminals 2 and 1) only when safe.

**Do not assume Ctrl+C immediately cancels a servo command already accepted by the controller.** Verify the robot and hand have actually stopped.

## 10. Troubleshooting

| Symptom | Check |
|---|---|
| `ros2: command not found` or package not found | Source `/opt/ros/humble/setup.bash` and `~/rb5_teleop/install/setup.bash` in that terminal |
| No RB5 state | Check RB5 power/network, `rb5_state_node`, and `/rb5/joint_states` |
| No DG5F state | Check driver launch, hand connection, and `/dg5f_right/dg5f_right_controller/state` |
| Client waits for state | Current client requires **both** RB5 (6D) and DG5F (20D) states, even in inference-only mode |
| GR00T server not responding | Confirm model finished loading and port `5555` is listening |
| `ModuleNotFoundError` for ROS, RealSense, or rbpodo | Use `/usr/bin/python3` for the client, not Conda base or `uv` |
| Camera cannot start | Verify camera serials, USB connections, and that no other process owns the camera |
| `cv2` not defined | Ensure the selected client file includes `import cv2` (fixed in v7+) |
| FastDDS deserialization warnings | Check ROS environment and topic messages; warnings were observed even when some valid messages arrived; do not assume communication is fully healthy |
| RB5 feels jerky or moves in tiny increments | Inspect `RB5_SMOOTH_ALPHA`, command tracking, plan replacement, and controller behavior; do not simply raise safety bounds |
| DG5F command and actual state differ greatly | Stop/limit testing and investigate controller response, joint ordering, target ranges, and tracking behavior |
| Competing hand commands | Do **not** run `manus_retarget_node` alongside GR00T hand control |

## 11. Nodes **not** required for GR00T inference

The following were used during teleoperation or data collection and should **not** be started for the current GR00T inference workflow:

```text
manus_data_publisher
manus_retarget_node
vive_tracker_node
dataset_recorder
```

In particular, a MANUS retargeting node publishing to the DG5F controller can conflict with GR00T hand commands.

## 12. Git / research-lab maintenance

- Keep the repository **Private** and grant access only to authorized lab members.
- Commit source code, documentation, and non-sensitive configuration templates.
- Exclude checkpoints (`*.safetensors`), large datasets/videos, secrets, and local environment directories.
- Keep a known-working version on `main`; test changes on a separate branch.
- Document tested hardware/software revisions, camera serial mappings, model revision, control parameters, and observed behavior in commits or test logs.
- Review upstream Isaac-GR00T licensing and attribution requirements before redistributing modified upstream code.

Suggested `.gitignore` additions:

```gitignore
checkpoints/
demo_data/
*.safetensors
*.mp4
*.webm
*.bag
.env
.venv/
__pycache__/
*.pyc
```

## Quick reference

**Run order:** DG5F driver → RB5 state node → GR00T server → inference client.  
**ROS environment:** `source /opt/ros/humble/setup.bash && source ~/rb5_teleop/install/setup.bash`  
**Server Python:** `uv run python`  
**Client Python:** `/usr/bin/python3`  
**Live control flags:** `--execute-rb5 --execute-dg5f`  
**Continuous operation:** `--max-cycles 0`  
**Camera view:** `--show-cameras`
