# Humanoid RL Locomotion

Reinforcement Learning control and ROS 2 deployment pipeline for the **Unitree G1 humanoid robot**, developed as a Bachelor's thesis project.

The project focuses on learning-based humanoid control in **NVIDIA Isaac Sim / Isaac Lab** and culminates in a dual-controller architecture composed of two specialized PPO policies:

* **v2-G — STAND**: specialized for stable upright standing and low-speed postural control.
* **v4-N — LOCOMOTION**: specialized for locomotion and commanded-velocity tracking.

The two policies are integrated through a stateful router with hysteresis and an explicit braking phase. A distributed **ROS 2 / Fast DDS** pipeline was additionally developed and validated between a Linux command source and the Windows machine running the Isaac Lab simulation and policy runtime.

> **Scope**
>
> The final deployment validation is hardware-free. It validates the ROS 2 transport, policy runtime and end-to-end integration against the simulated Unitree G1 in Isaac Lab. It does not represent deployment on the physical robot.

---

## Project overview

The final project includes:

* PPO-based humanoid control using RSL-RL;
* specialized standing and locomotion policies;
* quantitative controller evaluation;
* left/right symmetry augmentation for locomotion;
* low-speed controller specialization analysis;
* controlled LOCOMOTION-to-STAND handoff;
* braking-based pre-switch state conditioning;
* hysteretic dual-controller routing;
* independent holdout validation;
* ROS 2 communication over Ethernet using Fast DDS;
* frozen final checkpoints and integrity hashes;
* preserved deployment-validation artifacts.

The earlier ROS 2 / Gazebo standing-balance coursework project from which the research initially evolved is maintained separately and is intentionally not part of this repository.

---

## Final control architecture

```text
                         Velocity command
                                |
                                v
                     +---------------------+
                     | Dual-policy router  |
                     +---------------------+
                        |       |       |
                        |       |       |
                     STAND    BRAKE    LOCO
                        |       |       |
                        v       v       v
                      v2-G    v4-N    v4-N
                         \       |       /
                          \      |      /
                           +-----------+
                           | Unitree G1|
                           | Isaac Lab |
                           +-----------+
```

The frozen routing configuration is:

| Parameter                  |    Value |
| -------------------------- | -------: |
| STAND policy               |     v2-G |
| LOCOMOTION policy          |     v4-N |
| Enter locomotion threshold | 0.20 m/s |
| Exit locomotion threshold  | 0.10 m/s |
| Brake duration             |   0.36 s |
| Brake control steps        |       18 |
| Control frequency          |    50 Hz |
| Action blending            | Disabled |

During a stopping transition, **v4-N remains active with a zero velocity command for 18 control steps** before control is transferred to v2-G.

The braking stage therefore performs pre-switch state conditioning rather than interpolation between the actions produced by the two policies.

---

## Reinforcement Learning interface

The final controllers use **Proximal Policy Optimization (PPO)** through RSL-RL.

The common policy interface is:

| Property              |   Value |
| --------------------- | ------: |
| Observation dimension |     123 |
| Action dimension      |      37 |
| Simulation timestep   | 0.005 s |
| Simulation frequency  |  200 Hz |
| Control decimation    |       4 |
| Control timestep      | 0.020 s |
| Control frequency     |   50 Hz |

The 123-dimensional observation vector contains:

* base linear velocity;
* base angular velocity;
* projected gravity;
* commanded velocity;
* relative joint positions;
* joint velocities;
* previous actions.

The 37-dimensional policy output is converted into joint-position targets around the default articulation pose.

---

## Final controllers

### v2-G — STAND

v2-G is the final standing specialist.

It combines direct yaw-command training over an extended range with stronger orientation regularization and is used for:

* zero-command standing;
* upright stabilization;
* low-speed postural control;
* final control after a completed braking transition.

Frozen checkpoint:

```text
artifacts/final_checkpoints/v2g_model_600.pt
```

SHA-256:

```text
58468EED03F9EF35A9BC898063A3B14AC36C93A32F65DB42B404BDE93CD694D3
```

---

### v4-N — LOCOMOTION

v4-N is the final locomotion specialist.

It was trained from scratch and extends the direct-yaw locomotion baseline with native RSL-RL left/right symmetry data augmentation.

It is used for:

* forward locomotion;
* commanded-velocity tracking;
* the braking phase preceding a switch to v2-G.

Frozen checkpoint:

```text
artifacts/final_checkpoints/v4n_model_1499.pt
```

SHA-256:

```text
367D318AB4940A3AAE8F0FA568384212762258067DFC4CF9F624FA472BEB9342
```

The files under `artifacts/final_checkpoints/` are frozen scientific artifacts and should not be modified in place.

---

## Dual-controller validation

The two final policies exhibit complementary behavior in the low-speed region:

* **v2-G** provides stronger postural stability;
* **v4-N** provides more accurate velocity tracking as commanded speed increases.

This motivated the use of two specialized controllers instead of forcing a single policy to optimize both operating regimes.

The final routing parameters were calibrated on development runs and frozen before evaluation on an independent holdout set.

The holdout validation confirmed:

* the expected router transition sequence;
* correct cancellation of temporary braking phases;
* correct 18-step completed braking phases;
* absence of premature episode termination;
* stable final standing behavior.

The thresholds and braking duration are therefore **development-calibrated and frozen**, not claimed to be globally optimal.

---

## ROS 2 deployment

The final project includes a distributed ROS 2 deployment pipeline:

```text
Linux
ROS 2 command source
        |
        | Ethernet / Fast DDS
        v
Windows
ROS 2 runtime
        |
        v
Dual-controller router
        |
        +---- v2-G STAND
        |
        +---- v4-N LOCOMOTION
        |
        v
NVIDIA Isaac Lab
Unitree G1 simulation
```

The Linux side uses **ROS 2 Jazzy**, `rclpy` and Fast DDS.

The final ROS 2 interface is based on:

```text
/g1/deployment/cmd_vel
/g1/deployment/cmd_echo
/g1/deployment/status
/g1/deployment/base_twist_body
/g1/deployment/base_pose
/g1/deployment/stop
```

The canonical Linux command-source frame identifier is:

```text
g1_cmd_source_linux_phase10b_canonical
```

The Linux deployment sources are located under:

```text
deployment/ros2/linux/
```

The final Fast DDS Ethernet profile is:

```text
deployment/ros2/linux/config/g1_fastdds_ethernet_linux.xml
```

The canonical final validation probe is:

```text
deployment/ros2/linux/validation/
g1_phase10b_linux_ethernet_canonical_soak_probe.py
```

---

## Repository structure

```text
humanoid-rl-locomotion/
|
|-- source/
|   `-- humanoid_rl_lab/
|       `-- Isaac Lab environments, tasks, rewards and PPO configurations
|
|-- scripts/
|   |-- rsl_rl/
|   |   `-- training, evaluation and policy execution tooling
|   |
|   `-- thesis/
|       `-- scientific validation and final thesis tooling
|
|-- deployment/
|   `-- ros2/
|       `-- linux/
|           |-- runtime/
|           |-- validation/
|           |-- config/
|           `-- README.md
|
|-- artifacts/
|   |-- final_checkpoints/
|   |   |-- v2g_model_600.pt
|   |   |-- v4n_model_1499.pt
|   |   |-- README.md
|   |   `-- SHA256SUMS.txt
|   |
|   `-- deployment_validation/
|       `-- linux/
|
|-- docs/
|   `-- thesis/
|       `-- experimental_freezes/
|
|-- .vscode/
|-- .gitignore
|-- pyproject.toml
`-- README.md
```

Generated simulation logs, intermediate checkpoints, rendered videos, video frames, caches, training outputs and build products are intentionally excluded from version control.

---

## Installation

### 1. Install Isaac Sim and Isaac Lab

Install a compatible **Isaac Sim / Isaac Lab** environment following the official Isaac Lab installation documentation.

Isaac Lab evolves rapidly, so the exact Isaac Sim, Python and dependency versions should be selected according to the Isaac Lab release being used rather than inferred from this repository.

The project should remain outside the core Isaac Lab source tree.

---

### 2. Clone the repository

```bash
git clone git@github.com:ArcadeGeek/humanoid-rl-locomotion.git
cd humanoid-rl-locomotion
```

---

### 3. Install the project package

Using the Python interpreter associated with the Isaac Lab environment:

```bash
python -m pip install -e source/humanoid_rl_lab
```

If Isaac Lab is installed through its standalone helper scripts, use the corresponding Isaac Lab Python launcher instead of the system Python interpreter.

Editable installation is required so that the Gymnasium task registrations under `source/humanoid_rl_lab/` are available to the training and evaluation scripts.

---

## Inspecting registered environments

The repository retains the Isaac Lab task-discovery utility:

```bash
python scripts/list_envs.py
```

Task registrations are defined inside the `humanoid_rl_lab` package under:

```text
source/humanoid_rl_lab/humanoid_rl_lab/tasks/
```

---

## Training and evaluation tooling

The RSL-RL entry points are located under:

```text
scripts/rsl_rl/
```

Relevant scripts include:

```text
train.py
train_rsl_rl.py
play.py
play_rsl_rl.py
evaluate_g1.py
evaluate_g1_line.py
```

Available command-line options can be inspected directly, for example:

```bash
python scripts/rsl_rl/train.py --help
```

and:

```bash
python scripts/rsl_rl/play.py --help
```

The repository contains multiple historical task configurations used during experimental development. The final scientific controller configurations are preserved in the source tree and identified by the thesis tooling and frozen artifacts.

---

## Dummy-agent smoke tests

The original Isaac Lab project utilities for simple environment smoke testing are retained:

```bash
python scripts/zero_agent.py --task=<TASK_NAME>
```

and:

```bash
python scripts/random_agent.py --task=<TASK_NAME>
```

These utilities are useful for checking environment initialization independently of a trained policy.

---

## Linux ROS 2 deployment setup

From the repository root on the Linux deployment machine:

```bash
source /opt/ros/jazzy/setup.bash

export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=0
export FASTRTPS_DEFAULT_PROFILES_FILE="$PWD/deployment/ros2/linux/config/g1_fastdds_ethernet_linux.xml"
```

The canonical Linux runtime scripts are:

```text
deployment/ros2/linux/runtime/
```

Before executing a deployment script, its available command-line options can be inspected with:

```bash
python3 deployment/ros2/linux/runtime/g1_v01_linux_phase10b_canonical_straight.py --help
```

The final deployment architecture assumes the corresponding ROS 2 runtime is active on the Windows / Isaac Lab side.

---

## Deployment validation artifacts

Frozen Linux-side outputs associated with the final deployment are stored under:

```text
artifacts/deployment_validation/linux/
```

The repository intentionally preserves only final relevant evidence:

* the **Phase 10b canonical Ethernet soak**;
* telemetry associated with the final V01 scientific straight-motion capture;
* telemetry associated with the final V02 engineering curve capture.

Rendered videos and image frames are intentionally excluded.

The integrity of the preserved deployment files can be checked using:

```bash
sha256sum -c artifacts/deployment_validation/linux/SHA256SUMS.txt
```

---

## Final checkpoint integrity

The two frozen policy artifacts can be verified against:

```text
artifacts/final_checkpoints/SHA256SUMS.txt
```

On Linux:

```bash
cd artifacts/final_checkpoints
sha256sum -c SHA256SUMS.txt
```

A successful check confirms that the policy files match the frozen checkpoints used by the thesis experiments.

---

## Development utilities

The repository includes a pre-commit configuration for code-quality tooling.

Install:

```bash
pip install pre-commit
```

Run against the complete repository with:

```bash
pre-commit run --all-files
```

The top-level `.vscode/` directory also contains development configuration retained intentionally for the project.

Machine-specific VS Code state and generated IDE files are excluded through `.gitignore`.

---

## Reproducibility

The repository preserves the main components required to reconstruct the final experimental configuration:

* Isaac Lab task definitions;
* final reward configurations;
* PPO configurations;
* training and evaluation scripts;
* scientific validation tooling;
* final v2-G and v4-N checkpoints;
* frozen checkpoint SHA-256 hashes;
* dual-controller validation tooling;
* Linux ROS 2 runtime scripts;
* Fast DDS Ethernet configuration;
* final deployment-validation artifacts.

Intermediate checkpoints, generated logs, cache files, training runs, raw videos and frame sequences are intentionally excluded.

---

## Experimental provenance

The Git history retains the main controller-development lineages.

Relevant frozen tags include:

```text
g1-heading-hold-multiseed-v1
g1-yaw-direct-wide-yaw-flat-v2g
```

The `main` branch contains the consolidated final thesis implementation.

Historical experimental branches are retained in the remote repository where they provide useful provenance for the controller-development process.

---

## Limitations

The project has several explicit limitations:

* validation is performed in simulation rather than on the physical Unitree G1;
* v4-N is trained primarily for flat-terrain locomotion;
* rough-terrain experiments are qualitative out-of-distribution evaluations rather than terrain-trained benchmarks;
* the integrated router validation uses controlled startup conditions with zero initial root velocities;
* robustness to the original randomized root-velocity reset distribution is treated separately from router validation;
* routing thresholds and braking duration were empirically calibrated during development and are not claimed to be globally optimal;
* human-like locomotion was not directly optimized or formally measured.

These limitations are intentionally kept separate from the validated claims of the final architecture.

---

## Thesis context

This repository contains the experimental software developed for a Bachelor's thesis on reinforcement-learning-based control of a humanoid robot.

The final work combines:

1. specialized learned controllers for standing and locomotion;
2. quantitative simulation-based validation;
3. a controlled dual-policy handoff architecture;
4. independent holdout evaluation;
5. a distributed ROS 2 / Fast DDS deployment pipeline;
6. frozen policy and deployment artifacts for reproducibility.

The project therefore covers the full path from learned-policy development and validation to distributed ROS 2 runtime integration.

