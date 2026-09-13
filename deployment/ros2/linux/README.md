# ROS 2 Linux deployment

This directory contains the Linux-side ROS 2 components used by the final thesis deployment pipeline.

Final architecture:

Linux ROS 2 command source
    -> Ethernet / Fast DDS
    -> Windows ROS 2 runtime
    -> dual-controller router
    -> STAND v2-G / LOCOMOTION v4-N
    -> NVIDIA Isaac Lab simulation

## Runtime

- `runtime/g1_v01_linux_phase10b_canonical_straight.py`
  - canonical straight-command run
- `runtime/g1_v02_linux_phase10b_canonical_curve_pos.py`
  - engineering positive-yaw curve run

The canonical command-source frame identifier is:

`g1_cmd_source_linux_phase10b_canonical`

## Fast DDS configuration

`config/g1_fastdds_ethernet_linux.xml` is the final Linux Fast DDS configuration used for Ethernet communication with the Windows runtime.

## Validation

`validation/g1_phase10b_linux_ethernet_canonical_soak_probe.py` is the final canonical Linux-side soak validation probe.

## ROS 2 interface

The final interface uses:

- `/g1/deployment/cmd_vel`
- `/g1/deployment/cmd_echo`
- `/g1/deployment/status`
- `/g1/deployment/base_twist_body`
- `/g1/deployment/base_pose`
- `/g1/deployment/stop`

The Linux runtime uses ROS 2 Jazzy with `rclpy`, `geometry_msgs` and `std_msgs`.

Legacy Gazebo standing-balance code and intermediate ROS/Isaac migration experiments are intentionally excluded.
