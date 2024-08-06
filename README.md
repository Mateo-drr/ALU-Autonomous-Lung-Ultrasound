# ALU - Autonomous Lung Ultrasound

## Overview
ALU (Autonomous Lung Ultrasound) is a project designed to autonomously perform lung ultrasound scanning using a robotic system.

## Requirements
- ROS2 Humble
- Torch
- scikit-optimize

## Installation

### Prerequisites
Ensure you have ROS2 Humble installed. You can follow the installation instructions [here](https://docs.ros.org/en/humble/Installation.html).

## Usage

### Running the Simulation Controller
To run the simulation controller using CoppeliaSim:
```bash
ros2 launch ur_coppeliasim ur_coppelia_controllers.launch.py
```

### Running the Lung Ultrasound Node
To start the lung ultrasound node:
```bash
ros2 run lung_us talker
```

### Running the Real Robot Controller
To launch the controller for the real robot:
```bash
ros2 launch ur_launch ur_compliance_controller.launch.py
```

### Re-run the Robot Program
To resend the robot program if it closes:
```bash
ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger {}
```

### Build Packages
To build the packages after placing them in your ros workspace:
```bash
colcon build --symlink-install
```

## Configuration

### Kinematics Parameters
Edit the `ur_compliance_controller.launch.py` file to set the kinematics parameters:
```python
kinematics_params = "/home/mateo-drr/Documents/ALU---Autonomous-Lung-Ultrasound/ros2Packages/my_robot_calibration.yaml"
```



