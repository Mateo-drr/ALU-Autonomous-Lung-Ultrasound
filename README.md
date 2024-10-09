# ALU - Autonomous Lung Ultrasound

## Overview
ALU (Autonomous Lung Ultrasound) is a project designed to autonomously perform lung ultrasound scanning using a robotic system.

## Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/yourproject.git
```
Navigate to the directory:
```bash
cd /ALU-Autonomous-Lung-Ultrasound
```
Install the required packages
```bash
pip install -r requirements.txt
```

### Prerequisites
Ensure you have ROS2 Humble installed. You can follow the installation instructions [here](https://docs.ros.org/en/humble/Installation.html).

## Usage

### Setting the Controllers

To set up the controllers, follow the instructions in [this repository](https://github.com/fzi-forschungszentrum-informatik/cartesian_controllers).

**Note:** By default, the controllers do not publish the current robot position information. To enable this feature, you need to modify the following flags in the `cartesian_controller_base/src/cartesian_controller_base.cpp` file:

```cpp
//Line 104
auto_declare<bool>("solver.publish_state_feedback", false);
m_initialized = true;

//Line 129
auto_declare<int>("solver.iterations", 1);
auto_declare<bool>("solver.publish_state_feedback", false);
```
Both to:
```cpp
auto_declare<bool>("solver.publish_state_feedback", true);
```

### Build Packages
To build the packages navigate to `ros2Packages` and do:
```bash
colcon build --symlink-install
```
Don't forget to source both the controller workspace and this one.

### Launching the controller

**Note** For the control logic to work correctly if using another end-effector you need to update the URDF in the `ur_description` ros2 package.

To run the simulation controller using CoppeliaSim:
```bash
ros2 launch ur_coppeliasim ur_coppelia_controllers.launch.py
```

To launch the controller for the real robot:
```bash
ros2 launch ur-launch ur_compliance_controller.launch.py
```
The controller being used is the `cartesian_compliance_controller` which allows the robot to not excert excesive force onto the phantoms.

### Running the Lung Ultrasound Nodes
Two nodes were created, one for the acquisition of the data necessary for the development of the system, and the other one for the implementation of said system.

To start data acquisition node:
```bash
ros2 run lung_us talker
```
The parameters for the movement can be changed when the program is running, but the base and center target position has to be updated in the `aks.py` file

To start the autonomous acquisition nodes:
1. Move the robot to a desired starting position, assuring contact with the skin
2. Start the matlab ros2 node that constantly acquires the images and sends them. To do this run in matlab the file `ros2autoAq.m`
3. Start the acquisition node:
```bash
ros2 run lung_us listener
```
The system will perform N number of bayesian optimization cycles and early stop if the last 5 acquisitions have an error change of less than 0.02.


##Additional things

### Re-run the Robot Program
To resend the robot program if it closes:
```bash
ros2 service call /io_and_status_controller/resend_robot_program std_srvs/srv/Trigger {}
```






