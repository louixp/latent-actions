# Latent Action

Latent Action allows users to control high dimensional (3-DoF or 7-DoF) Franka Emika Panda arm with low dimensional (2-DoF) interfaces. The arm is simulated with [Panda Gym](https://github.com/qgallouedec/panda-gym) built on top of PyBullet physics engine. The low dimensional interface is a GamePad controller. The mapping between the two spaces is learned with a conditional variational autoencoder (cVAE).

## Pre-requisites

1. A GamePad controller (tested on [Logitech F310](https://www.amazon.com/gp/product/B003VAHYQY/r)).
2. Install the software dependencies:
	```bash
	pip install -r requirements.txt
	```
3. (Optional) A Franka Emika Panda arm.

## Usage

To start the simulation, run:
```bash
python3 enjoy.py --model_class cVAE --checkpoint_path [CHECKPOINT_PATH]
```

The simulation spawn three processes:
1. A PyBullet simulation that renders the robot and the environment.
2. A vector field visualization that shows the conditional latent space (The red dot indicates the current stick position).
3. The embeded conditional 2D manifold of the latent space within the full 3D space.

https://github.com/louixp/latent-actions/assets/52590858/cc0df693-7353-482c-8c15-c353165d3e11

To control the arm, use the right stick of the GamePad controller. To exit the simulation, press back button on the controller.

## Real robot
It works with a real robot too!

https://github.com/louixp/latent-actions/assets/52590858/fc7d480d-c259-4740-ba51-1be8dc00ab09

