# Instructions template {#instructions-duckling_controller status=ready}

For these demonstrations you will need : 


- Duckiebot in configuration `DB19`


- Camera calibration completed



## Video of expected results {#demo-duckling_controller-expected}

The expected behaviors are shown in the following videos.

<p>&nbsp;</p>

[Expected behavior](https://youtu.be/uOWxnWE0eHQ) on the simulator. 

<p>&nbsp;</p>

[Expected behavior](https://youtu.be/NsuGWjWAxIg) on the duckiebot. 

<p>&nbsp;</p>



## For the simulator {#demo-duckling_controller-simulator}


### Description
This [script](https://github.com/jerome-labonte-udem/duckietown-visual-servo/tree/daffy/visual_servo) is meant as a first exploration of using visual servoing 
to control the duckiebot. The visual servo package that can be installed on the duckiebot
contains some improvements on this script but we decided to provide these files 
as a sandbox to experiment new ideas.

### Architecture

Here are the main components of this project.

<p>&nbsp;</p>

* estimation.py contains everything that is related to estimating the relative 
pose of the target point to the robot.
* control.py contains everything that is related to generating the
duckiebot commands from the pose estimation
* visual_servo.py is the main script that puts the two components together 
* config.py contains the configuration values

### Usage
Before running for the first time, create a virtual environment and from the root folder if this repository
run:
```bash
pip install -e .
```

To run the visual servo script, move inside the visual_servo directory and run:
```bash
python ./visual_servo.py
```

You can set the logging level to debug by adding the --debug flag. This will allow you to see additional information generated for debugging purposes

### Expected behaviour
Once the gym window is opened, your duckie but will move toward
the stationnary bot and try to park 15cm behind it and look in the same direction. 
You can change this value in the config.py file. If your bot doesn't not move, in means
 it has not detected the pattern. You can either press backspace to reinitialize the bot in 
a new location or move your duckiebot until it detects the pattern.

### Known problems
The estimation of the pose is not very precise at the moment. This is due to 
bad camera parameters. We did not succeed in finding the right parameters to use. 
You can see some of our tries in the estimation.py file, where
each camera_mode was a guess about a way to get the right values.


## For the Duckiebot {#demo-duckling_controller-duckiebot}


### Description
This [script](https://github.com/Frank-Hebert/lane_control) is meant as a first exploration of using visual servoing 
to control the duckiebot. 

### Architecture

Here are the main components of this project.

<p>&nbsp;</p>

* include/estimation.py contains everything that is related to estimating the relative 
pose of the target point to the robot.
* include/control.py contains everything that is related to generating the
duckiebot commands from the pose estimation
* config/lane_controller_node/default.yaml contains the configuration values
* src/lane_controller_node.py contains the logic to calculate and send the commands to the wheels

### Usage

Add this package like you would with any other packages. In your dt-exercises folder, go to your packages location (.../exercises_ws/src/) and git clone this github.

<p>&nbsp;</p>

To activate the code on your robot, go to an exercises folder of your choice and run these commands :

```console
dts exercises build
dts exercises test --duckiebot_name ![ROBOT_NAME] --local --pull
```

Then, open your browser and go to your local host address (http://localhost:8087/). From this NoVNC environement, launch the joystick controller and press "A" to activate the following and parking procedure.

### Expected behaviour
If your duckiebot detect a circles grid pattern (rear bumper of a duckiebot), it will go toward it and will park itself behind the other duckiebot.


### Known problems
The robot can oscillate when he's near the other bumper. By sending a command high enough to fight the static friction, it can overshoot the target and it will repeat this pattern again and again.