# Instructions Cross Pro-Duck {#instructions-cross-pro-duck status=ready}

<div class='requirements' markdown="1">

<!-- Requires: Duckiebot in configuration `DB18` or `DB19` -->

Requires: Duckietown Simulator without obstacles. Project Code: [GitHub repo](https://github.com/himarora/dt-visual-servoing/tree/visual-servoing-cv-checkpoints/visual-servoing)

<!-- Requires: Camera calibration completed -->

</div>

## Video of expected results {#demo-cross-pro-duck-expected}

Below is a compilation of some of our best runs with the duckiebot performing right turns, crossing straight through an intersection, and doing lane following up to a red line! 

<figure id="cross-right_turn_inst">
    <figcaption> Succesfull Right Turns.</figcaption>
    <dtvideo src="vimeo:495935292"/>
</figure>

<figure id="cross-lane_follow_inst">
    <figcaption> Succesfull Lane Following with Red Line Approach.</figcaption>
    <dtvideo src="vimeo:495954271"/>
</figure>

<figure id="cross-straight_turn_inst">
    <figcaption> Succesfull Straight Crossing.</figcaption>
    <dtvideo src="vimeo:495954451"/>
</figure>

## Laptop setup notes {#demo-cross-pro-duck-laptop-setup}
The user is expected to have the latest version of `duckietown-shell` installed. It is also expected to have the latest Duckietown Docker images. In addition, the user needs to install the Python packages that we list down in the `requirements.txt` file within our project directory. The main requirement of interest is the python `dubins`[^dubins_python] library, which is required for the path planning. Although not the most long term solution, this package was included by simply adding `pip3 install dubins` in the `launchers/run_interface.sh` file.


## Duckietown setup notes {#demo-cross-pro-duck-duckietown-setup}
Although we only test our solution in the `LFI-norm-4way-000` map in the simulator, we expect it to work similarly in any other map that does not involve obstacles such as other duckiebots, duckies, cones, etc. This means that any map can be chosen from [here](https://github.com/duckietown/challenge-aido_LF/tree/daffy-aido5/sampling/sampled) provided it has `LF` or `LFI` as prefix. 

<!-- ## Duckiebot setup notes {#demo-groupname-duckiebot-setup}

Write here any special setup for the Duckiebot, if needed.

Do not repeat instructions here that are already included in the [Duckiebot operation manual](+opmanual_duckiebot#opmanual_duckiebot). -->

## Pre-flight checklist {#demo-cross-pro-duck-pre-flight}

**Check 1:** The Duckietown shell is working by running `dts`. It is expected to see the shell's version and a prompt waiting for a command.

**Check 2:** You have cloned the project source code from [here](https://github.com/himarora/dt-visual-servoing/tree/visual-servoing-cv-checkpoints/visual-servoing) and you are in the root of the project. You should see `exercise_ws` as one of the directories on running `ls`.

**Check 3:** The `exercise_ws` directory has all of the custom ROS nodes that we provide: `custom_msgs`, `ground_projection`, `lane_control`, `lane_filter`, `line_detector`, and `vision_opencv`.

**Check 4:** The `exercise_ws/checkpoints` directory has at least one directory containing a set of images referring to the kind of path the duckiebot is expected to follow.

## Instructions {#demo-cross-pro-duck-run}

Here, give step by step instructions to reproduce the demos shown in [the videos]{#demo-cross-pro-duck-expected}.

**Step 1**: In the `line_detector/config/default.yaml` file, uncomment the desired `checkpoints_path` as described by the ending of the path. Note that `turn_left` does not currently work.

**Step 2**: Build the exercise by running `dts exercises build`. You should see messages like "All 9 packages succeeded!" and "Build complete" at the end.

**Step 3:** Run `dts exercises test --sim --pull` command. You should initially see the Duckietown shell pulling the latest Docker images. Once that is done, the shell will launch the ROS interface and the agent. 

**Step 4**: The agent can be controlled via the noVNC2 web app hosted by default at [localhost:8087](localhost:8087). Launch the virtual joystick app by running `dt-launcher-joystick` in the noVNC2 command prompt to control the robot. The robot's current view can be visualized in `RQT Image View` on topic `/agent/camera_node/image/compressed`. The detected lines can be viewed on topic `/agent/line_detector_node/debug/vs_lines_all/compressed`.

**Step 5**: For the purpose of this demonstration, the user is expected to drive the Duckiebot to a point from where the checkpoint image is visible. For an intersection crossing, this means driving the bot right in front of the red line until the red line to the right of you is visible. For lane following/red line approach the bot simply needs to be in a lane. This is to ensure that the target image is visible from the current view which is a requirement of the Visual Servoing algorithm.

**Step 6**: To make the robot drive autonomously through the checkpoints, the user can run `docker exec -it $(docker ps -aqf "name=^agent$") /bin/bash /code/exercise_ws/src/servo.sh` command in their home terminal or simply press <kbd>e</kbd> and then the <kbd>up</kbd> arrow in the virtual joystick. Then, press <kbd>a</kbd> in the virtual joystick. The robot should then start moving, follow the reference trajectory, stop once it has reached the end, without any intervention.

**Step 7:** To manually drive the robot through the checkpoints, simply follow the same procedure as in Step 6, however do not press <kbd>a</kbd>. You can simply use the arrow keys in the virtual joystick to drive the robot with the checkpoints being active. It is possible to switch into autonomous mode at any point by pressing <kbd>a</kbd> and to switch back to manual mode by pressing <kbd>s</kbd>. If you wish to stop checkpoints from being used, simply press <kbd>e</kbd> and then the <kbd>down</kbd> arrow in the virtual joystick.

**Step 8**: Interested users are again referred to the topic `/agent/line_detector_node/debug/vs_lines_all/compressed` to monitor the state of the robot. The visualization displays all the detected yellow, red, and white lines. It also displays in green the lines or intersection points in blue that are being used to compute the transformation matrix.

**Step 9**: Once finished, users can drive the robot back to the starting position to try again or load a different set of checkpoints by specifying its path in  parameter of `LineDetectorNode`'s config and restarting the agent.

#### Collecting Checkpoints
In addition to the default set of checkpoints that we provide, it is easy to collect custom checkpoints. 

**Step 1**: Set the `checkpoints_path` parameter of `LineDetectorNode`'s config to `"None"`.

**Step 2**: Perform Steps 2 - 4 of the previous section

**Step 3**: Using the joystick, drive to the locations that you want to save as checkpoints. At each intended location, run `docker exec -it $(docker ps -aqf "name=^agent$") /bin/bash /code/exercise_ws/src/checkpoint.sh` or press <kbd>e</kbd> and then the <kbd>left</kbd> arrow in the virtual joystick to save the current image as a checkpoint. Repeat this step until all the checkpoints have been captured. At each save, you should see an stdout message like "Saved checkpoint with id xxxxxxxx. Total checkpoints: 3."

**Step 4**: Run `docker exec -it $(docker ps -aqf "name=^agent$") /bin/bash /code/exercise_ws/src/checkpoints_to_disk.sh`or <kbd>e</kbd> and then the <kbd>right</kbd> arrow in the virtual joystick to save all the collected checkpoints to disk. To load the checkpoints in future, simply set the `checkpoints_path` parameter to the path printed in the output of the command.

<!-- Make sure you are specifying where to write each line of code that needs to be executed, and what should the expected outcome be. If there are typical pitfalls / errors you experienced, point to the next section for troubleshooting. -->

## Troubleshooting {#demo-cross-pro-duck-troubleshooting}

Symptom: The Duckiebot did not follow the reference trajectory/checkpoints.

Resolution: Unfortunately, our solution is not very robust right now and it is expected that the robot will occasionally irreversibly deviate from its intended path. If this happens, stop the robot by pressing the key <kbd>s</kbd> and drive back to the initial position to try again or try different checkpoints/maps/turns. It is strongly advised that users try to manually run through checkpoints before engaging autonomous mode. Most of our successful experiments involve checkpoints for lane following, turning right or going straight at an intersection.

Symptom: The visual output gets stuck in the simulator.

Resolution: Follow the regular steps of reloading the agent, simulator, and your machine in that order. Sometimes problems would get fixed only after a good old fashioned restart of the entire machine, although the reasoning behind this is unclear...

Symptom: Some error gets thrown but the code continues running afterwards.

Resolution: Some fringe detection failures can throw errors at the agent start or even during runtime if something very strange is going on. If the code continues running then no restart is required.

Symptom: My saved checkpoints are not being loaded.

Resolution: Make sure you specify the full path `/code/exercise_ws/checkpoints/YOUR_FOLDER_NAME`.

Symptom: My checkpoints are not being added or saved.

Resolution: Make sure your `checkpoints_path` is set to `None`.

## Demo failure demonstration {#demo-cross-pro-duck-failure}
Below is a video showing some of the common failures that we came across. Note that these occured with various settings, some of which have been better tuned since then. However, if a reader is to progress this work then they will inevitably change the parameters again and may observe similar issues. 

In no particular order, some of the failures are:

* Overly aggressive turning, forward velocity.
* Under aggressive turning, forward velocity.
* Line detector gets confused and starts trying to align itself with an incorrect line.
* Poorly set homography tolerance leading to the checkpoint getting stuck.
* Poor checkpoint choices leading to the robot being completely lost. This is especially difficult for left turns.

<figure id="cross-failues">
    <figcaption> Succesfull Straight Crossing.</figcaption>
    <dtvideo src="vimeo:495966350"/>
</figure>

## Reference

[^dubins_python]: https://pypi.org/project/dubins/
