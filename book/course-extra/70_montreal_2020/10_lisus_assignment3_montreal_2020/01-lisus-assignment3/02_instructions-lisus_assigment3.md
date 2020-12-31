# Lisus Assignment 3 Instructions {#instructions-lisus-assignment3 status=ready}

<div class='requirements' markdown="1">

Requires: Duckiebot in configuration  `DB18` or `DB19`.

Requires: Duckietown without intersections.

Requires: Camera calibration completed.

Requires: Wheel calibration completed.

</div>

## Video of expected results {#demo-lisus-assignment3-expected}

This video shows the Gaussian Process (GP) lane state estimate being used in a pure-pursuit controller. These instructions detail how to reproduce this result.

<figure id="cross-right_turn_inst">
    <figcaption> Expected pure pursuit results.</figcaption>
    <dtvideo src="vimeo:49607814"/>
</figure>

## Laptop setup notes {#demo-lisus-assignment3-laptop-setup}

The laptop should be pushed with the latest version of the `duckietown shell`. The full folder implemented using the *dt-exercises* approach can be found at https://github.com/lisusdaniil/dt-exercises/tree/daffy/lane_state_prediction. Simply clone the repo and launch `dts exercises test` with either the simulator or the duckiebot implementation.


## Duckietown setup notes {#demo-lisus-assignment3-duckietown-setup}

This is a simple lane following algorithm so ideally the duckietown is set up in an infinite loop!

## Pre-flight checklist {#demo-lisus-assignment3-pre-flight}

**Check 1:** The duckiebot is fully set up and has been successfully [moved]{#rc-control} and you can see what the [duckiebot sees]{#read-camera-data}.

**Check 2:** The `duckietown shell` and duckiebot have both been upgraded to the latest version.

**Check 3:** The `lane_state_prediction` folder is cloned and `dts exercises build` has been run successfully.

**Check 4:** In order to train the GP model, the users personal machine must have the`json`, `numpy`, `sklearn`, `pickle`, and `matplotlib` Python libraries installed. This is because the GP training file is run outside of the duckietown shell!

## Pure Pursuit Implementation Instructions {#demo-lisus-assignment3-run}

Below are instructions to reproduce the video shown above, which makes use of a trained GP model to predict the lane state in a pure pursuit controller. These instructions include changes needed to run the algorithm in the simulator vs on the duckiebot.

**Step 1:** Go to the `config/default.yaml` files in the `lane_controller_node` and in the `line_detector_node` and make sure that either the SIM or IRL settings are uncommented depending on whether you wish to run the test in the simulator or in the real world respectively. For the `line_detector_node`, it is recommended that the user tunes the colour settings to their own environment for best results.

**Step 2:** Either launch the simulator through `dts exercises test --sim` or launch the code on the duckiebot `dts exercises test -b ![DUCKIEBOT_NAME] --local`. 

**Step 3:** Open up the noVNC2 web app hosted by default at [localhost:8087](localhost:8087). Launch the virtual joystick app by running `dt-launcher-joystick` in the noVNC2 command prompt to control the robot. The robot's current view can be visualized in `RQT Image View` on topic `/agent/camera_node/image/compressed`. Additionally, the user can view the detected and projected line segments that are used in the GP prediction through the `RQT Image View` on topic `/agent/ground_projection_node/debug/ground_projection_image/compressed`.

**Step 4:** Use the joystick to ensure the duckiebot is positioned within the lane, ideally facing the correct flow of traffic head on.

**Step 5:** Simply press the <kbd>a</kbd> key while in the joystick to start the autonomous lane following. Exit lane following and control the bot manually at any point by pressing the <kbd>s</kbd> key.

**Step 6:** Regardless of whether the lane following is activated, the program will print the currently predicted lane state to the command prompt of your local machine from where the the exercise was launched from. Drive around manually and compare the output to what you think the lane state is. The provided trained GP model should produce an accuracy of approximately 90% in simulation and 80% in the real world.

## Data Collection and Training Instructions{#instructions-data-record-lisus-assignment3}

Below are instructions if the user wishes to collect more data and train their own model.

**Step 1**: Specify the desired file name to store your data in through the `train_filename_GP `parameter in the `lane_controller_node` `config/default.yaml` file. Please ensure that the file ends with `.json`.

**Step 2:** Follow Steps 1-3 from the [pure pursuit instructions]{#demo-lisus-assignment3-run}.

**Step 3:** Use the joystick to drive the duckiebot around to desired collection spots. Press the <kbd>e</kbd> key to trigger saving mode and then click the <kbd>up</kbd>, <kbd>left</kbd>, or <kbd>right</kbd> arrows to save the current line segments as corresponding to a straight, left, or right lane state respectively. As soon as a segment is saved, the terminal will display your selection and the saving mode will exit. If you wish to exit saving mode without saving anything, simply click the <kbd>s</kbd> key.

**Step 4:** When you have collected all of your desired data, press the <kbd>e</kbd> key and then the <kbd>down</kbd> arrow to save your data to the file name you specified in Step 1. This will not erase your current data, so you can continue adding to your current set after saving. The file will be saved to the `/code/exercise_ws/data/` directory.

**Step 5: ** If you wish to restart collection or to save to a different file name, change the file name and restart your agent.

**Step 6:** When you have collected whatever datasets you wish, open the `train_GP.py` file and include the desired files in the `list_train_files` or `list_test_files`. The lists can take any number of seperate files. There are also tags to control whether to load the GP model from file or to train a new one, whether to save the model you'll use, and whether to do data augmentation. The parameters for the training have been chosen based on some trial and error, but the reader is referred to https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html for additional information.

**Step 7:** If `save_model` is set to True, the trained model will be saved in the `/code/exercise_ws/data/` directory under the name specified in `model_save_name`. If you wish to run the lane following with your new model, simply indicate the name of the model in the `GP_model_file` parameter in the `lane_controller_node` `config/default.yaml` file. Only the filename itself needs to be specified, as the code automatically calls the `/code/exercise_ws/data/` directory. 

## Troubleshooting {#demo-lisus-assignment3-troubleshooting}

Add here any troubleshooting / tips and tricks required, in the form:


Symptom: The Duckiebot flies

Resolution: Unplug the battery and send an email to info@duckietown.org


Symptom: I run `this elegant snippet of code` and get this error: `a nasty line of gibberish`

Resolution: Power cycle until it works.

## Demo failure demonstration {#demo-lisus-assignment3-failure}

The controller could fail if the GP model makes incorrect predictions. The reader is referred to the [results section]{#lisus-assignment3-final-formal}

for more details about the types of prediction failures and a discussion on their significance.