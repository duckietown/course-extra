# Instructions template {#instructions-improving-rl-baseline status=ready}

<div class='requirements' markdown="1">

Requires: Python 3.8

</div>

<div class='requirements' markdown="1">

Requires: Duckiebot in configuration `DB19`

Requires: Duckietown without intersections

Requires: Camera calibration completed

</div>

## Video of expected results {#demo-improving-rl-baseline-expected}

First, we show a video of the expected behavior (if the demo is successful).

Make sure the video is compliant with Duckietown, i.e. : the city meets the [appearance specifications](+opmanual_duckietown#dt-ops-appearance-specifications) and the Duckiebots have duckies on board.

## Laptop setup notes {#demo-improving-rl-baseline-laptop-setup}

Does the user need to do anything to modify their local laptop configuration?


## Duckietown setup notes {#demo-improving-rl-baseline-duckietown-setup}

Here, describe the assumptions about the Duckietown, including:

* Layout (tiles types)
* Infrastructure (traffic lights, WiFi networks, ...) required
* Weather (lights, ...)

Do not write instructions on how to build the city here, unless you are doing something very particular that is not in the [Duckietown operation manual](+opmanual_duckietown#duckietowns). Here, merely point to them.

## Duckiebot setup notes {#demo-improving-rl-baseline-duckiebot-setup}

Write here any special setup for the Duckiebot, if needed.

Do not repeat instructions here that are already included in the [Duckiebot operation manual](+opmanual_duckiebot#opmanual_duckiebot).

## Pre-flight checklist {#demo-improving-rl-baseline-pre-flight}

The code is in Python 3.8.

## Instructions {#demo-improving-rl-baseline-run}

Clone the project repo:

`git clone https://github.com/melisandeteng/challenge-aido_LF-baseline-RL-sim-pytorch.git`

Check that you are working from the ![DARLA] branch.
Install the requirements in ![requirements.txt].  
 Then, initialize the submodules. 

`cd challenge-aido_LF-baseline-RL-sim-pytorch`

`git submodule init`

`git submodule update`


### Collecting your dataset

### Training the perceptual model
You will be using the DARLA submodule. 
Check that the version you are using is the same as (here)[https://github.com/lifetheater57/DARLA/tree/duckietown-rl].
Check the ![config/defaults.yaml] configuration file, this is where you can choose to train your beta-VAE, or to train a DAE to get targets to use for training the beta-VAE subsequently. 

$ python train.py --config CONFIG_FILE 


If you wish to train the beta-VAE with the original input images as targets, use : 

$ python train.py --config CONFIG_FILE --no-dae


## Troubleshooting {#demo-improving-rl-baseline-troubleshooting}


## Demo failure demonstration {#demo-improving-rl-baseline-failure}

Finally, put here video of how the demo can fail, when the assumptions are not respected.

You can upload the videos to the [Duckietown Vimeo account](https://vimeo.com/duckietown) and link them here.
