# Lisus Assignment 3 Instructions {#instructions-lisus-assignment3 status=ready}

<div class='requirements' markdown="1">

Requires: Duckiebot in configuration  `DB18` or `DB19`.

Requires: Duckietown without intersections.

Requires: Camera calibration completed.

Requires: Wheel calibration completed.

</div>

## Video of expected results {#demo-lisus-assignment3-expected}

First, we show a video of the expected behavior (if the demo is successful).

Make sure the video is compliant with Duckietown, i.e. : the city meets the [appearance specifications](+opmanual_duckietown#dt-ops-appearance-specifications) and the Duckiebots have duckies on board.

## Laptop setup notes {#demo-lisus-assignment3-laptop-setup}

The laptop should be pushed with the latest version of the `duckietown shell`. The full folder implemented using the *dt-exercises* approach can be found at https://github.com/lisusdaniil/dt-exercises/tree/daffy/lane_state_prediction. Simply clone the repo and launch `dts exercises test` with either the simulator or the duckiebot implementation.


## Duckietown setup notes {#demo-lisus-assignment3-duckietown-setup}

This is a simple lane following algorithm so ideally the duckietown is set up in an infinite loop!

## Pre-flight checklist {#demo-lisus-assignment3-pre-flight}

**Check 1:** The duckiebot is fully set up and has been succesfully [moved]{#rc-control} and you can see what the [duckiebot sees]{#read-camera-data}.

**Check 2:** The `duckietown shell` and duckiebot have both been upgraded to the latest version.

**Check 3:** The `lane_state_prediction` folder is cloned and `dts exercises build` has been run succesfully.

## Data Collection Instructions{#instructions-data-record-lisus-assignment3}

## Instructions {#demo-lisus-assignment3-run}

Here, give step by step instructions to reproduce the demo.

Step 1: XXX

Step 2: XXX

Make sure you are specifying where to write each line of code that needs to be executed, and what should the expected outcome be. If there are typical pitfalls / errors you experienced, point to the next section for troubleshooting.

## Troubleshooting {#demo-lisus-assignment3-troubleshooting}

Add here any troubleshooting / tips and tricks required, in the form:


Symptom: The Duckiebot flies

Resolution: Unplug the battery and send an email to info@duckietown.org


Symptom: I run `this elegant snippet of code` and get this error: `a nasty line of gibberish`

Resolution: Power cycle until it works.

## Demo failure demonstration {#demo-lisus-assignment3-failure}

Finally, put here video of how the demo can fail, when the assumptions are not respected.

You can upload the videos to the [Duckietown Vimeo account](https://vimeo.com/duckietown) and link them here.
