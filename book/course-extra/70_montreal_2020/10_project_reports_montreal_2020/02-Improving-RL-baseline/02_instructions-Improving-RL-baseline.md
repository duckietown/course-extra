# Instructions template {#instructions-improving-rl-baseline status=ready}

<div class='requirements' markdown="1">

Requires: Python 3.6+

Requires: PyTorch 1.7.x

Requires: access to a Nvidia GPU (locally or remotely)

</div>

## Images of expected results {#demo-improving-rl-baseline-expected}

TODO: Add some DAE results dans VAE results.

## Laptop setup notes {#demo-improving-rl-baseline-laptop-setup}

If you are using a local Nvidia GPU, be sure to have access to it with PyTorch.

To test it, open a python session within your development environment.

Then, import PyTorch and check if the GPU is available:

```
import torch
"cuda" if torch.cuda.is_available() else "cpu"
```

If you get `cuda`, you're right to go. Otherwise, you have to check your PyTorch installation and be sure that you have a CUDA enabled version of PyTorch and that the required CUDA version matches the version of the library installed on your laptop. If you have to reinstall it, go on PyTorch website for [instructions](https://pytorch.org/get-started/locally/).

## Duckietown setup notes {#demo-improving-rl-baseline-duckietown-setup}

Nothing special about the duckietown setup is required beforehand to reproduce the results. The required Duckietown map used in the simulator is available under `duckietown_rl/maps/dataset_generator.yaml`.

## Duckiebot setup notes {#demo-improving-rl-baseline-duckiebot-setup}

Nothing special was done with the duckiebot setup.

## Pre-flight checklist {#demo-improving-rl-baseline-pre-flight}

The code is in Python 3.6 and use PyTorch 1.7.1.

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

To collect the dataset, follow the instructions [TODO: add instructions]

In the code, the camera images for a run will be saved in one numpy array. 

In order to use your dataset to train the perceptual model, you will need to convert the arrays to image files (JPG, PNG, JPEG). 
Then, make a ![.txt] file containing the paths to the images you want to include for training your model. You can check the [make_txt.py](https://github.com/lifetheater57/DARLA/blob/duckietown-rl/utils/make_txt.py) script as a helper. 
### Training the perceptual model
You will be using the DARLA submodule. 
Check that the version you are using is the same as (here)[https://github.com/lifetheater57/DARLA/tree/duckietown-rl].
Check the ![config/defaults.yaml] configuration file, this is where you can choose to train your beta-VAE, or to train a DAE to get targets to use for training the beta-VAE subsequently. 

`python train.py --config CONFIG_FILE`


If you wish to train the beta-VAE with the original input images as targets, use : 

`python train.py --config CONFIG_FILE --no-dae`


## Troubleshooting {#demo-improving-rl-baseline-troubleshooting}


## Demo failure demonstration {#demo-improving-rl-baseline-failure}

Finally, put here video of how the demo can fail, when the assumptions are not respected.

You can upload the videos to the [Duckietown Vimeo account](https://vimeo.com/duckietown) and link them here.
