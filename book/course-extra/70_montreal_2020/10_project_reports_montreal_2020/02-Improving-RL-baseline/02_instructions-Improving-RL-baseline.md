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

### Initializing the project directory

To initialize the project directory follow the instructions below.

Clone the project repo.

    $ git clone https://github.com/melisandeteng/challenge-aido_LF-baseline-RL-sim-pytorch.git

Get into the project directory.

    $ cd challenge-aido_LF-baseline-RL-sim-pytorch

Checkout the `darla` branch.

    $ git checkout darla

Install the requirements.

    $ python3 -m pip install -r requirements.txt
 
 Then, initialize the submodules. 

    $ git submodule init
    $ git submodule 
    $ update

### Config file

Enter the DARLA directory.

    $ cd DARLA

You might want to copy the default configuration file `config/defaults.yaml` or edit the file directly.

The following table summerize the configuration values that are either not straight forward or that you most likely will have to change.

Option | Description
--- | ---
`output_path` | the location of the output files
`module` | which neural network to train
`save_n_epochs` | the checkpoint interval
`data:loaders:batch_size` | number of element per batch
`data:loaders:num_workers` | number of worker for the loader
`data:files:base` | the location of the file listing examples
`data:files:train` | file listing examples
`data:shape` | the shape of the images in format channel, height, width
`comet:api_key` | the commet key associated with your account
`comet:project_name` | the commet name of your project
`comet:workspace` | the commet workspace of your project

Be sure to check the configuration file values. The default values related to the model were chosen to fit the original DARLA paper [](#bib:higgins2018darla). The other values are generic values that you might want to tune. In particular, if your GPU has access to a lot of RAM you might want to increase the `data:loaders:batch_size` value (up to 32 for a 6 Gb GPU for example). If your CPU has several cores, you might want to increase the `data:loaders:num_workers` value (1 per virtual core for example).

If you want to use [Comet ml](https://www.comet.ml/) to visualize you training statistics, then set the values under the `comet` section of the config file. 

Note: if you change the size in `shape`, set the same size in the resize transform section of the config file and set the `ignore` flag of this transform to `false`.

### Collecting your dataset

To collect the dataset from the source of the project directory follow the instructions below.

Enter the RL directory.

    $ cd ../duckietown_rl

Generate the dataset.

    $ python3 dataset_generator_control.py --env-name Duckietown-loop-empty

The following options are also available.

Option | Description | Default value
--- | --- | ---
`dataset-size` | number of images to generate | 50
`dataset-path` | location to save the dataset | 'datasets/image_%Y_%m_%d_%H_%M_%S/'
`compress` | save the images as a series of png pictures rather than npy file(s) | False
`split` | number of images per file (if used without --compress) | 2000

To reproduce our results, use the map we created by appending `--map-name $PWD/maps/dataset_generator.yaml` to the command. Later, you will need to have the images in png format, so you should use the flag `--compress`.

Once the files are generated, if they were generated in png, `cd` in the folder that contain the images. Then, list then in a file.

    $ ls -d $PWD/* > train.txt

Finally, copy `train.txt` in the directory designed by the keys ̀`data:files:base` and assign the name of this file to the key `data:files:train` of config file you will use (`config/default.yaml` by default).

### Training the perceptual model

Enter the DARLA directory.

    $ cd ../DARLA

If you chose to edit `config/defaults.yaml` directly, then you can launch the training with the following command.

    $ python train.py

If you chose to create a new configuration file let's say `custom.yaml`, then execute the following command instead.

    $ python train.py --config "./config/custom.yaml"

There are different options available for the training of the $\beta$-VAE or the DAE.

Option | Description | Default value
--- | --- | ---
`config` | path to config file | ./config/defaults.yaml
`no-comet` | launch Comet exp or not | false
`comet-tags` | tags for Comet exp | None
`dae-checkpoint` | dae checkpoint from which to start the training | None
`vae-checkpoint` | vae checkpoint from which to start the training | None
`no-dae` | train the VAE without using a DAE for the loss | false
`exp-id` | Comet experience id to resume | None

If you want to resume your training from a checkpoint, then use the `--dae-checkpoint path/to/checkpoint.pth` or `--vae-checkpoint path/to/checkpoint.pth` flags. The paths are relative to ̀`output_path/checkpoints/` where `output_path` is the corresponding value in the config file. If you resume an experiment, use Comet and also want to resume the experiment in Comet, don't forget to use the `--exp-id "experiment_hash"` flag.

If you wish to train the beta-VAE using directly the original input images and the reconstructed images instead of applying the DAE before using them in the loss function, use the `--no-dae` flag.

If you want to add some flags to your Comet experiment, then you can add them with `--comet-tag tag1 tag2`.

TODO: add ref to the dev process in the report.

## Troubleshooting {#demo-improving-rl-baseline-troubleshooting}

Nothing ever go wrong, right?

## Demo failure demonstration {#demo-improving-rl-baseline-failure}

TODO: add image of bad DAE results
