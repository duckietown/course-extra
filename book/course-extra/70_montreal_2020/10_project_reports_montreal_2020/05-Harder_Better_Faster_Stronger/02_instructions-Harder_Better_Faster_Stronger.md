# DuckieTown adaptation of An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions {#instructions-Harder_Better_Faster_Stronger status=ready}

This project is an adaption of the [https://github.com/sercant/mobile-segmentation](https://github.com/sercant/mobile-segmentation) repository. They presented a computationally efficient approach to semantic segmentation, while achieving a high mean intersection over union (mIOU), 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices.

This models performs pretty well in the Duckietown Simulator too! 75.60% on the Duckietown Segmentation Dataset.

This page describes how to implement the project Harder, Better, Faster, Stroger and get similar results as we did.

<div class='requirements' markdown="1">

Requires: Duckiebot in configuration `DB19`

Requires: Duckietown without intersections

Requires: Camera calibration completed

Requires: Laptop with Duckietown Shell command installed and setup.

</div>

## Video of expected results {#demo-Harder_Better_Faster_Stronger-expected}

If all the instuctions are followed and the entire setup is succesfull, you should get similar performance on the AIDO LF challenge as shown below:

<br/>
<figure>
    <figcaption>Lane Following Performance on the AIDO LF challenge</figcaption>
    <dtvideo src="vimeo:493139195"/>
</figure>
<br/>


<br/>
<figure>
    <figcaption>Lane Following Performance on the Duckiebot</figcaption>
    <dtvideo src="vimeo:493555586"/>
</figure>

## Laptop setup notes {#demo-Harder_Better_Faster_Stronger-laptop-setup}

Laptop must have Duckietown Shell, Docker, etc, as configured in [Unit C-1 - Laptop Setup](https://docs.duckietown.org/daffy/opmanual_duckiebot/out/laptop_setup.html)


## Duckietown setup notes {#demo-Harder_Better_Faster_Stronger-duckietown-setup}

We make the following assumtions about the Duckietown setup:

* Duckietown without intersections required
* Enough lighting around the track

## Duckiebot setup notes {#demo-Harder_Better_Faster_Stronger-duckiebot-setup}

The Duckiebot must be in `DB19` configuration.

## Pre-flight checklist {#demo-Harder_Better_Faster_Stronger-pre-flight}

The pre-flight checklist describes the steps that are sufficient to ensure that the demo will be correct:

Check: A virtual environment, as it will prevent conflicts with the python installed on the computer.

Check: All the packages as mention in the `requirements.txt` file are installed in your virtual environment.

## Instructions {#demo-Harder_Better_Faster_Stronger-run}

### Getting ready {#demo-Harder_Better_Faster_Stronger-getting-ready}

1. Download or generate a "raw" DuckieTown dataset. [Refer to the Duckietown Dataset Generator](https://github.com/rjean/mobile-segmentation/blob/master/dataset_generator/README.md)
2. Convert the dataset to a MS COCO Compatible format that can be used by most segmentation models. [Refer to the Conversion Jupyter Notebooks](https://github.com/rjean/mobile-segmentation/blob/master/notebooks/README.md)
3. Prepare COCO Compatible dataset for training. Example scripts and code is available under the `dataset` folder. The dataset should be in `tfrecord` format.


### Model zoo {#demo-Harder_Better_Faster_Stronger-model-zoo}

Please refer to the original repository for pre-trained models.

### Training {#demo-Harder_Better_Faster_Stronger-training}

To learn more about the available flags you can check `common.py` and the specific script that you are trying to run (e.g. `train.py`).

2-3 epochs of fine-tuning should be enough, more would likely cause overfitting. The model is already pre-trained on Cityscapes, so the final training is basically domain adaptation. 

The "output_stride" parameter can be used to allow this network to work on smaller resolution images. The Network was originally designed to work with 640x480 images, with an output stride of 16. For smaller images, such as the ones we use in Duckietown, the bottleneck is to narrow. Reducing the output stride to 8 for 320x240 and 4 for 160x120 reduce this bottleneck. The only drawback is that those network take as much time as the 640x480 image on lower resolution image. 

Long story short: Next time, we should generate 640x480 datasets, because lowering the resolution will not help!

**Example Training Configuration**

Training on Duckietown:

```
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/model.ckpt \
    --training_number_of_steps=12000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=4 \
    --train_crop_size=120 \
    --train_crop_size=160 \
    --train_batch_size=16 \
    --dataset=duckietown \
    --train_split=train \
    --dataset_dir=./dataset/duckietown2/merged_with_real/tfrecords \
    --save_summaries_images \
    --train_logdir=./logs \
    --loss_function=sce
```

Training on Duckietown with Bezier:

```
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/model.ckpt \
    --training_number_of_steps=120000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=8 \
    --train_crop_size=240 \
    --train_crop_size=320 \
    --train_batch_size=16 \
    --dataset=duckietown \
    --train_split=train \
    --dataset_dir=./dataset/duckietown/bezier/tfrecords \
    --save_summaries_images \
    --train_logdir=./logs \
    --loss_function=sce
```

Training on Cityscapes:

```
python train.py \
    --model_variant=shufflenet_v2 \
    --tf_initial_checkpoint=./checkpoints/model.ckpt \
    --training_number_of_steps=120000 \
    --base_learning_rate=0.001 \
    --fine_tune_batch_norm=True \
    --initialize_last_layer=False \
    --output_stride=16 \
    --train_crop_size=769 \
    --train_crop_size=769 \
    --train_batch_size=16 \
    --dataset=cityscapes \
    --train_split=train \
    --dataset_dir=./dataset/cityscapes/tfrecord \
    --train_logdir=./logs \
    --loss_function=sce
```

Training with 8gb commodity GPU:
```
python train.py     --model_variant=shufflenet_v2     --tf_initial_checkpoint=./checkpoints/model.ckpt     --training_number_of_steps=120000     --base_learning_rate=0.001     --fine_tune_batch_norm=True     --initialize_last_layer=False     --output_stride=16     --train_crop_size=769     --train_crop_size=769     --train_batch_size=3     --dataset=cityscapes     --train_split=train     --dataset_dir=./dataset/cityscapes/tfrecord     --train_logdir=./logs     --loss_function=sce
```

**Important:** To use DPC architecture in your model, you should also set this parameter:

    --dense_prediction_cell_json=./core/dense_prediction_cell_branch5_top1_cityscapes.json

### Evaluation {#demo-Harder_Better_Faster_Stronger-evaluation}

The trained model can be evaluated by executing the following commands on your laptop terminal:

**Example evaluation configuration**

Duckietown "merged_with_real":
```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=120 \
    --eval_crop_size=160 \
    --output_stride=4 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=duckietown \
    --dataset_dir=./dataset/duckietown2/merged_with_real/tfrecords
```
Duckietown "bezier":
```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=240 \
    --eval_crop_size=320 \
    --output_stride=8 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=duckietown \
    --dataset_dir=./dataset/duckietown2/bezier/tfrecords
```


Cityscapes:
```sh
python evaluate.py \
    --model_variant=shufflenet_v2 \
    --eval_crop_size=1025 \
    --eval_crop_size=2049 \
    --output_stride=4 \
    --eval_logdir=./logs/eval \
    --checkpoint_dir=./logs \
    --dataset=cityscapes \
    --dataset_dir=./dataset/cityscapes/tfrecord
```

### Visualize {#demo-Harder_Better_Faster_Stronger-visualize}

In order to visualize and see the results of the trained model you can run the following commands in the terminal on your laptop:

**DuckieTown**
In order to visualize segmentation for the Duckietown dataset:
```
python visualize.py --checkpoint_dir logs \
     --vis_logdir logs \
      --dataset_dir dataset/duckietown2/merged_with_real/tfrecords/ \
      --output_stride 4 \
      --dataset duckietown
```

**Cityscapes**
In order to visualize segmentation for the Cityscapes dataset:
```
python visualize.py --checkpoint_dir checkpoints --vis_logdir logs --dataset_dir dataset/cityscapes/tfrecord/
```

**Important:** If you are trying to evaluate a checkpoint that uses DPC architecture, you should also set this parameter:  

    --dense_prediction_cell_json=./core/dense_prediction_cell_branch5_top1_cityscapes.json

### Running on Duckietown {#demo-Harder_Better_Faster_Stronger-running-on-duckietown}

A pure pursuit controller will take as an input the output of the points generated by the segementation mask. 

See the [solution folder](https://github.com/rjean/mobile-segmentation/tree/master/solution)


## Troubleshooting {#demo-Harder_Better_Faster_Stronger-troubleshooting}

Add here any troubleshooting / tips and tricks required, in the form:


Symptom: The Duckiebot flies

Resolution: Unplug the battery and send an email to info@duckietown.org


Symptom: I run `this elegant snippet of code` and get this error: `a nasty line of gibberish`

Resolution: Power cycle until it works.

## Demo failure demonstration {#demo-Harder_Better_Faster_Stronger-failure}

Finally, put here video of how the demo can fail, when the assumptions are not respected.

You can upload the videos to the [Duckietown Vimeo account](https://vimeo.com/duckietown) and link them here.
