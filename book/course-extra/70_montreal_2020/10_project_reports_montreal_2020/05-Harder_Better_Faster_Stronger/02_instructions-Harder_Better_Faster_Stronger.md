# Harder Better Faster Stronger : Project Instructions {#instructions-Harder_Better_Faster_Stronger status=ready}

This project is an adaption of the [https://github.com/sercant/mobile-segmentation](https://github.com/sercant/mobile-segmentation) repository. They presented a computationally efficient approach to semantic segmentation, while achieving a high mean intersection over union (mIOU) of 70.33% on the Cityscapes challenge. The network proposed is capable of running real-time on mobile devices.

This models performs pretty well in the Duckietown Simulator too! The mIoU is 75.60% on the Duckietown Segmentation Dataset.

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

Here is some common problem your might encounter with our solution.

Symptom: The Duckiebot does not move.

Resolution: Check battery charge. Hard reboot the Duckybot. (I know, but it is still worth mentionning)

Symptom: Segmentation Quality is poor, and the Duckiebot crashes.

Resolution: We fine-tuned different models for different task. Please make sure than the model specified in the Object Detection Node is the right one. See the following two files:

- exercise_ws/src/object_detection/src/object_detection_node.py
- exercise_ws/src/object_detection/include/object_detection/model.py

Symptom: Segmentation is good, but the Duckiebot goes really fast in erratic motion.

Resolution: Change the speed, turn_speed, K and D parameters in the lane controller node until the Duckie becomes well-behaved.

Symptom: Duckiebot drives fine, but looses control and starts oscillating.

Resolution: Go slower! Lower "turn_speed" and "speed". A good idea is to start with low speeds (0.2 - 0.3) and then increase speed and gains (K, D) iteratively.

Symptom: Duckiebot oscillates even at slow speeds.

Resolution: Lower the K gain until it stop oscillating. Increase the "D" gain to get better performance in curves. Don't increase too much though, it will oscillate again at some point!

Symptom: Duckiebot has not fear and can crash into objects.

Resolution: Go slower! The perception system is limited by the network latency and CPU power of your machine. Running at 30 FPS on a Desktop allows going faster than running at 12 frames per seconds on a 3 years old laptop. A faster CPU will give the Duckiebot better reflexes. If you have time, do the inference on GPU, this would allow a laptop to process frames much faster, and give better reaction times to the Duckiebot.

## Demo failure demonstration {#demo-Harder_Better_Faster_Stronger-failure}

Here is a video of the Duckiebot going way too fast. It will eventually crash, just a matter of time!

<figure id="example-embed3">
    <figcaption>A Duckie driving too fast!</figcaption>
    <dtvideo src="vimeo:494207868"/>
</figure>
