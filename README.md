# implicit-recon

## Introduction
The implicit-recon repository contains functionality for the reconstruction of shapes using
implicit representations from images or voxel grids with intensity values.
This repository is structured as follows:

```
implicit-recon
├── example_output
│   ├── logs
│   ├── predictions_chd_ct
│   ├── saved_models
│   └── stl
├── example_runs
└── src
    ├── metrics
    ├── models
    └── utilities
```

## Requirements
The code has the following dependencies:
`numpy`, `pillow`, `scikit-image`, `tqdm`, `torch`, `torchvision`, 
`torchsummary`, `tensorboard`, `numpy-stl`

These can be installed (in your virtual environment) by running

    pip install -r requirements.txt

## Creating the data
The Congenital Heart Disease CT data-set was kindly made available to us by 
Dr. Xiaowei Xu from the Department of Computer Science and Engineering,
University of Notre Dame.

Running the Python script `src/utilities/compute_distance_fields.py`
splits this data-set into training, validation and test data (and optionally
computes distance fields) with input images in `images_1_slice_512` and
segmented binary masks (labels) in `BP_masks_512`. This results in the 
following directory tree, rooted at the CHD_orig directory:

```
CHD_orig
└── output_512
    ├── test
    │   ├── BP_masks_512
    │   └── images_1_slice_512
    ├── train
    │   ├── BP_masks_512
    │   └── images_1_slice_512
    └── val
        ├── BP_masks_512
        └── images_1_slice_512
```


## Scripts
The `example_runs` directory contains several Python scripts that can either 
be called from the command line (straight from their `example_runs` 
directory), or imported in another Python script. Documentation for precise
usage is obtained by running the script with the `--help` argument.
* `train.py`: Train a VGGTrunc1, VGGTrunc2, or UNetImplicit neural network.
* `infer.py`: Inference of a trained VGGTrunc1, VGGTrunc2, or UNetImplicit
  neural network applied to image files in a directory.
* `parameter_study.py`: Import functionality from `train.py` to train for 
  several UNetImplicit architectures with different network parameters.

In addition, the `src/utilities` directory contains:
* `png_to_mesh.py`: Convert alphabetically ordered PNG files in a directory
  into an STL file.


## Example workflow
Suppose our dataset is split the directory tree
shown above, for which the root of the above directory tree is
`BASE_DATA_DIR="CHD_orig/"` and its subdirectory is
`SUB_DATA_DIR="output_512"`. As above, input images are stored in the
subdirectory `IMAGE_DIR="images_1_slice_512"`, while binary masks (labels)
are stored in the subdirectory `MASK_DIR="BP_masks_512"`. The base directory
for the output is the repository root `BASE_OUTPUT_DIR="implicit_recon"`.

Let's examine an example workflow for:
* training a `NETWORK="UNetImplicit"`,
* storing and loading the model weights in `MODEL_FILE="../example_output/saved_models/weights_chd_ct.pth"`
* applying inference to the directory `IMAGE_INFERENCE_DIR=$BASE_DATA_DIR$SUB_DATA_DIR"/test/images_1_slice_512"`,
* storing the inferred predictions in the directory `PRED_DIR="../example_output/predictions_chd_ct"`, and
* and converting to an STL file in `STL_DIR="../example_output/stl"`. 

With the above definitions, the following commands should then be run from the `example_runs` directory:

```
#!/bin/bash
python train.py --base_data_dir $BASE_DATA_DIR --sub_data_dir $SUB_DATA_DIR --base_output_dir $BASE_OUTPUT_DIR \
            --image_dir $IMAGE_DIR --mask_dir $MASK_DIR --model_in $MODEL_FILE --model_out $MODEL_FILE \
            --network $NETWORK --n_epochs 10
python infer.py --model_path $MODEL_FILE --image_paths $IMAGE_INFERENCE_DIR --output_path $PRED_DIR
python ../src/utilities/png_to_mesh.py --init_slice 0 --final_slice 134 --aspect_ratio 2 \
   --in_path $PRED_DIR --out_path $STL_DIR --out_fname model_chd_ct.stl --out_res 0 0 0
```

