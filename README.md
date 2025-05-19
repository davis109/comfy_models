# StableVITON Demo

This repository contains a demo implementation for StableVITON, a virtual try-on model that uses semantic correspondence learning with latent diffusion models to create realistic clothing try-on images.

## Overview

StableVITON is a state-of-the-art virtual try-on system. This demo script provides visualization tools to explore the dataset structure and see how the model inputs are organized before running the full model inference.

## Dataset Structure

The dataset is organized into `train` and `test` folders with the following structure:

```
train/
├── image/                 # Person images
├── cloth/                 # Clothing item images
├── cloth-mask/            # Binary masks for clothing items
├── agnostic-v3.2/         # Person images with clothing region masked out
├── agnostic-mask/         # Binary masks for agnostic regions
├── image-densepose/       # DensePose representations of person images
└── other folders...

test/
├── image/                 # Person images
├── cloth/                 # Clothing item images
├── cloth-mask/            # Binary masks for clothing items
└── other folders...
```

The `train_pairs.txt` and `test_pairs.txt` files contain mappings between person images and clothing items to try on.



### Options

- `--data_root`: Path to the data directory (train or test)
- `--pairs_file`: Path to the pairs file (train_pairs.txt or test_pairs.txt)
- `--output_dir`: Directory to save results
- `--num_samples`: Number of samples to visualize
- `--is_test`: Whether to use test dataset mode

## Results

The script creates visualizations of each sample pair, showing:
1. The person image
2. The clothing item to try on
3. The clothing mask
4. The agnostic image (person with clothing region removed)
5. The DensePose representation
6. A placeholder for the virtual try-on result


## Full Implementation

For the full implementation with model inference:

1. Download the model weights from the [HuggingFace repository](https://huggingface.co/rlawjdghek/StableVITON)
2. Follow the instructions in the original repository to set up the environment and run inference
3. Use the `inference.py` script in the StableVITON directory with the proper model weights

## Acknowledgements

The dataset and model architecture are from the original [StableVITON project](https://github.com/rlawjdghek/StableVITON). 
