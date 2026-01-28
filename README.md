# Diffusion-based Landsat 8 to Sentinel-2 Translation for Consistent Land Cover Classification

This repository contains the code for our paper "Diffusion-based Landsat 8 to Sentinel-2 Translation for Consistent Land Cover Classification", currently under review.

For now, the data can be accessed here: https://drive.google.com/drive/folders/13Pk6pIg0QAj0y9ERzmQAF9RBRPuEbzM7?usp=sharing.

## Instructions

Download and unpack the raw image data (`Sentinel-2.zip`, `Landsat8.zip`, and `Landsat8-Panchro.zip`), the regions file (`lucas_regions.gpkg`), as well as the weights of the pretrained land cover classification model (`ClassifierDeepLabV3_2025-10-13_22-08-36`) using the provided link. 

Specify the directories where the data and the checkpoints are stored in the `local_config.yml` file.

After installing the necessary dependencies, you can run the training, application, and evaluation scripts.
