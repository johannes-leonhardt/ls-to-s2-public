import os
import yaml
from pathlib import Path

import numpy as np
import geopandas as gpd
from skimage.exposure import match_histograms
import torch
from tqdm import tqdm

from Models.dataset import TranslationDataset as Dataset

## Load local config

with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])

## Apply Brovey pansharpening to test set

regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
test_regions = regions[regions.split == "test"]

for country, region in zip(test_regions.country, test_regions.region):
    
    l8_trans_path = os.path.join(data_root, "Images", "Landsat8-Translated", "Brovey_Linear", "2018", country)
    try:
        os.makedirs(l8_trans_path)
    except FileExistsError:
        pass

    test_ds = Dataset(data_root, country, region)

    for _, ls, ls_pan, filename in tqdm(test_ds, desc=f"Processing {country}, {region}"):

        # Spectral adjustment

        # # Model: S2 = L8 (pansharpening only)
        # # Do nothing

        # # Model: S2 = a * L8 (scale-only regression)
        # a = torch.tensor([1.1723421, 1.0422608, 1.0277764, 0.92718023])
        # ls = a[:, None, None] * ls

        # # Model: S2 = a * L8 + b (linear regression)
        # a = torch.tensor([1.0384278, 0.7610578, 0.8206611, 0.6978625])
        # b = torch.tensor([0.014070325, 0.058377877, 0.028135397, 0.076993])
        # ls = a[:, None, None] * ls + b[:, None, None]

        # To numpy
        ls = ls.permute(1,2,0).numpy()
        ls_pan = ls_pan.permute(1,2,0).numpy()

        # Spatial adjustment with Brovey pansharpening
        ls_mean = np.mean(ls, axis=2, keepdims=True) + 1e-8
        ls_pan = match_histograms(ls_pan, ls_mean)
        ls_sharpened = ls * (ls_pan / ls_mean)
        ls_sharpened = np.clip(ls_sharpened, 0, 1)

        # Back to torch and save result
        ls_sharpened = torch.tensor(ls_sharpened).permute(2,0,1)
        torch.save(ls_sharpened, os.path.join(l8_trans_path, filename))
