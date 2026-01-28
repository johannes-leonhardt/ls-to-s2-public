import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
from pathlib import Path

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.dataset import TranslationDataset
from Models.diffusion import DiffusionModel_Concat, DiffusionModel_CBN

## Preliminaries
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])

## Settings
run_name = "Palette_2025-12-10_17-39-54"
settings = {
    "batch_size": 256,
    "n_channels": 4,
    "n_timesteps": 500,
    "use_panchromatic": True,
    "conditioning_method": "Concat", # Concat or CBN; set to Concat for Palette config
    "w": 0, # set to 0 for Palette config
    "n_samples": 1
}

## Load best model
model_path = os.path.join(models_root, run_name, "checkpoint.pt") 
if settings["conditioning_method"] == "Concat":
    diffusion_class = DiffusionModel_Concat
elif settings["conditioning_method"] == "CBN":
    diffusion_class = DiffusionModel_CBN
diffusion_model = diffusion_class(
    n_channels = settings["n_channels"],
    n_conditions = settings["n_channels"] + 2 if settings["use_panchromatic"] else settings["n_channels"] + 1, # extra channel for timestep
    n_timesteps = settings["n_timesteps"],
    scheduler_params = (1e-4, 0.02),
    # scheduler_params = 0.008
).to(device)
diffusion_model.load_state_dict(torch.load(model_path, map_location=device))
diffusion_model.eval()

## Test regions
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
test_regions = regions[regions.split == "test"]

for country, region in zip(test_regions.country, test_regions.region):

    l8_trans_path = os.path.join(data_root, "Images", "Landsat8-Translated", f"{run_name}_{settings['w']}", "2018", country)
    try:
        os.makedirs(l8_trans_path)
    except FileExistsError:
        pass

    test_ds = TranslationDataset(data_root, country, region)
    test_dl = DataLoader(test_ds, batch_size=settings["batch_size"], drop_last=False)

    for _, l8, l8_pan, filenames in tqdm(test_dl, desc=f"Processing {country}, {region}"):

        l8 = l8.to(device)
        if settings["use_panchromatic"]:
            l8 = torch.cat((l8, l8_pan.to(device)), dim=1)

        with torch.inference_mode():
            ls_translated = diffusion_model.sample(l8, w=settings["w"], n_samples=settings["n_samples"])

        # Save
        for i in range(ls_translated.shape[0]):
            torch.save(ls_translated[i].cpu().clone(), os.path.join(l8_trans_path, filenames[i]))