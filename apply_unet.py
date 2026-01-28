import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import yaml
from pathlib import Path

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.dataset import TranslationDataset as Dataset
from Models.unet import UNet 

## Preliminaries
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])

## Settings
run_name = "UNet_L1_2026-01-28_09-54-19"
settings = {
    "batch_size": 256,
    "in_channels": 5,
    "out_channels": 4,
    "use_pan": True,
    "model": {
        "depth": 5,
        "growth_factor": 6,
        "spatial_attention": "None",
        "up_mode": "upconv",
        "ca_layer": False
    }
}

## Load best model
model_path = os.path.join(models_root, run_name, "checkpoint.pt") 
aunet = UNet(
    in_channels=settings["in_channels"],          
    out_channels=settings["out_channels"],         
    depth=settings["model"]["depth"],                
    growth_factor=settings["model"]["growth_factor"],      
    spatial_attention=settings["model"]["spatial_attention"],  
    up_mode=settings["model"]["up_mode"],              
    ca_layer=settings["model"]["ca_layer"]           
).to(device)
aunet.load_state_dict(torch.load(model_path, map_location=device))
aunet.eval()

## Test regions
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
test_regions = regions[regions.split == "test"]

for country, region in zip(test_regions.country, test_regions.region):

    l8_trans_path = os.path.join(data_root, "Images", "Landsat8-Translated", run_name, "2018", country)
    try:
        os.makedirs(l8_trans_path)
    except FileExistsError:
        pass

    test_ds = Dataset(data_root, country, region)
    test_dl = DataLoader(test_ds, batch_size=settings["batch_size"], drop_last=False)

    for _, l8, l8_pan, filenames in tqdm(test_dl, desc=f"Processing {country}, {region}"):

        l8 = l8.to(device)
        l8_pan = l8_pan.to(device)

        with torch.no_grad():
            if settings["use_pan"]:
                l8_translated = aunet(MS = l8, PAN = l8_pan)
            else:   
                l8_translated = aunet(MS = l8, PAN = None)

        # Save
        for i in range(l8_translated.shape[0]):
            torch.save(l8_translated[i].cpu().clone(), os.path.join(l8_trans_path, filenames[i]))