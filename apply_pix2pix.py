import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
from pathlib import Path

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models.dataset import TranslationDataset as Dataset
from Models.pix2pix import Generator

## Preliminaries
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])

## Settings
run_name = "Pix2Pix_2025-10-20_15-31-45"
settings = {
    "batch_size": 256,
    "in_channels": 5,
    "out_channels": 4,
    "use_pan": True
}

## Load best model
model_path = os.path.join(models_root, run_name, "generator_checkpoint.pt")
generator = Generator(n_in=settings["in_channels"], n_out=settings["out_channels"]).to(device)
generator.load_state_dict(torch.load(model_path, map_location=device))
generator.eval()

## Test regions
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
test_regions = regions[regions.split == "test"]

for country, region in zip(test_regions.country, test_regions.region):

    l8_trans_path = os.path.join(
        data_root, "Images", "Landsat8-Translated", f"{run_name}_final", "2018", country
    )
    os.makedirs(l8_trans_path, exist_ok=True)

    test_ds = Dataset(data_root, country, region)
    test_dl = DataLoader(test_ds, batch_size=settings["batch_size"], drop_last=False)

    for s2, l8, l8_pan, filenames in tqdm(test_dl, desc=f"Processing {country}, {region}"):

        s2 = s2.to(device)
        l8_pan = l8_pan.to(device) if settings["use_pan"] else None

        with torch.no_grad():
            fake_y = generator(s2, l8_pan)

        ## Save results
        for i in range(fake_y.shape[0]):
            torch.save(fake_y[i].cpu().clone(), os.path.join(l8_trans_path, filenames[i]))