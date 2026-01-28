import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
import copy
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm 

from Models.dataset import TranslationDataset
from Models.diffusion import DiffusionModel_Concat, DiffusionModel_CBN
from Models.ema import EMA

## Preliminaries
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])
run_name = "Palette_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(models_root, run_name))

## Settings
settings = {
    "batch_size": 64,
    "n_channels": 4,
    "learning_rate": 1e-4,
    "n_epochs": 500,
    "n_timesteps": 500,
    "use_panchromatic": True,
    "conditioning_method": "Concat", # Concat or CBN; set to Concat for Palette config
    "p_uncond": 0 # set to 0 for Palette config
}
with open(os.path.join(models_root, run_name, "settings.yml"), "w") as settings_file: 
    yaml.dump(settings, settings_file, default_flow_style=False)

## Data stuff
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
train_regions = regions[regions.split == "train"]
val_regions = regions[regions.split == "val"]

train_ds = []
for country, region in zip(train_regions.country, train_regions.region):
    train_ds.append(TranslationDataset(data_root, country, region))
train_ds = ConcatDataset(train_ds)
train_loader = DataLoader(train_ds, shuffle=True, batch_size=settings["batch_size"])

val_ds = []
for country, region in zip(val_regions.country, val_regions.region):
    val_ds.append(TranslationDataset(data_root, country, region))
val_ds = ConcatDataset(val_ds)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=settings["batch_size"])

## Unet Model
if settings["conditioning_method"] == "Concat":
    diffusion_class = DiffusionModel_Concat
elif settings["conditioning_method"] == "CBN":
    diffusion_class = DiffusionModel_CBN
diffusion_model = diffusion_class(
    n_channels = settings["n_channels"],
    n_conditions = settings["n_channels"] + 2 if settings["use_panchromatic"] else settings["n_channels"] + 1, # extra channel for timestep
    n_timesteps = settings["n_timesteps"],
    scheduler_params = (1e-4, 0.02)
).to(device)
ema = EMA(diffusion_model)
optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=settings["learning_rate"])
lr_lambda = lambda epoch: max((settings["n_epochs"] - epoch) / settings["n_epochs"], 0.1)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

## Training

train_loss_tracker, val_loss_tracker = [], []
min_val_loss = np.inf

for epoch in range(settings["n_epochs"]):
    
    # Training loop
    diffusion_model.train()
    train_loss = 0
    for s2, l8, l8_pan, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{settings['n_epochs']}"):
        s2, l8 = s2.to(device), l8.to(device)
        if settings["use_panchromatic"]:
            l8 = torch.cat((l8, l8_pan.to(device)), dim=1)
        optimizer.zero_grad()
        loss = diffusion_model(s2, l8, settings["p_uncond"])
        loss.backward()
        optimizer.step()
        ema.update()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    print(f"Epoch {epoch+1}/{settings['n_epochs']}] Training Loss: {train_loss:.4f}")
    train_loss_tracker.append(train_loss)

    # Validation loop
    diffusion_model.eval()
    ema_model = copy.deepcopy(diffusion_model)
    ema.copy_to(ema_model)
    ema_model.eval()
    val_loss = 0
    with torch.inference_mode():
        for s2, l8, l8_pan, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{settings['n_epochs']} - Validation"):
            s2, l8 = s2.to(device), l8.to(device)
            if settings["use_panchromatic"]:
                l8 = torch.cat((l8, l8_pan.to(device)), dim=1)
            # loss = diffusion_model(s2, l8, settings["p_uncond"])
            loss = ema_model(s2, l8, settings["p_uncond"])
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"           Validation Loss: {val_loss:.4f}")
    val_loss_tracker.append(val_loss)
    scheduler.step()

    plt.figure()
    plt.plot(np.arange(0, epoch+1), train_loss_tracker, label="Training Loss")
    plt.plot(np.arange(0, epoch+1), val_loss_tracker, label="Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(models_root, run_name, "loss_curve_diffusion.png"))
    plt.close()

    if val_loss < min_val_loss:
        # torch.save(diffusion_model.state_dict(), os.path.join(models_root, run_name, f"checkpoint.pt"))
        torch.save(ema_model.state_dict(), os.path.join(models_root, run_name, f"checkpoint.pt"))
        min_val_loss = val_loss