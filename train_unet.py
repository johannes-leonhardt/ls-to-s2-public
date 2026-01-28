import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import yaml
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm 
from torchmetrics.functional.image import structural_similarity_index_measure

from Models.dataset import TranslationDataset
from Models.unet import UNet


## Preliminaries
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])
run_name = "UNet_L1_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(models_root, run_name))

## Loss
def get_loss_function(settings):
    name = settings["loss_function"].lower()
    alpha = settings.get("loss_alpha", 0.8)  # Gewichtung zw. L1/L2 und SSIM
    if name == "l1":
        return F.l1_loss
    elif name == "mse":
        return F.mse_loss
    elif name == "l1+ssim":
        def loss_fn(pred, target):
            l1 = F.l1_loss(pred, target)
            ssim_loss = 1 - structural_similarity_index_measure(pred, target, data_range=(0,1))
            return alpha * l1 + (1 - alpha) * ssim_loss
        return loss_fn
    elif name == "mse+ssim":
        def loss_fn(pred, target):
            mse = F.mse_loss(pred, target)
            ssim_loss = 1 - structural_similarity_index_measure(pred, target, data_range=(0,1))
            return alpha * mse + (1 - alpha) * ssim_loss
        return loss_fn
    else:
        raise ValueError(f"Unknown loss function: {name}")

## Settings
settings = {
    "batch_size": 64,
    "n_channels": 5,
    "output_channels": 4,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "loss_function": "l1",
    "use_pan": True,
    "model": {
        "depth": 5,
        "growth_factor": 6,
        "spatial_attention": "None",
        "up_mode": "upconv",
        "ca_layer": False,
    }
}
with open(os.path.join(models_root, run_name, "settings.yml"), "w") as settings_file: 
    yaml.dump(settings, settings_file, default_flow_style=False)

criterion = get_loss_function(settings)

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
aunet = UNet(
    in_channels = settings["n_channels"],
    out_channels = settings["output_channels"],
    depth = settings["model"]["depth"],
    growth_factor = settings["model"]["growth_factor"],
    spatial_attention = settings["model"]["spatial_attention"],
    up_mode = settings["model"]["up_mode"],
    ca_layer = settings["model"]["ca_layer"],
).to(device)
optimizer = torch.optim.AdamW(aunet.parameters(), lr=settings["learning_rate"])
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

## Training
train_loss_tracker, val_loss_tracker = [], []
min_val_loss = np.inf

for epoch in range(settings["num_epochs"]):
    
    # Training loop
    aunet.train()
    train_loss = 0
    for s2, l8, l8_pan, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{settings['num_epochs']}"):
        l8, s2, l8_pan = l8.to(device), s2.to(device), l8_pan.to(device)
        optimizer.zero_grad()
        if settings["use_pan"]:
            out = aunet(MS = l8, PAN = l8_pan)
        else:
            out = aunet(MS = l8, PAN = None)
        loss = criterion(out, s2)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{settings['num_epochs']}] Training Loss: {avg_loss:.4f}")
    train_loss_tracker.append(avg_loss)

    # Validation loop
    aunet.eval()
    val_loss = 0
    with torch.no_grad():
        for s2, l8, l8_pan, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{settings['num_epochs']} - Validation"):
            l8 = l8.to(device)
            s2 = s2.to(device)
            l8_pan = l8_pan.to(device)
            if settings["use_pan"]:
                out = aunet(MS = l8, PAN = l8_pan)
            else:
                out = aunet(MS = l8, PAN = None)
            loss = criterion(out, s2)#settings["loss_function"](out, s2)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    print(f"           Validation Loss: {val_loss:.4f}")
    val_loss_tracker.append(val_loss)
    sched.step(val_loss)

    plt.figure()
    plt.plot(np.arange(0,epoch+1), train_loss_tracker, label="Training Loss")
    plt.plot(np.arange(0,epoch+1), val_loss_tracker, label="Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(models_root, run_name, "loss_curve_unet.png"))
    plt.close()

    if val_loss < min_val_loss:
        torch.save(aunet.state_dict(), os.path.join(models_root, run_name, f"checkpoint.pt"))
        min_val_loss = val_loss