import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import yaml
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

import numpy as np
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm 

from Models.dataset import TranslationDataset as Dataset
from Models.pix2pix import Generator, Discriminator

## Preliminaries
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])
run_name = "Pix2Pix_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join(models_root, run_name))

## Settings
settings = {
    "batch_size": 64,
    "in_channels": 5,
    "out_channels": 4,
    "use_pan": True,
    "learning_rate": 1e-4,
    "num_epochs": 500,
    "loss_function": "l1+gan",
    "lambda_l1": 100.0,
}
with open(os.path.join(models_root, run_name, "settings.yml"), "w") as settings_file: 
    yaml.dump(settings, settings_file, default_flow_style=False)

## Data stuff
regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
train_regions = regions[regions.split == "train"]
val_regions = regions[regions.split == "val"]

train_ds = []
for country, region in zip(train_regions.country, train_regions.region):
    train_ds.append(Dataset(data_root, country, region))
train_ds = ConcatDataset(train_ds)
train_loader = DataLoader(train_ds, shuffle=True, batch_size=settings["batch_size"])

val_ds = []
for country, region in zip(val_regions.country, val_regions.region):
    val_ds.append(Dataset(data_root, country, region))
val_ds = ConcatDataset(val_ds)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=settings["batch_size"])

## Pix2Pix model
generator = Generator(
    n_in=settings["in_channels"], n_out=settings["out_channels"]
).to(device)
discriminator = Discriminator(
    in_channels=settings["in_channels"] + settings["out_channels"]
).to(device)
bce_loss = nn.BCEWithLogitsLoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=settings["learning_rate"], betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=settings["learning_rate"], betas=(0.5, 0.999))

## Loss function
def gan_loss(pred, target_is_real):
    target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    return bce_loss(pred, target)

## Training 
train_loss_tracker = []
val_loss_tracker = []
min_val_loss = float("inf")

for epoch in range(settings["num_epochs"]):
    
    # Training loop
    generator.train()
    discriminator.train()
    train_loss = 0
    for s2, l8, l8_pan, filenames in tqdm(train_loader, desc=f"Epoch {epoch+1}/{settings['num_epochs']}"):
        x, y, pan = l8.to(device), s2.to(device), l8_pan.to(device)
        
        # Discriminator
        optimizer_D.zero_grad()
        fake_y = generator(x, pan)
        d_input = x if pan is None else torch.cat([x, pan], dim=1)
        pred_real = discriminator(d_input, y)
        pred_fake = discriminator(d_input, fake_y.detach())
        d_loss_real = gan_loss(pred_real, True)
        d_loss_fake = gan_loss(pred_fake, False)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        d_loss.backward()
        optimizer_D.step()

        # Generator
        optimizer_G.zero_grad()
        pred_fake = discriminator(d_input, fake_y)
        g_gan = gan_loss(pred_fake, True)
        g_l1 = F.l1_loss(fake_y, y)
        g_loss = g_gan + g_l1 * settings["lambda_l1"]
        g_loss.backward()
        optimizer_G.step()
        train_loss += g_l1.item()
    avg_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{settings['num_epochs']}] Training Loss: {avg_loss:.4f}")
    train_loss_tracker.append(avg_loss)

    # Validation loop
    generator.eval()
    with torch.no_grad():
        val_l1 = []
        for s2, l8, l8_pan, filenames in tqdm(val_loader, desc="Validation"):
            x, y, pan = l8.to(device), s2.to(device), l8_pan.to(device)
            fake_y = generator(x, pan)
            val_l1.append(F.l1_loss(fake_y, y).item())
        val_loss = np.mean(val_l1)
        val_loss_tracker.append(val_loss)
    print(f"           Validation Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < min_val_loss:
        torch.save(generator.state_dict(), os.path.join(models_root, run_name, "generator_checkpoint.pt"))
        torch.save(discriminator.state_dict(), os.path.join(models_root, run_name, "discriminator_checkpoint.pt"))
        min_val_loss = val_loss

    # Save loss curve
    plt.figure()
    plt.plot(np.arange(0,epoch+1), train_loss_tracker, label="Training Loss")
    plt.plot(np.arange(0,epoch+1), val_loss_tracker, label="Validation Loss")
    plt.legend()
    plt.savefig(os.path.join(models_root, run_name, "loss_curve_pix2pix.png"))
    plt.close()