import os
os.environ["CUDA_VISIBLE_DEVICES"]="4"
import yaml
from pathlib import Path

import geopandas as gpd
import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsoluteError, MeanSquaredError
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.classification import Accuracy, F1Score, JaccardIndex
from tqdm import tqdm

from Models.dataset import TranslationDataset
from Models.classifiers import *

## Preliminaries

if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(f"Using device: {device}")

with open("local_config.yml", "r") as f:
    config = yaml.safe_load(f)
data_root = Path(config["data_root"])
models_root = Path(config["models_root"])

## Classification model

n_classes = 9 # 8 for LUCAS; 9 for ESA
classifier = DeepLabV3_SMP(4, n_classes, "resnet34").to(device)
run_name = "ClassifierDeepLabV3_2025-10-13_22-08-36"
classifier.load_state_dict(torch.load(os.path.join(models_root, run_name, "checkpoint.pt")))
classifier.eval()

## Data

regions = gpd.read_file(os.path.join(data_root, 'lucas_regions.gpkg'))
val_regions = regions[regions.split == "val"]
test_regions = regions[regions.split == "test"]
l8_trans_paths = [ 
    os.path.join(data_root, "Images", "Landsat8-Translated", "UNet_L1_2026-01-28_09-54-19", "2018"),
] # Empty list -> test Sentinel-2; one element -> test individual method; multiple elements -> test average of methods

## Metrics

# Translation metrics
mae = MeanAbsoluteError().to(device)
rmse = MeanSquaredError(squared=False).to(device)
ssim = StructuralSimilarityIndexMeasure(data_range=(0,1)).to(device)

# Image quality metrics
is_rgb = InceptionScore(normalize=True).to(device)
is_nirrg = InceptionScore(normalize=True).to(device)
fid_rgb = FrechetInceptionDistance(normalize=True).to(device)
fid_nirrg = FrechetInceptionDistance(normalize=True).to(device)
kid_rgb = KernelInceptionDistance(normalize=True).to(device)
kid_nirrg = KernelInceptionDistance(normalize=True).to(device)

# Land cover classification consistency metrics
acc = Accuracy("multiclass", num_classes=n_classes, average='micro').to(device)
f1 = F1Score("multiclass", num_classes=n_classes, average='macro').to(device)
iou = JaccardIndex("multiclass", num_classes=n_classes, average="macro").to(device)

for country, region in zip(val_regions.country, val_regions.region):

    val_ds = TranslationDataset(data_root, country, region)
    val_dl = DataLoader(val_ds, batch_size=256, drop_last=False)
    
    for s2, _, _, _ in tqdm(val_dl, desc=f"Initializing FID and KID with real data from {country}, {region}"):

        s2 = s2.to(device)

        fid_rgb.update(s2[:,[2,1,0]], real=True)
        fid_nirrg.update(s2[:,[3,2,1]], real=True)
        kid_rgb.update(s2[:,[2,1,0]], real=True)
        kid_nirrg.update(s2[:,[3,2,1]], real=True)

for country, region in zip(test_regions.country, test_regions.region):

    test_ds = TranslationDataset(data_root, country, region)
    test_dl = DataLoader(test_ds, batch_size=256, drop_last=False)
    
    for s2, _, _, filenames in tqdm(test_dl, desc=f"Evaluating {country}, {region}"):
        
        # Load translated data
        if len(l8_trans_paths) == 0: # Compare to real S2 data
            ls_trans = s2.detach().clone()
        elif len(l8_trans_paths) == 1: # Compare to output of single model
            ls_trans = torch.stack([torch.load(os.path.join(l8_trans_paths[0], country, filename)) for filename in filenames])
        else:
            ls_trans = []
            for l8_trans_path in l8_trans_paths: # Compare to averaged outputs from multiple models
                ls_trans_i = torch.stack([torch.load(os.path.join(l8_trans_path, country, filename)) for filename in filenames])
                ls_trans.append(ls_trans_i)
            ls_trans = torch.mean(torch.stack(ls_trans, dim=0), dim=0)
            
        s2[torch.isnan(s2)] = 0
        s2 = torch.clip(s2, 0, 1)
        ls_trans[torch.isnan(ls_trans)] = 0
        ls_trans = torch.clip(ls_trans, 0, 1)

        # Send to device
        s2, ls_trans = s2.to(device), ls_trans.to(device)

        # How well does the translated Landsat imagery match the Sentinel-2 reference?
        mae.update(s2, ls_trans)
        rmse.update(s2, ls_trans)
        ssim.update(s2, ls_trans)

        # How realistic is the translated Landsat imagery?
        is_rgb.update(ls_trans[:, [2,1,0]])
        is_nirrg.update(ls_trans[:, [3,2,1]])
        fid_rgb.update(ls_trans[:,[2,1,0]], real=False)
        fid_nirrg.update(ls_trans[:,[3,2,1]], real=False)
        kid_rgb.update(ls_trans[:,[2,1,0]], real=False)
        kid_nirrg.update(ls_trans[:,[3,2,1]], real=False)

        with torch.inference_mode():
            lc_hat_s2 = torch.argmax(classifier(s2), dim=1)
            lc_hat_l8 = torch.argmax(classifier(ls_trans), dim=1)
        acc.update(lc_hat_l8, lc_hat_s2)
        f1.update(lc_hat_l8, lc_hat_s2)
        iou.update(lc_hat_l8, lc_hat_s2)

mae_res = mae.compute().item()
rmse_res = rmse.compute().item()
ssim_res = ssim.compute().item()
is_res = (is_rgb.compute()[0].item() + is_nirrg.compute()[0].item()) / 2
fid_res = (fid_rgb.compute().item() + fid_nirrg.compute().item()) / 2
kid_res = (kid_rgb.compute()[0].item() + kid_nirrg.compute()[0].item()) / 2
acc_res = acc.compute().item()
f1_res = f1.compute().item()
iou_res = iou.compute().item()

print(f"Image translation metrics: MAE: {mae_res}, RMSE: {rmse_res}, SSIM: {ssim_res}.")
print(f"Image quality metrics: IS: {is_res}, FID: {fid_res}, KID: {kid_res}.")
print(f"Land cover classification consistency metrics: Acc.: {acc_res}, F1: {f1_res}, IoU: {iou_res}.")