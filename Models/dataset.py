import os

import numpy as np
import torch

class TranslationDataset(torch.utils.data.Dataset):

    mins = torch.tensor([0.0, 0.0, 0.0, 0.0]) / 10000
    maxs = torch.tensor([3000.0, 3000.0, 3000.0, 7000.0]) / 10000

    def __init__(self, root, country, region):
        
        super().__init__()

        self.root = root
        self.country = country
        self.region = region
        self.s2_path = os.path.join(self.root, "Images", "Sentinel-2", "2018", self.country)
        self.l8_path = os.path.join(self.root, "Images", "Landsat8", "2018", self.country)
        self.l8_pan_path = os.path.join(self.root, "Images", "Landsat8-Panchro", "2018", self.country)
        self.filenames = sorted([filename for filename in os.listdir(self.s2_path) if (self.region in filename and filename.endswith(".pt"))])

    def __len__(self):

        return len(self.filenames)
    
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        s2 = torch.load(os.path.join(self.s2_path, filename))
        s2 = normalize(s2, self.mins, self.maxs)
        l8 = torch.load(os.path.join(self.l8_path, filename))
        l8 = normalize(l8, self.mins, self.maxs)
        l8_pan = torch.load(os.path.join(self.l8_pan_path, filename))
        
        return s2, l8, l8_pan, filename

class ClassificationDatasetEsa(torch.utils.data.Dataset):

    mins = torch.tensor([0.0, 0.0, 0.0, 0.0]) / 10000
    maxs = torch.tensor([3000.0, 3000.0, 3000.0, 7000.0]) / 10000

    def __init__(self, root, country, region):
        
        super().__init__()

        self.root = root
        self.country = country
        self.region = region
        self.s2_path = os.path.join(self.root, "Images", "Sentinel-2", "2018", self.country)
        self.lc_path = os.path.join(self.root, "Land Cover", "ESAWC", "2020", self.country)
        self.filenames = sorted([filename for filename in os.listdir(self.lc_path) if (self.region in filename and filename.endswith(".pt"))])

    def __len__(self):

        return len(self.filenames)
    
    def __getitem__(self, idx):

        filename = self.filenames[idx]
        s2 = torch.load(os.path.join(self.s2_path, filename))
        s2 = normalize(s2, self.mins, self.maxs)
        lc = torch.load(os.path.join(self.lc_path, filename)).long()
        lc = remap_esawc(lc)
        
        return s2, lc, torch.tensor([]), filename
    
def normalize(img, mins, maxs):

    return (img - mins[:, None, None]) / (maxs[:, None, None] - mins[:, None, None])

def remap_esawc(lc):

    lc[lc == 10] = 2 # Tree Cover
    lc[lc == 20] = 4 # Shrubland
    lc[lc == 30] = 3 # Grassland
    lc[lc == 40] = 1 # Cropland
    lc[lc == 50] = 0 # Built-up
    lc[lc == 60] = 5 # Bare/sparse vegetation
    lc[lc == 70] = 8 # Snow and Ice
    lc[lc == 80] = 6 # Permanent water bodies
    lc[lc == 90] = 7 # Herbaceous wetland
    lc[lc == 95] = 7 # Mangroves
    lc[lc == 100] = 5 # Moss and lichen

    return lc

def esawc_to_image(lc):

    colormap = [
        (0, np.array([192, 57, 43]) / 255), # Built-up
        (1, np.array([244, 208, 63]) / 255), # Cropland
        (2, np.array([11, 83, 69]) / 255), # Tree cover
        (3, np.array([121, 193, 113]) / 255), # Grassland
        (4, np.array([153, 102, 51]) / 255), # Shrubland
        (5, np.array([131, 145, 146]) / 255), # Bare / sparse vegetation
        (6, np.array([33, 97, 140]) / 255), # Permanent water bodies
        (7, np.array([174, 214, 241]) / 255), # Herbaceous wetland
        (8, np.array([247, 254, 255]) / 255), # Snow and Ice
    ]

    lc_vis = np.zeros((lc.shape[0], lc.shape[1], 3))
    for i in range(len(colormap)):
        lc_vis[lc == colormap[i][0]] = colormap[i][1]
    
    return lc_vis