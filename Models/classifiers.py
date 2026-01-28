
import torch.nn as nn
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
    
class DeepLabV3_SMP(nn.Module):

    def __init__(self, n_in, n_out, encoder):

        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.encoder = encoder
        
        self.upscale = transforms.Resize((224, 224))
        self.network = smp.DeepLabV3Plus(encoder, encoder_weights=None, in_channels=n_in, classes=64)
        self.downscale = nn.AdaptiveAvgPool2d((64, 64))
        self.output_layer = nn.Conv2d(64, n_out, kernel_size=1)

    def forward(self, x):
        
        x = self.upscale(x)
        x = self.network(x)
        x = self.downscale(x)
        out = self.output_layer(x)

        return out
