import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from Models.diffusion_backbones import UNet, UNet_CBN

def _dm_schedules(betas, T):

    beta1, beta2 = betas
    beta_t = (beta2 - beta1) * torch.arange(0, T, dtype=torch.float32) / T + beta1
    alpha_t = 1 - beta_t
    alphabar_t = torch.cumsum(torch.log(alpha_t), dim=0).exp()

    return {
        "beta_t": beta_t,
        "alpha_t": alpha_t,
        "alphabar_t": alphabar_t
    }

class DiffusionModel_Concat(nn.Module):
    
    def __init__(self, n_channels, n_conditions, n_timesteps, scheduler_params):

        super().__init__()
        
        self.n_channels = n_channels
        self.n_T = n_timesteps
        self.nn_model = UNet(n_in=n_channels+n_conditions, n_out=n_channels, depth=3)
        for k, v in _dm_schedules(scheduler_params, n_timesteps).items():
            self.register_buffer(k, v)

    def prepare_condition(self, c, t, p_uncond=0):

        # Replace random conditions with null tokens for CFG
        mask = torch.bernoulli((1 - p_uncond) * torch.ones((c.shape[0]))).to(c.device)
        c = mask[:, None, None, None] * c
        # Concatentae condition with timestep
        t = t.float() / self.n_T
        ct = torch.cat((c, t[:, None, None, None].expand(c.shape[0], -1, c.shape[2], c.shape[3])), dim=1)

        return ct

    def forward(self, x, c, p_uncond=0):
        
        device = c.device
        t = torch.randint(0, self.n_T, (x.shape[0],)).to(device) # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x).to(device)  # eps ~ N(0, 1)
        x_t = (
            torch.sqrt(self.alphabar_t[t, None, None, None]) * x
            + torch.sqrt(1 - self.alphabar_t[t, None, None, None]) * noise
        )
        ct_t = self.prepare_condition(c, t, p_uncond)
        xct_t = torch.cat((x_t, ct_t), dim=1)
        out = self.nn_model(xct_t)
        loss = F.mse_loss(noise, out)
        
        return loss
    
    @torch.inference_mode()
    def sample(self, c, w=0, n_samples=1):

        device = c.device
        n_sample = c.shape[0]
        img_size = c.shape[2:]

        xs = []
        for _ in range(n_samples):
            x_i = torch.randn(n_sample, self.n_channels, *img_size).to(device)
            n_T_seq = reversed(range(self.n_T))
            for i in tqdm(n_T_seq):
                t = torch.tensor([i]).repeat(n_sample).to(device)
                ct_cond = self.prepare_condition(c, t, p_uncond=0)
                xct_cond = torch.cat((x_i, ct_cond), dim=1)
                eps_cond = self.nn_model(xct_cond)
                if w != 0:
                    ct_uncond = self.prepare_condition(c, t, p_uncond=1)
                    xct_uncond = torch.cat((x_i, ct_uncond), dim=1)
                    eps_uncond = self.nn_model(xct_uncond)
                    eps = (1 + w) * eps_cond - w * eps_uncond
                else:
                    eps = eps_cond
                
                # DDPM
                z = torch.randn(n_sample, self.n_channels, *img_size).to(device) if i > 0 else 0
                x_i = (
                    (1 / torch.sqrt(self.alpha_t[i]))
                    * (x_i - ((1 - self.alpha_t[i]) / (torch.sqrt(1 - self.alphabar_t[i]))) * eps)
                    + torch.sqrt(self.beta_t[i]) * z
                )

            xs.append(x_i)

        x = torch.mean(torch.stack(xs, dim=0), dim=0) if n_samples > 1 else xs[0]

        return x
    
class DiffusionModel_CBN(nn.Module):
    
    def __init__(self, n_channels, n_conditions, n_timesteps, scheduler_params):

        super().__init__()
        
        self.n_channels = n_channels
        self.n_T = n_timesteps
        self.nn_model = UNet_CBN(n_in=n_channels, n_out=n_channels, n_conditions=n_conditions, depth=3)
        for k, v in _dm_schedules(scheduler_params, n_timesteps).items():
            self.register_buffer(k, v)

    def prepare_condition(self, c, t, p_uncond=0):

        # Replace random conditions with null tokens for CFG
        mask = torch.bernoulli((1 - p_uncond) * torch.ones((c.shape[0]))).to(c.device)
        c = mask[:, None, None, None] * c
        # Concatentae condition with timestep
        t = t.float() / self.n_T
        ct = torch.cat((c, t[:, None, None, None].expand(c.shape[0], -1, c.shape[2], c.shape[3])), dim=1)

        return ct

    def forward(self, x, c, p_uncond=0):
        
        device = c.device
        t = torch.randint(0, self.n_T, (x.shape[0],)).to(device) # t ~ Uniform(0, n_T)
        noise = torch.randn_like(x).to(device)  # eps ~ N(0, 1)
        x_t = (
            torch.sqrt(self.alphabar_t[t, None, None, None]) * x
            + torch.sqrt(1 - self.alphabar_t[t, None, None, None]) * noise
        )
        c_t = self.prepare_condition(c, t, p_uncond)
        out = self.nn_model(x_t, c_t)
        loss = F.mse_loss(noise, out)
        
        return loss
    
    @torch.inference_mode()
    def sample(self, c, w=0, n_samples=1):

        device = c.device
        n_sample = c.shape[0]
        img_size = c.shape[2:]

        xs = []
        for _ in range(n_samples):
            x_i = torch.randn(n_sample, self.n_channels, *img_size).to(device)
            n_T_seq = reversed(range(self.n_T))
            for i in tqdm(n_T_seq):
                t = torch.tensor([i]).repeat(n_sample).to(device)
                ct_cond = self.prepare_condition(c, t, p_uncond=0)
                xct_cond = torch.cat((x_i, ct_cond), dim=1)
                eps_cond = self.nn_model(xct_cond)
                if w != 0:
                    ct_uncond = self.prepare_condition(c, t, p_uncond=1)
                    xct_uncond = torch.cat((x_i, ct_uncond), dim=1)
                    eps_uncond = self.nn_model(xct_uncond)
                    eps = (1 + w) * eps_cond - w * eps_uncond
                else:
                    eps = eps_cond
                
                # DDPM
                z = torch.randn(n_sample, self.n_channels, *img_size).to(device) if i > 0 else 0
                x_i = (
                    (1 / torch.sqrt(self.alpha_t[i]))
                    * (x_i - ((1 - self.alpha_t[i]) / (torch.sqrt(1 - self.alphabar_t[i]))) * eps)
                    + torch.sqrt(self.beta_t[i]) * z
                )

            xs.append(x_i)

        x = torch.mean(torch.stack(xs, dim=0), dim=0) if n_samples > 1 else xs[0]

        return x