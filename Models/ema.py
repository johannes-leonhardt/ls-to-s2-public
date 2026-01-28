import torch

class EMA:

    def __init__(self, model, beta=0.999):

        self.beta = beta
        self.model = model
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):

        for k, v in self.model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.beta).add_(v, alpha=1-self.beta)

    def copy_to(self, model):

        model.load_state_dict(self.shadow, strict=False)