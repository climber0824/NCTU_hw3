import torch.nn as nn
import torch.nn.functional as F


class InterpolateModule(nn.Module):
    def __init__(self, *args, **kwdargs):
        super().__init__()

        self.args = args
        self.kwdargs = kwdargs

    def forward(self, x):
        return F.interpolate(x, *self.args, **self.kwdargs)
