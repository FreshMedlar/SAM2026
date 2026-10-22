import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearNoBias(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LinearNoBias, self).__init__()
        self.w = nn.Linear(in_dim, out_dim, bias=False)
        self.ha = None
        self.hb = None

    def forward(self, x):
        self.ha = x
        if x.requires_grad:
            self.ha.retain_grad()
        out = self.w(x)
        self.hb = out
        if out.requires_grad:
            self.hb.retain_grad()
        return out

    def backproj_nobias(self, y):
        self.hb.grad = y
        x = torch.mm(y, self.w.weight)
        self.ha.grad = x
        return x

def mask_processing(x, mask_mode, x_masks):
    if mask_mode == 1:
        x_masks.append((F.relu(x) != 0).detach())
    elif mask_mode == 2:
        if len(x_masks) > 0:
            x = x * (x_masks[0].detach())
            x_masks.pop(0)
    return x

class SynapticNet(nn.Module):
    """
    4-layer feedforward or feedback net as defined in the original notebook:
    784 -> 200 -> 100 -> 200 -> 10
    """
    def __init__(self, in_d=784, out_d=10, nonlin=F.relu):
        super(SynapticNet, self).__init__()
        self.w = LinearNoBias(in_d, 200)
        self.w2 = LinearNoBias(200, 100)
        self.w3 = LinearNoBias(100, 200)
        self.u = LinearNoBias(200, out_d)
        self.nonlin = nonlin

    def get_linear_layers(self):
        return [self.w, self.w2, self.w3, self.u]

    def forward(self, x, x_masks=None, nonlin=None):
        if isinstance(x_masks, list):
            if len(x_masks) == 0:
                mask_mode = 1
            else:
                x_masks = x_masks.copy()
                mask_mode = 2
        else:
            mask_mode = 0

        used_nonlin = nonlin if nonlin is not None else self.nonlin

        x = x.view(x.shape[0], -1)

        x = self.w(x)
        x = mask_processing(x, mask_mode, x_masks)
        x = used_nonlin(x)

        x = self.w2(x)
        x = mask_processing(x, mask_mode, x_masks)
        x = used_nonlin(x)

        x = self.w3(x)
        x = mask_processing(x, mask_mode, x_masks)
        x = used_nonlin(x)

        x = self.u(x)
        return x

    def backproj(self, x, x_masks=None, nonlin=None):
        if x_masks:
            x_masks = list(reversed(x_masks.copy()))
            mask_mode = 2
        else:
            mask_mode = 0

        x = self.u.backproj_nobias(x)
        if nonlin:
            x = nonlin(x)
        if x_masks:
            x = mask_processing(x, mask_mode, x_masks)

        x = self.w3.backproj_nobias(x)
        if nonlin:
            x = nonlin(x)
        if x_masks:
            x = mask_processing(x, mask_mode, x_masks)

        x = self.w2.backproj_nobias(x)
        if nonlin:
            x = nonlin(x)
        if x_masks:
            x = mask_processing(x, mask_mode, x_masks)

        x = self.w.backproj_nobias(x)
        return x

class StandardBPNet(nn.Module):
    """
    Equivalent architecture trained with standard Autograd Backpropagation.
    """
    def __init__(self, in_d=784, out_d=10):
        super(StandardBPNet, self).__init__()
        self.w = nn.Linear(in_d, 200, bias=False)
        self.w2 = nn.Linear(200, 100, bias=False)
        self.w3 = nn.Linear(100, 200, bias=False)
        self.u = nn.Linear(200, out_d, bias=False)

    def get_linear_layers(self):
        return [self.w, self.w2, self.w3, self.u]

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.w(x))
        x = F.relu(self.w2(x))
        x = F.relu(self.w3(x))
        x = self.u(x)
        return x
