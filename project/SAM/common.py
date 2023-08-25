# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pdb


# From https://github.com/facebookresearch/detectron2/blob/main/detectron2/layers/batch_norm.py # noqa
# Itself from https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def blender(input_tensor, output_masks):
    maks_tensor = input_tensor.clone()

    for i, m in enumerate(output_masks):
        c = [30 / 255.0, 144 / 255.0, 255 / 255.0]
        maks_tensor[:, 0:1, :, :] = torch.where(m, c[0], maks_tensor[:, 0:1, :, :])
        maks_tensor[:, 1:2, :, :] = torch.where(m, c[1], maks_tensor[:, 1:2, :, :])
        maks_tensor[:, 2:3, :, :] = torch.where(m, c[2], maks_tensor[:, 2:3, :, :])

    return 0.5 * input_tensor + 0.5 * maks_tensor
