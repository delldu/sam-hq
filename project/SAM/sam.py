# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
from torch import nn
from torch.nn import functional as F

from .image_encoder import TinyViT
from .mask_decoder import MaskDecoderHQ
from .prompt_encoder import PromptEncoder

from typing import List, Tuple

import pdb


def resize_pad(x, max_h: int, max_w: int, max_times: int) -> Tuple[torch.Tensor, int, int]:
    # Need Resize ?
    B, C, H, W = x.size()
    if H > max_h or W > max_w:
        s = min(max_h / H, max_w / W)
        SH, SW = int(s * H), int(s * W)
        resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=True)
    else:
        resize_x = x

    # Need Pad ?
    pad_h, pad_w = resize_x.size(2), resize_x.size(3)
    if pad_h % max_times != 0 or pad_w % max_times != 0:
        # pad_h:  802 pad_w:  1024 for max_times == 8 ?
        r_pad = 0
        if (pad_w % max_times) != 0:
            r_pad = max_times - (pad_w % max_times)
        b_pad = 0
        if (pad_h % max_times) != 0:
            b_pad = max_times - (pad_h % max_times)

        resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad), mode="replicate")
    else:
        resize_pad_x = resize_x

    return resize_pad_x, pad_h, pad_w


def pad_resize(y, pad_h: int, pad_w: int, h: int, w: int):
    y = y[:, :, 0 : pad_h, 0 : pad_w]  # Remove Pads
    return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=True)  # Remove Resize



class MobileSAM(nn.Module):
    mask_threshold: float = 0.0
    def __init__(self, pixel_mean=[123.675, 116.28, 103.53], pixel_std=[58.395, 57.12, 57.375]):
        super().__init__()
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 2

        prompt_embed_dim = 256
        image_size = 1024
        vit_patch_size = 16
        image_embedding_size = image_size // vit_patch_size # 64

        self.image_size = image_size
        self.image_encoder = TinyViT()
        self.prompt_encoder = PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoderHQ(
            num_multimask_outputs=3,
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)


        self.load_weights()


    def forward(self, image, boxes=None):
        '''
        image: 1xCxHxW
        boxes: Bx4
        '''
        
        



        return image



        # B, C, H, W = x.size()

        # resize_pad_x, pad_h, pad_w = resize_pad(x, self.MAX_H, self.MAX_W, self.MAX_TIMES)
        # y = self.standard_forward(resize_pad_x)

        # return pad_resize(y, pad_h, pad_w, H, W)        


    def standard_forward(self, image):
        image = self.preprocess(image)

        features, self.interm_features = self.image_encoder(image)

        layer_points = build_grid_points()

        coords = layer_points[0].to(image.device)
        labels = torch.Tensor(layer_points[0].size(0)).to(image.device)

        batch_size = 32
        for i in range(0, coords.size(0), batch_size):
            batch_coords = coords[i: i + batch_size]
            batch_labels = labels[i: i + batch_size]
            batch_iou_predictions, batch_res_masks = self.points_forward(features, batch_coords, batch_labels)

        return image # batch_res_masks


    def points_forward(self, features, coords, labels) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
        # image.size() -- [1, 3, 1024, 1024]
        # features = self.image_encoder(image)

        features.size() -- [1, 256, 64, 64]
        coords.size() -- [64, 2]
        labels.size() -- [64]
        '''
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(coords[:, None, :], labels[:, None]),
            boxes=None,
            masks=None,
        )
        res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=False,
            interm_embeddings=self.interm_features,
        )
        return iou_predictions, res_masks


    def postprocess_masks(self, masks, input_size: Tuple[int, int], padded_size: Tuple[int, int]):
        # masks.size() -- [64, 3, 256, 256]
        # padded_size -- (1024, 1024)
        # input_size -- (1024, 1024)

        # self.image_encoder.img_size -- 1024
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :padded_size[0], :padded_size[1]]
        masks = F.interpolate(masks, input_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x) -> Tuple[torch.Tensor, int, int]:
        # x.size() -- [1, 3, 1024, 1024], x.dtype=uint8
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))

        return x


    def load_weights(self, model_path="models/SAM.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"model weight file '{checkpoint}'' not exist !!!")

