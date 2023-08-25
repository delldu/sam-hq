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
from typing import List, Tuple, Optional
import pdb

class MobileSAM(nn.Module):
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


    def forward(self, image, boxes):
        '''
        image: 1xCxHxW
        boxes: Bx4
        '''
        B, C, H, W = image.size()
        boxes[:, 0] = boxes[:, 0].clamp(0, W)
        boxes[:, 1] = boxes[:, 1].clamp(0, H)
        boxes[:, 2] = boxes[:, 2].clamp(0, W)
        boxes[:, 3] = boxes[:, 3].clamp(0, H)

        image, pad_h, pad_w, resize_s = self.pre_process_image(image * 255.0)
        boxes = boxes * resize_s

        point_list = []
        boxes_list = []
        for box in boxes:
            # x1: float, y1: float, x2: float, y2: float = box[0], box[1], box[2], box[3]
            x1: float = float(box[0].item())
            y1: float = float(box[1].item())
            x2: float = float(box[2].item())
            y2: float = float(box[3].item())
            # Suppose small box is point
            if (x2 - x1) < 10.0 and (y2 - y1) < 10.0:
                x: float = (x1 + x2)/2.0
                y: float = (y1 + y2)/2.0
                point_list.append(torch.tensor([x, y]).to(image.device))
            else:
                boxes_list.append(torch.tensor([x1, y1, x2, y2]).to(image.device))

        coords: Optional[torch.Tensor] = None
        labels: Optional[torch.Tensor] = None
        if len(point_list) > 0:
            coords = torch.stack(point_list, dim=0)
            labels = torch.ones((len(point_list),)).to(image.device)
        boxes: Optional[torch.Tensor] = None
        if len(boxes_list) > 0:
            boxes = torch.stack(boxes_list, dim=0)

        iou_predictions, res_masks = self.forward_x(image, coords, labels, boxes)

        masks = self.post_process_mask(res_masks, (pad_h, pad_w), (H, W))

        masks = (masks > 0.0)

        return masks # Bx1xHxW


    def forward_x(self, image, coords: Optional[torch.Tensor], labels: Optional[torch.Tensor], boxes: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        features, interm_features = self.image_encoder(image)

        points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if coords is not None and labels is not None:
            points = (coords[None, :, :], labels[None,  :])

        # self.input_size -- (1024, 1024)
        # points -- 
        # (tensor([[[221., 482.],
        #  [498., 633.],
        #  [750., 379.]]], device='cuda:0'),
        #  tensor([[1, 1, 1]], device='cuda:0', dtype=torch.int32))

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
        )
        iou_predictions, res_masks = self.mask_decoder(
            image_embeddings=features,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
            hq_token_only=True,
            interm_embeddings=interm_features,
        )
        # tensor([[0.5946],
        #         [0.1905],
        #         [0.7547]], device='cuda:0')
        # res_masks.size() -- torch.Size([3, 1, 256, 256])
        # keep_mask = iou_predictions > 0.5
        # iou_predictions = iou_predictions[keep_mask]
        # res_masks = res_masks[keep_mask]
        return iou_predictions, res_masks


    def post_process_mask(self, masks, padded_size: Tuple[int, int], input_size: Tuple[int, int]):
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

    def pre_process_image(self, x) -> Tuple[torch.Tensor, int, int, float]:
        # Normal
        x = (x - self.pixel_mean) / self.pixel_std

        # # Pad
        # h, w = x.shape[-2:]
        # padh = self.image_encoder.img_size - h
        # padw = self.image_encoder.img_size - w
        # x = F.pad(x, (0, padw, 0, padh))

        max_h = self.image_encoder.img_size
        max_w = self.image_encoder.img_size

        # Resize
        B, C, H, W = x.size()
        resize_s = 1.0
        if H > max_h or W > max_w:
            resize_s = min(1.0 * max_h/H, 1.0 * max_w/W)
            SH, SW = int(resize_s * H), int(resize_s * W)
            resize_x = F.interpolate(x, size=(SH, SW), mode="bilinear", align_corners=True)
        else:
            resize_x = x

        # Pad
        pad_h, pad_w = resize_x.size(2), resize_x.size(3)
        b_pad = max_h - pad_h
        r_pad = max_w - pad_w
        resize_pad_x = F.pad(resize_x, (0, r_pad, 0, b_pad))

        return resize_pad_x, pad_h, pad_w, resize_s


    def load_weights(self, model_path="models/SAM.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path

        if os.path.exists(checkpoint):
            print(f"Loading model weight from {checkpoint} ...")
            self.load_state_dict(torch.load(checkpoint))
        else:
            print("-" * 32, "Warnning", "-" * 32)
            print(f"model weight file '{checkpoint}'' not exist !!!")

