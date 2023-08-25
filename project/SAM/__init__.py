"""Image Recognize Anything Model Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, Thu 13 Jul 2023 01:55:56 PM CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm
from PIL import Image, ImageDraw

import torch
import todos
from . import sam
from .common import blender
from torchvision.transforms import Compose, ToTensor, ToPILImage
from pathlib import Path
import pdb

def draw_points_boxes(tensor, boxes):
    tensor.unsqueeze(0)
    image = ToPILImage()(tensor.squeeze(0))

    draw = ImageDraw.Draw(image)

    for box in boxes:
        x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
        # Suppose small box is point
        if (x2 - x1) < 10.0 and (y2 - y1) < 10.0:
            x = int((x1 + x2)/2.0)
            y = int((y1 + y2)/2.0)
            draw.ellipse((x-5, y-5, x+5, y+5), fill="yellow")
        else:
            draw.rectangle((x1, y1, x2, y2), fill=None, outline="green", width=1)

    image = ToTensor()(image)

    return image.unsqueeze(0)


def create_model():
    """
    Create model
    """

    model = sam.MobileSAM()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running model on {device} ...")

    return model, device


def get_model():
    """Load jit script model."""

    model, device = create_model()
    # print(model)

    # https://github.com/pytorch/pytorch/issues/52286
    torch._C._jit_set_profiling_executor(False)
    # C++ Reference
    # torch::jit::getProfilingMode() = false;                                                                                                             
    # torch::jit::setTensorExprFuserEnabled(false);

    # model = torch.jit.script(model)
    # todos.data.mkdir("output")
    # if not os.path.exists("output/SAM.torch"):
    #     model.save("output/SAM.torch")

    return model, device


def predict(test_database, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_model()
    transform = Compose([
        lambda image: image.convert("RGB"),
        ToTensor(),
    ])

    # start predict
    progress_bar = tqdm(total=len(test_database))
    for filename, boxes_list in test_database.items():
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_image = transform(image).unsqueeze(0).to(device)
        input_boxes = boxes_list.to(device)

        with torch.no_grad():
            output_mask = model(input_image, input_boxes)

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        output_tensor = blender(input_image, output_mask)
        output_tensor = draw_points_boxes(output_tensor.cpu(), boxes_list)
        todos.data.save_tensor([input_image, output_tensor], output_file)

    progress_bar.close()

    todos.model.reset_device()
