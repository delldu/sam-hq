# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2023(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, Mon 21 Aug 2023 03:06:09 PM CST
#
# ************************************************************************************/
#
import pdb
import os
import torch
import SAM

from tqdm import tqdm

if __name__ == "__main__":
    model, device = SAM.get_model()
    # model = torch.jit.script(model)
    # print(model)

    B, C, H, W = 1, 3, 1025, 1036

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as p:
        for ii in range(10):
            x = torch.randn(B, C, H, W) * 255
            boxes = torch.tensor(
                [
                    [0.0, 0.0, 1024.0, 1024.0],
                ]
            )
            with torch.no_grad():
                y = model(x.to(device), boxes.to(device))
            torch.cuda.synchronize()
        p.step()

    print(p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    os.system("nvidia-smi | grep python")
