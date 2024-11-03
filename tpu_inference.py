import os
import json
import torch
import numpy as np
import PIL
from PIL import Image
from IPython.display import HTML
from pyramid_dit import PyramidDiTForVideoGeneration
from IPython.display import Image as ipython_image
from diffusers.utils import load_image, export_video, export_to_gif
import argparse
import random

# import process pool executor
from concurrent.futures import ProcessPoolExecutor
import time
from huggingface_hub import snapshot_download


""" import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
    SpmdFullyShardedDataParallel as FSDPv2,
    _prepare_spmd_partition_spec,
)
from torch_xla import runtime as xr
 """


def main(
    prompt,
    negative_prompt,
    temp,
    ext_ip=None,
    SERVER_URL=None,
):
    start = time.time()

    """
    xr.use_spmd()

    # Define the mesh following common SPMD practice
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    # To be noted, the mesh must have an axis named 'fsdp', which the weights and activations will be sharded on.
    mesh = xs.Mesh(device_ids, mesh_shape, ("fsdp", "model"))
    xs.set_global_mesh(mesh)"""

    variant = "diffusion_transformer_768p"  # For high resolution
    # variant='diffusion_transformer_384p'       # For low resolution

    model_path = "/dev/shm/model"  # The downloaded checkpoint dir
    snapshot_download(
        "rain1011/pyramid-flow-sd3",
        local_dir=model_path,
        local_dir_use_symlinks=False,
        repo_type="model",
    )
    model_dtype = "bf16"

    # device = xm.xla_device()
    device = torch.device("cpu")

    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_variant=variant,
    )

    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)

    """ def shard_output(output, mesh):
        output = output[0]
        xs.mark_sharding(
            output,
            xs.get_global_mesh(),
            _prepare_spmd_partition_spec(output, shard_maximal=True),
        )


    model.vae = FSDPv2(model.vae)
    model.dit = FSDPv2(model.dit, shard_output=shard_output)
    model.text_encoder = FSDPv2(model.text_encoder)
    """

    if model_dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    # used for 384p model variant
    # width = 640
    # height = 384

    # used for 768p model variant
    width = 1280
    height = 768

    with torch.no_grad(), torch.autocast(
        "xla", enabled=True if model_dtype != "fp32" else False, dtype=torch_dtype
    ):
        frames = model.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=20,
            video_num_inference_steps=15,
            height=height,
            width=width,
            temp=temp,
            guidance_scale=9.0,  # The guidance for the first frame, set it to 7 for 384p variant
            video_guidance_scale=5.0,  # The guidance for the other video latent
            output_type="pil",
            save_memory=False,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
            num_images_per_prompt=1,  # The number of images to generate for each prompt
        )


    export_video(frames, prompt, ext_ip, SERVER_URL, fps=24)
    end = time.time()
    print(f"Time taken: {end - start} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion Video Generation")
    parser.add_argument(
        "--prompt", type=str, required=True, help="Text prompt for image generation"
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="nsfw, worst quality, low quality, normal quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, missing fingers, fewer digits, bad anatomy, bad hands, misplaced objects, mutated anatomy, cropped, text, error, artifacts, horror, jpeg artifacts, signature, watermark, username, artist name",
    )
    parser.add_argument(
        "--temp",
        type=int,
        default=2,
        help="temp in [1, 31] <=> frame in [1, 241] <=> duration in [0, 10s]",
    )
    parser.add_argument(
        "--ext_ip",
        type=str,
        default=None,
        help="External IP address of the server",
    )
    parser.add_argument(
        "--SERVER_URL",
        type=str,
        default=None,
        help="External IP address of the server",
    )

    args = parser.parse_args()
    main(
        args.prompt,
        args.negative_prompt,
        args.temp,
        args.ext_ip,
        args.SERVER_URL,
    )
