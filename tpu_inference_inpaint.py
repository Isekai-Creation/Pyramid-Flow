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
import time
import argparse
import random
from vision_process import get_image
from huggingface_hub import snapshot_download


def main(
    prompt,
    negative_prompt,
    temp,
    image_url=None,
    ext_ip=None,
    SERVER_URL=None,
):
    start = time.time()
    variant = "diffusion_transformer_768p"  # For high resolution
    # variant='diffusion_transformer_384p'       # For low resolution

    width = 1280
    height = 768

    model_path = "/dev/shm/model"  # The downloaded checkpoint dir
    snapshot_download(
        "rain1011/pyramid-flow-sd3",
        local_dir=model_path,
        local_dir_use_symlinks=False,
        repo_type="model",
    )
    model_dtype = "bf16"

    device = torch.device("cpu")

    model = PyramidDiTForVideoGeneration(
        model_path,
        model_dtype,
        model_variant=variant,
    )

    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device)

    if model_dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif model_dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    if image_url:
        image = get_image(image_url, resized_width=width, resized_height=height)

    with torch.no_grad(), torch.cuda.amp.autocast(
        enabled=True if model_dtype != "fp32" else False, dtype=torch_dtype
    ):
        frames = model.generate_i2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=image,
            num_inference_steps=20,
            video_num_inference_steps=15,
            temp=temp,
            guidance_scale=9.0,
            video_guidance_scale=5.0,
            output_type="pil",
            save_memory=False,  # If you have enough GPU memory, set it to `False` to improve vae decoding speed
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
        "--image_url",
        type=str,
        default=None,
        help="URL of the initial image for image-to-image generation",
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
        args.image_url,
        args.ext_ip,
        args.SERVER_URL,
    )
