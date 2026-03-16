"""
Baseline evaluation using the Wan2.1-I2V-14B base model on the Force Prompting benchmark.

Runs standard (non-distilled) I2V diffusion inference on benchmark images + captions.
Use --append_force_text to also run the text-only force baseline.

# from Self-Forcing root
python scripts/eval_force_benchmark_WAN.py \
  --checkpoint_dir /path/to/Wan2.1-I2V-14B \
  --output_folder videos/wan_i2v_baseline/original_prompt

# with force description appended to caption
python scripts/eval_force_benchmark_WAN.py \
  --checkpoint_dir /path/to/Wan2.1-I2V-14B \
  --output_folder videos/wan_i2v_baseline/with_force_text \
  --append_force_text
"""
import argparse
import glob
import os
import sys

import pandas as pd
import torch
from PIL import Image
from torchvision.io import write_video
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
from utils.misc import set_seed
from wan.configs import i2v_1_3B
from wan.image2video import WanI2V


BENCHMARK_DIR = "/users/jzou22/scratch/force-prompting/datasets/point-force/test/benchmark"

OBJECT_DESCRIPTIONS = {
    "_apple1": "apple",
    "_apple2": "apple",
    "_apple3": "apple",
    "_apple4": "apple",
    "_balloon3": "hot air balloon",
    "_balloon4": "balloon",
    "_blueberrybush1": "blueberry",
    "_blueberrybush2": "blueberry",
    "_blueberrybush3": "blueberry",
    "_blueberrybush4": "blueberry",
    "_dandelion1": "dandelion",
    "_dandelion3": "dandelion",
    "_dandelion4": "dandelion",
    "_ivy1": "ivy",
    "_ornament1": "bear ornament",
    "_ornament2": "horse ornament",
    "_ornament4": "star ornament",
    "_rose2": "rose",
    "_rose3": "rose",
    "_rose4": "rose",
    "_rose5": "rose",
    "_sunflower2": "sunflower",
    "_sunflower3": "sunflower",
    "_swing3": "swinging chair",
    "_toycar1": "toy car",
    "_toycar2": "toy car",
    "_toycar3": "toy car",
    "_toycar4": "toy bus",
    "_toytrainontrack1": "toy train",
    "_toytrainontrack2": "toy train",
    "_toytrainontrack3": "toy train",
    "_toytrainontrack5": "toy train",
    "_toytrainontrack6": "toy train",
    "_toytrainontrack7": "toy train",
    "_windmill2": "top of the windmill",
}


def get_baseline_prompt_point_force(prompt, object_description, force, angle):
    if force <= 0.25:
        force_str = "is moved not very forcefully"
    elif force <= 0.75:
        force_str = "is moved forcefully"
    else:
        force_str = "is moved very forcefully"

    if (0 <= angle <= 22.5) or (337.5 < angle <= 360):
        dir_str = "to the right"
    elif angle <= 67.5:
        dir_str = "upwards and to the right"
    elif angle <= 112.5:
        dir_str = "upwards"
    elif angle <= 157.5:
        dir_str = "upwards and to the left"
    elif angle <= 202.5:
        dir_str = "to the left"
    elif angle <= 247.5:
        dir_str = "downwards and to the left"
    elif angle <= 292.5:
        dir_str = "downwards"
    else:
        dir_str = "downwards and to the right"

    base = prompt.rstrip()
    while base and base[-1].lower() not in "abcdefghijklmnopqrstuvwxyz":
        base = base[:-1]
    return f"{base}. The {object_description} {force_str}, {dir_str}"


def load_benchmark_samples(benchmark_dir):
    samples = []
    csv_paths = sorted(glob.glob(os.path.join(benchmark_dir, "**", "*.csv"), recursive=True))
    for csv_path in csv_paths:
        if os.path.basename(csv_path) == "benchmark_details.csv":
            continue
        images_dir = os.path.join(os.path.dirname(csv_path), "images")
        csv_stem = os.path.splitext(os.path.basename(csv_path))[0]
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            image_path = os.path.join(images_dir, row["image"])
            if os.path.exists(image_path):
                samples.append((image_path, csv_stem, row))
            else:
                print(f"Warning: image not found: {image_path}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to Wan2.1-I2V-14B checkpoint directory")
    parser.add_argument("--benchmark_dir", type=str, default=BENCHMARK_DIR)
    parser.add_argument("--output_folder", type=str, default="videos/wan_i2v_baseline")
    parser.add_argument("--append_force_text", action="store_true",
                        help="Append natural-language force description to the caption")
    parser.add_argument("--sampling_steps", type=int, default=40)
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=3.0,
                        help="Noise schedule shift (3.0 recommended for 480p)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_grad_enabled(False)

    # t5_cpu keeps the T5 encoder on CPU to save VRAM; offload_model is passed per-call below
    click.echo(click.style(f"\n[Pipeline] Initializing WanI2V from {args.checkpoint_dir}", fg="yellow", bold=True))
    pipeline = WanI2V(
        config=i2v_1_3B,
        checkpoint_dir=args.checkpoint_dir,
        device_id=0,
        t5_cpu=True,
    )
    click.echo(click.style("[Pipeline] Model ready.\n", fg="yellow", bold=True))

    samples = load_benchmark_samples(args.benchmark_dir)
    click.echo(click.style(f"[Pipeline] Found {len(samples)} benchmark samples in {args.benchmark_dir}", fg="blue"))

    subfolder = "with_force_text" if args.append_force_text else "original_prompt"
    output_dir = os.path.join(args.output_folder, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    click.echo(click.style(f"[Pipeline] Saving videos to {output_dir}\n", fg="blue"))

    for i, (image_path, csv_stem, row) in enumerate(tqdm(samples)):
        caption = row["caption"]
        angle = float(row["angle"])
        force = float(row["force"])
        file_id = os.path.splitext(row["image"])[0]

        if args.append_force_text:
            obj_desc = OBJECT_DESCRIPTIONS.get(file_id, "object")
            caption = get_baseline_prompt_point_force(caption, obj_desc, force, angle)

        click.echo(click.style(f"[{i+1}/{len(samples)}] {csv_stem} | angle={angle:.1f} force={force:.2f}", fg="magenta"))
        click.echo(click.style(f"  prompt: {caption[:80]}{'...' if len(caption) > 80 else ''}", fg="white"))

        # benchmark images are 720x480; resize to 832x480
        image = Image.open(image_path).convert("RGB").resize((832, 480))

        # returns [C=3, N_frames, H, W] in [-1, 1]
        video = pipeline.generate(
            input_prompt=caption,
            img=image,
            max_area=480 * 832,
            frame_num=81,
            shift=args.shift,
            sampling_steps=args.sampling_steps,
            guide_scale=args.guide_scale,
            seed=args.seed,
            offload_model=True,
        )

        # [3, N, H, W] in [-1, 1] -> [N, H, W, 3] in [0, 255] uint8
        video = video.permute(1, 2, 3, 0)
        video = ((video + 1.0) / 2.0).clamp(0, 1)
        video = (video * 255).to(torch.uint8).cpu()

        fname = f"{csv_stem}__angle_{angle:.1f}__force_{force:.2f}.mp4"
        write_video(os.path.join(output_dir, fname), video, fps=16)
        click.echo(click.style(f"  saved -> {fname}", fg="green"))

    click.echo(click.style("\n[Pipeline] Done.", fg="yellow", bold=True))


if __name__ == "__main__":
    main()
