import os
import random
import torch
from diffusers import StableDiffusionXLPipeline


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_seeds(path="prompt/00_seed.txt"):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


# --------------------------------------------------
# Load SDXL
# --------------------------------------------------

def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe.to("cuda")

    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("xFormers enabled")
    except Exception:
        print("xFormers not available")

    return pipe


# --------------------------------------------------
# Batch generation
# --------------------------------------------------

def batch_generate_from_files(
    intro_path,
    beauty_path,
    object_path,
    style_path,
    output_dir,
    negative_prompt="",
    steps=30,
    cfg=7.0
):
    os.makedirs(output_dir, exist_ok=True)

    """
    intro_lines = load_lines(intro_path)
    beauty_lines = load_lines(beauty_path)
    object_lines = load_lines(object_path)
    style_lines = load_lines(style_path)
    """
    intro_lines = ["a portrait of a"]
    beauty_lines = ["beautiful"]
    object_lines = ["person"]
    style_lines = ["professional photography"]

    seeds = load_seeds()
    #assert len(seeds) >= 100, "Need at least 100 seeds"

    total = len(intro_lines) *len(beauty_lines) * len(object_lines) * len(style_lines) * len(seeds)
    print(f"Total amount of images: {total}")

    pipe = load_pipeline()

    for seed in seeds:
        print(f"\n🎲 Seed {seed}")

        count = 0

        for i_i, obj in enumerate(intro_lines, 1):
            for i_b, beauty in enumerate(beauty_lines, 1):
                for i_o, obj in enumerate(object_lines, 1):
                    for i_s, style in enumerate(style_lines, 1):
                        count += 1

                        prompt = f"{intro} {beauty} {obj}, {style}"
                        print(f"[{count}/{total}] {prompt}, seed: {seed}")

                        generator = torch.Generator(device="cuda").manual_seed(seed)

                        image = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            width=512,
                            height=744,
                            original_size=(512, 744),
                            target_size=(512, 744),
                            generator=generator,
                        ).images[0]

                        filename = f"{seed}_{i_1}_{i_b}_{i_o}_{i_s}_0_0.png"
                        image.save(os.path.join(output_dir, filename))


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":
    batch_generate_from_files(
        intro_path="prompt/01_intro.txt"
        beauty_path="prompt/02_beauty.txt",
        object_path="prompt/03_object.txt",
        style_path="prompt/04_style.txt",
        output_dir="output/100seeds",
        negative_prompt="watermark, text, frame",
    )
