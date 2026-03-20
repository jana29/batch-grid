import os
import random
import torch
from diffusers import StableDiffusionXLPipeline


from config_default import *
try:
    from config_local import *
    print(f"✅ saving to locally defined OUTPUT_DIR {OUTPUT_DIR}")
except ImportError:
    print(f"⚠️ using default OUTPUT_DIR {OUTPUT_DIR}")


# --------------------------------------------------
# Utilities
# --------------------------------------------------

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_seeds(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()][:AMOUNT]


# ---- save settings to txt file -------------------
from datetime import datetime



# --------------------------------------------------
# Load SDXL
# --------------------------------------------------
from diffusers import EulerDiscreteScheduler

def load_pipeline():
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

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

def batch_generate_from_files():
    
    # -- get values from config file: --
    output_dir = OUTPUT_DIR

    seeds_path = SEEDS_PATH
    
    intro_path = INTRO_PATH
    beauty_path = BEAUTY_PATH
    object_path = OBJECT_PATH
    style_path = STYLE_PATH

    negative_prompt = NEGATIVE_PROMPT
    steps = GEN_STEPS
    cfg = GEN_GUIDANCESCALE
    w = WIDTH
    h = HEIGHT
    # --

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

    seeds = load_seeds(seeds_path)

    total = len(intro_lines) *len(beauty_lines) * len(object_lines) * len(style_lines) * len(seeds)
    print(f"Total amount of images: {total}")

    pipe = load_pipeline()
    
    count = 0
    for seed in seeds:
        print(f"\n Seed {seed}")
        for i_i, intro in enumerate(intro_lines, 1):
            for i_b, beauty in enumerate(beauty_lines, 1):
                for i_o, obj in enumerate(object_lines, 1):
                    for i_s, style in enumerate(style_lines, 1):
                        count += 1

                        prompt = f"{intro} {beauty} {obj}, {style}"
                        print(f"[{count}/{total}] {prompt}, seed: {seed}")

                        generator = torch.Generator("cuda").manual_seed(seed)

                        image = pipe(
                            prompt=prompt,
                            negative_prompt=negative_prompt,
                            num_inference_steps=steps,
                            guidance_scale=cfg,
                            width=w,
                            height=h,
                            #original_size=(512, 744),
                            #target_size=(512, 744),
                            generator=generator,
                        ).images[0]

                        filename = f"{seed}_{i_i}_{i_b}_{i_o}_{i_s}_0_0.png"
                        image.save(os.path.join(output_dir, filename))
    
    # ---- write overview txt file ------
    file = os.path.join(output_dir, "settings.txt")

    with open(file, "w") as f:
        f.write(f"timestamp: {datetime.now()}\n")
        f.write(f"model: {load_pipeline().pipe}\n")
        f.write(f"inference steps: {steps}\n")
        f.write(f"cfg / guidance scale: {cfg}\n")
        f.write(f"resolution: {w}x{h}\n")
        f.write(f"negative_prompt: {negative_prompt}\n")
        f.write(f"seed_file: {SEEDS_PATH}\n")
        f.write("prompt_template: {intro} {beauty} {obj}, {style}", f"\n")


# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":
    batch_generate_from_files()
