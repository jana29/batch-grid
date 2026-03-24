import os
import random
import torch
from diffusers import StableDiffusionXLPipeline


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

def batch_generate( 
    # -- get values from config file: --
    output_dir = "./output",
    pipe=None,

    seeds = [210394857610295],
    
    intros = [(0, "a portrait of a")],
    beauties = [(3, "beautiful")],
    objects = [(0, "person")],
    styles = [(0, "professional photography")],

    amount = 1,

    negative_prompt = "watermark, text, picture frame, face card, multiple faces",
    step_values = [30],
    cfg_values = [8],
    w = 512,
    h = 744
):
    if pipe is None:
        pipe = load_pipeline()

    count = 0
    for seed in seeds:
        #generator = torch.Generator("cuda").manual_seed(seed)
        for i_i, intro in intros:
            for i_b, beauty in beauties:
                for i_o, obj in objects:
                    for i_s, style in styles:
                        for steps in step_values:
                            for cfg in cfg_values:
                                count += 1

                                prompt = f"{intro} {beauty} {obj}, {style}"
                                print(f"[{count}/{amount}] {prompt}, seed: {seed}, steps: {steps}, cfg: {cfg}")

                                generator = torch.Generator("cuda").manual_seed(seed)
                                
                                image = pipe(
                                    prompt=prompt,
                                    negative_prompt=negative_prompt,
                                    num_inference_steps=steps,
                                    guidance_scale=cfg,
                                    width=w,
                                    height=h,
                                    original_size=(w, h),
                                    target_size=(w, h),
                                    generator=generator,
                                ).images[0]

                                filename = f"{seed}_{i_i}_{i_b}_{i_o}_{i_s}_0_0_{steps}_{cfg}.png"
                                image.save(os.path.join(output_dir, filename))




# --------------------------------------------------
# Entry
# --------------------------------------------------

if __name__ == "__main__":
    #print("Running Test with default settings")
    print("Run this file through run_1.py")
    #batch_generate_from_files()
