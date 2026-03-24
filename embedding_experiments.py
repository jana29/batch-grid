import os
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerDiscreteScheduler


# --------------------------------------------------
# Load SDXL
# --------------------------------------------------

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
    except:
        print("xFormers not available")

    return pipe


# --------------------------------------------------
# ENCODING UTIL
# --------------------------------------------------

def encode_prompt(pipe, prompt, negative_prompt):

    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt,
        negative_prompt=negative_prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    return (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    )


# --------------------------------------------------
# MANIPULATION OPS
# --------------------------------------------------

def token_weight(prompt_embeds, token_index, weight):
    prompt_embeds[:, token_index] *= weight
    return prompt_embeds


def scale_embedding(prompt_embeds, scale):
    return prompt_embeds * scale


def interpolate_embeddings(e0, e1, t):
    return (1 - t) * e0 + t * e1


def scale_pooled(pooled_embeds, scale):
    return pooled_embeds * scale


# --------------------------------------------------
# Batch generation with embedding manipulation
# --------------------------------------------------

def batch_generate_embeddings(
    manipulation_types,
    manipulation_values,

    output_dir,
    pipe=None,

    step_values=[30], cfg_values=[8],

    seeds=[510891975915924],

    intros=[(1,"a portrait of a")],
    beauties=[(3,"beautiful")],
    objects=[(1,"person")],
    styles=[(1,"professional photography")],

    total=0,

    negative_prompt="watermark, text, picture frame, face card, multiple faces",
    w=512,
    h=744
):

    if pipe is None:
        pipe = load_pipeline()

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for seed in seeds:
        #generator = torch.Generator("cuda").manual_seed(seed)
        for i_i, intro in intros:
            for i_b, beauty in beauties:
                for i_o, obj in objects:
                    for i_s, style in styles:
                        
                        prompt = f"{intro} {beauty} {obj}, {style}"
                        base = encode_prompt(pipe, prompt, negative_prompt)

                        for steps in step_values:
                            for cfg in cfg_values:
                                 for i_m, manipulation in manipulation_types:
                                    for m in manipulation_values:
                                        count += 1

                                        generator = torch.Generator("cuda").manual_seed(seed)

                                        (
                                            prompt_embeds,
                                            negative_embeds,
                                            pooled,
                                            negative_pooled,
                                        ) = base

                                        # clone for safety
                                        prompt_embeds = prompt_embeds.clone()
                                        pooled = pooled.clone()

                                        # -------- apply manipulation --------

                                        if i_m == 1:
                                            prompt_embeds = scale_embedding(prompt_embeds, m)
                                        elif i_m == 2:
                                            pooled = scale_pooled(pooled, m)
                                        elif i_m == 4:
                                            prompt_embeds = scale_embedding(prompt_embeds, m)
                                            pooled = scale_pooled(pooled, m)
                                        elif i_m == 3 or i_m > 4:
                                            raise ValueError(
                                                f"batch_generate_embeddings() can only do 'simple' embedding scaling\n"
                                                f"therefore it only works with manipylation_type 1,2 or 4"
                                                f"Selected manipylation_type = ({i_m}) {manipulation}\n"
                                            )

                                        # (token weighting + interpolation later)

                                        # -------- diffusion --------

                                        print(f"[{count}/{total}] {prompt}, seed={seed}, steps={steps}, cfg={cfg}")
                                        print(f"manipulation: {manipulation} {m}")

                                        image = pipe(
                                            prompt_embeds=prompt_embeds,
                                            negative_prompt_embeds=negative_embeds,
                                            pooled_prompt_embeds=pooled,
                                            negative_pooled_prompt_embeds=negative_pooled,
                                            num_inference_steps=steps,
                                            guidance_scale=cfg,
                                            width=w,
                                            height=h,
                                            generator=generator,
                                        ).images[0]

                                        filename = f"{seed}_{i_i}_{i_b}_{i_o}_{i_s}_{i_m}_{m}_{steps}_{cfg}.png"

                                        image.save(os.path.join(output_dir, filename))

# -----------------------------------------
# interpolate
# -----------------------------------------
def batch_generate_interpolation(
    t_values,
    output_dir,
    pipe=None,

    step_values=[30],
    cfg_values=[8],
    seeds=[0],

    intros=[(1,"a portrait of a")],
    beauties=[(1,"beautiful"), (2,"ugly")],
    objects=[(1,"person")],
    styles=[(1,"cinematic")],

    total=0,

    negative_prompt="watermark",
    w=512, h=744,
):
    if pipe is None:
        pipe = load_pipeline()

    os.makedirs(output_dir, exist_ok=True)

    embed_infos = []
    filename_prompt="0_0_0_0"
    for i_i, intro in intros:
            for i_b, beauty in beauties:
                for i_o, obj in objects:
                    for i_s, style in styles:
                        prompt = f"{intro} {beauty} {obj}, {style}"
                        emb = encode_prompt(pipe, prompt, negative_prompt)

                        embed_infos.append((emb, i_i, i_b, i_o, i_s))
    if len(embed_infos)!=2:
        raise ValueError(
            "Interpolation requires exactly 2 prompts.\n"
            f"Got {len(embed_infos)} prompts from component lengths:\n"
            f"intros={len(intros)}, beauties={len(beauties)}, "
            f"objects={len(objects)}, styles={len(styles)}\n"
        )
    (embA, i_i0, i_b0, i_o0, i_s0) = embed_infos[0]
    (embB, i_i1, i_b1, i_o1, i_s1) = embed_infos[1]
    def pair_str(a,b):
        return f"{a},{b}" if a != b else str(a)
    # get middle part of filename, e.g. 0_3,8_0_0
    filename_prompt = (
        f"{pair_str(i_i0,i_i1)}_{pair_str(i_b0,i_b1)}_{pair_str(i_o0,i_o1)}_{pair_str(i_s0,i_s1)}"
    )

    (pA, nA, poolA, npoolA) = embA
    (pB, _, poolB, _) = embB

    """
    embed cloning outside of loop?
        # clone once to avoid mutation side effects
        pA = pA.clone()
        pB = pB.clone()
        poolA = poolA.clone()
        poolB = poolB.clone()
    """

    if total == 0:
        total = len(seeds)*len(step_values)*len(cfg_values)*len(t_values)
    count = 0
    for seed in seeds:
        for steps in step_values:
            for cfg in cfg_values:
                for t in t_values:

                    count += 1
                    print(f"[{count}/{total}] seed={seed} t={t}")

                    # --- fresh tensors every render ---
                    pA = embA[0].clone()
                    poolA = embA[2].clone()

                    pB = embB[0].clone()
                    poolB = embB[2].clone()

                    prompt_embeds = interpolate_embeddings(pA, pB, t)
                    pooled = interpolate_embeddings(poolA, poolB, t)

                    generator = torch.Generator("cuda").manual_seed(seed)

                    image = pipe(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=nA,
                        pooled_prompt_embeds=pooled,
                        negative_pooled_prompt_embeds=npoolA,
                        num_inference_steps=steps,
                        guidance_scale=cfg,
                        width=w,
                        height=h,
                        generator=generator,
                    ).images[0]

                    t_str = f"{t:.4f}".replace(".", "p")
                    filename = f"{seed}_{filename_prompt}_3_{t_str}_{steps}_{cfg}.png"

                    image.save(os.path.join(output_dir, filename))

    print("✅ interpolation batch finished")

if __name__ == "__main__":
    print("Run this file through run_2_embedding.py")