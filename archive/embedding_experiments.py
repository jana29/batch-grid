"""
about embedding manipulation

diffusion generation simplified, according to chatGPT:

text → tokenizer → text encoder → prompt embeddings
                                    ↓
                        diffusion sampling loop
                        (steps, cfg, scheduler, seed…)

embeddings = conditioning signal
steps / cfg = sampling process controls
-> They act at different stages.

⭐ What embeddings actually represent

Embeddings encode:
- semantic meaning
- style direction
- attribute strength
- relationships between tokens

Example:
    embedding("beautiful person") - embedding("ugly person")
→ “beauty direction” in latent space.

"""


import time
import torch
from itertools import product


def generate_embedding_scale_grid(
    pipe,
    seeds,
    beauty,
    objects,
    styles,
    scales,
    negative_prompt,
    steps,
    cfg,
    width,
    height,
    output_dir,
    skip_existing=True,
):

    len_prompt_combos = len(product(beauty, objects, styles))
    total_images = len_prompt_combos * len(seeds) * len(scales)
    print(f"\nEmbedding experiment")
    print(f"Prompt combos: {len_prompt_combos}")
    print(f"Seeds: {len(seeds)}")
    print(f"Scales: {len(scales)}")
    print(f"TOTAL IMAGES: {total_images}\n")

    start_time = time.time()
    img_index = 0


    for combo_index, (b, o, s) in enumerate(product(beauty, objects, styles), 1):

        prompt = f"a portrait of a {b} {o}, {s}"

        # encode ONCE per prompt combo
        prompt_embeds, neg_embeds, pooled, neg_pooled = pipe.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

        for seed in seeds:
            for scale in scales:

                img_index += 1

                filename = f"{seed}_{combo_index}_em{scale:.2f}.png"
                path = os.path.join(output_dir, filename)

                if skip_existing and os.path.exists(path):
                    print(f"[skip] {filename}")
                    continue

                generator = torch.Generator("cuda").manual_seed(seed)

                scaled_embeds = prompt_embeds * scale

                image = pipe(
                    prompt_embeds=scaled_embeds,
                    negative_prompt_embeds=neg_embeds,
                    pooled_prompt_embeds=pooled,
                    negative_pooled_prompt_embeds=neg_pooled,
                    num_inference_steps=steps,
                    guidance_scale=cfg,
                    width=width,
                    height=height,
                    original_size=(width, height),
                    target_size=(width, height),
                    generator=generator,
                ).images[0]

                image.save(path)

                elapsed = time.time() - start_time
                eta = (elapsed / img_index) * (total_images - img_index)

                print(
                    f"[{img_index}/{total_images}] "
                    f"seed={seed} scale={scale:.2f} "
                    f"ETA={eta/60:.1f}m"
                )


def interpolate_embeddings(
    pipe,
    prompt_a,
    prompt_b,
    steps_list,
    negative_prompt,
    seeds,
    steps,
    cfg,
    width,
    height,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    emb_a, neg_a, pool_a, neg_pool_a = pipe.encode_prompt(
        prompt=prompt_a,
        negative_prompt=negative_prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    emb_b, _, pool_b, _ = pipe.encode_prompt(
        prompt=prompt_b,
        negative_prompt=negative_prompt,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    total = len(steps_list) * len(seeds)
    i = 0

    for t in steps_list:
        emb = emb_a * (1 - t) + emb_b * t
        pool = pool_a * (1 - t) + pool_b * t

        for seed in seeds:
            i += 1

            generator = torch.Generator("cuda").manual_seed(seed)

            image = pipe(
                prompt_embeds=emb,
                negative_prompt_embeds=neg_a,
                pooled_prompt_embeds=pool,
                negative_pooled_prompt_embeds=neg_pool_a,
                num_inference_steps=steps,
                guidance_scale=cfg,
                width=width,
                height=height,
                generator=generator,
            ).images[0]

            fname = f"interp_{t:.2f}_{seed}.png"
            image.save(os.path.join(output_dir, fname))

            print(f"[{i}/{total}] interp={t:.2f} seed={seed}")