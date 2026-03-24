import shutil
from datetime import datetime
from pathlib import Path

from embedding_experiments import load_pipeline, batch_generate_embeddings
from create_grid import *

import numpy as np

# --- timestamp ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./output/{ts}_run_2_embedding"

SEEDS_PATH="prompt/00_seed.txt"

INTRO_PATH="prompt/01_intro.txt"
BEAUTY_PATH="prompt/02_beauty.txt"
OBJECT_PATH="prompt/03_object.txt"
STYLE_PATH="prompt/04_style.txt"

MANIPULATION_TYPE_PATH="manipulation/05_manipulation_type.txt"
MANIPULATION_SCALE_VALUES=[-1, -0.75,-0.5,-0.25, 0.0, 0.25,0.5,0.75, 1.0, 1.25,1.5,1.75, 2.0]

NEGATIVE_PROMPT="watermark, text, picture frame, face card, multiple faces"

GEN_STEPS=[30]
GEN_CFG=[8]

WIDTH=512
HEIGHT=744

NUM_COLS=8

# --------------------------------------------------
# load from files
# --------------------------------------------------

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_seeds(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def select_lines(lines, selector):
    if selector is None:
        return list(enumerate(lines, 1))

    if len(lines)==1 and len(selector)==1:
        return [(selector[0], lines[0])]

    if isinstance(selector, range):
        return [(i, lines[i-1]) for i in selector if i-1 < len(lines)]

    if isinstance(selector, list):
        return [(i, lines[i-1]) for i in selector if i-1 < len(lines)]

    if isinstance(selector, slice):
        start = selector.start or 1
        stop = selector.stop or len(lines)+1
        step = selector.step or 1
        return [(i, lines[i-1]) for i in range(start, stop, step)]

    raise ValueError("Unsupported selector")

def select_seeds(seeds, selector):
    if selector is None:
        return seeds
    if len(seeds)==1 and len(selector)==1:
        return [(selector[0], lines[0])]
    return [seeds[i] for i in selector if i < len(seeds)]


# --------------------------------------------------
# MAIN RUN FUNCTION
# --------------------------------------------------

def run_embedding_scale(
    manipulation_type, manipulation_indices,
    scale_values,

    pipe,
    folder_name,

    rows="manipulation_type", cols="manipulation_value",     # grid a x b (focus variables)
    num_cols=NUM_COLS,

    seeds=[510891975915924], seed_selector=[1],

    intro_lines=["a portrait of a"], intro_selector=[1],
    beauty_lines=["beautiful"], beauty_selector=[3],
    object_lines=["person"], object_selector=[1],
    style_lines=["professional photography"], style_selector=[1],

    steps=GEN_STEPS, cfg=GEN_CFG,

    negative_prompt=NEGATIVE_PROMPT,
    w=WIDTH,
    h=HEIGHT
):

    folder = f"{OUTPUT_DIR}/{folder_name}"
    Path(folder).mkdir(parents=True, exist_ok=True)

    seeds_all = select_seeds(seeds, seed_selector)
    intros = select_lines(intro_lines, intro_selector)
    beauties = select_lines(beauty_lines, beauty_selector)
    objects = select_lines(object_lines, object_selector)
    styles = select_lines(style_lines, style_selector)
    manipulations = select_lines(manipulation_type, manipulation_indices)

    amount = len(seeds)*len(intros)*len(beauties)*len(objects)*len(styles)*len(manipulations)*len(scale_values)*len(steps)*len(cfg)
    print(f"Images to generate: {amount}")

    batch_generate_embeddings(
        manipulations,
        scale_values,

        folder,
        pipe,

        steps, cfg,

        seeds,
        intros,
        beauties,
        objects,
        styles,

        amount,

        negative_prompt,
        w,
        h
    )
    print(f"✅ {amount}/{amount} files generated")

    # ----------- generate grid -------------
    if rows is None:
        generate_linear_grid(
            folder,
            cols,
            num_cols,
            WIDTH, HEIGHT
        )
    elif cols is None:
        generate_linear_grid(
            folder,
            rows,
            num_cols,
            WIDTH, HEIGHT
        )
    else:
        generate_ab_grid(
            folder,
            rows, cols, 
            WIDTH, HEIGHT
        )

    # ---- save settings info in txt file -----
    write_run_report(
        folder,
        rows, cols, num_cols,
        seeds, intros, beauties, objects, styles,
        amount,
        negative_prompt,
        manipulations, scale_values,
        steps, cfg,
        w, h
    )

    



# --------------------------------------------------
# ENTRY
# --------------------------------------------------

if __name__ == "__main__":
    
    seeds_all = load_seeds(SEEDS_PATH)
    """
    intro_lines_all = load_lines(INTRO_PATH)
    beauty_lines_all = load_lines(BEAUTY_PATH)
    object_lines_all = load_lines(OBJECT_PATH)
    style_lines_all = load_lines(STYLE_PATH)
    """

    manipulation_type_lines_all = load_lines(MANIPULATION_TYPE_PATH)
    print("values from files are ready")
    
    pipe = load_pipeline()
    print("Pipeline ready")

    focus = [
        "seed", #0
        "intro", #1
        "beauty", #2
        "object", #3
        "style", #4
        "manipulation_type", #5 (we don't do that here)
        "manipulation_value", #6 (we don't do that here)
        "steps", #7 
        "cfg" #8
    ]

    # default settings: 
    # "a portrait of a beautiful person, professional photography", seed: 510891975915924, GEN_STEPS,GEN_GUIDANCESCALE,NEGATIVE_PROMPT
    run_embedding_scale(
        manipulation_type_lines_all, None,
        MANIPULATION_SCALE_VALUES,
        pipe,
        "mani_type_x_scales",
        seeds=seeds_all, seed_selector=[10]
    )

    # 10 seeds x emb_scaled
    # "a portrait of a beautiful person, professional photography", GEN_STEPS,GEN_GUIDANCESCALE,NEGATIVE_PROMPT
    run_embedding_scale(
        manipulation_type_lines_all, [1],
        MANIPULATION_SCALE_VALUES,
        pipe,
        "10_seeds_x_mani1-scales",
        rows="seed", cols="manipulation_value",     # grid a x b (focus variables)

        seeds=seeds_all, seed_selector=range(10,100+1, 10)
    )
    # 100 scales
    # "a portrait of a beautiful person, professional photography", GEN_STEPS,GEN_GUIDANCESCALE,NEGATIVE_PROMPT
    run_embedding_scale(
        manipulation_type_lines_all, [1],
        np.arange(-2.0, 2.1, (2.0-(-2.0)/100)),
        pipe,
        "10_seeds_x_mani1-scales",
        rows=None, cols="manipulation_value",     # grid a x b (focus variables)
        num_cols=10,

        seeds=seeds_all, seed_selector=[10]
    )
    """
    run_embedding_scale(
        manipulation_type_lines_all, [1],
        MANIPULATION_SCALE_VALUES,
        pipe,
        "embedding_scale_test"
        rows="seed", cols="manipulation_value",     # grid a x b (focus variables)
        num_cols=NUM_COLS,

        seeds=[510891975915924], seed_selector=[1],

        intro_lines=["a portrait of a"], intro_selector=[1],
        beauty_lines=["beautiful"], beauty_selector=[3],
        object_lines=["person"], object_selector=[1],
        style_lines=["professional photography"], style_selector=[1],
    )
    """
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
    print(f"✅ Zip created: {OUTPUT_DIR}.zip")