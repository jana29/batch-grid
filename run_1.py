# ---- INFO ----
# this file generates sets of 100 images + image grid + csv & txt file with info
# ... with the following settings:
# - 100 seeds
# - 10 seeds / beauty indices 1-10
# -> total of XX images, X grids, ...

# IN TERMINAL:
# conda activate sdxl
# python run_1.py

import shutil
from datetime import datetime
from pathlib import Path

# --- image batch settings ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./output/{ts}_run_1"

SEEDS_PATH="prompt/00_seed.txt"

INTRO_PATH="prompt/01_intro.txt"
BEAUTY_PATH="prompt/02_beauty.txt"
OBJECT_PATH="prompt/03_object.txt"
STYLE_PATH="prompt/04_style.txt"

NEGATIVE_PROMPT="watermark, text, picture frame, face card, multiple faces" #never changed (so far)

MANIPULATION="NONE"     #cannot change in this file

GEN_STEPS=30            #default
GEN_GUIDANCESCALE=8     #default

HEIGHT = 744            #never changed (so far)
WIDTH = 512             #never changed (so far)

# --- grid settings ----

NUM_COLS=10

# =======================================================================
#                               load from files
# =======================================================================

def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_seeds(path):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]

def select_lines(lines, selector):
    if selector is None:
        return list(enumerate(lines, 1))

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
    return [seeds[i] for i in selector if i < len(seeds)]


# =======================================================================
#                               MAIN FUNCTION
# =======================================================================
def run(
    cols, rows,     # grid a x b

    seed_lines, seed_selector,

    intro_lines, intro_selector,
    beauty_lines, beauty_selector,
    object_lines, obj_selector,
    style_lines, style_selector,

    pipe,

    folder_name,

    steps=[GEN_STEPS],
    cfg=[GEN_GUIDANCESCALE]
):
    # set output subfolder

    folder = f"{OUTPUT_DIR}/{folder_name}"
    Path(folder).mkdir(parents=True, exist_ok=True)
    print(f"Generating Images in Folder {folder_name}")

    # select
    seeds = select_seeds(seed_lines, seed_selector)

    intros = select_lines(intro_lines, intro_selector)
    beauties = select_lines(beauty_lines, beauty_selector)
    objects = select_lines(object_lines, obj_selector)
    styles = select_lines(style_lines, style_selector)

    amount = len(seeds) * len(intros) *len(beauties) * len(objects) * len(styles) * len(steps) * len(cfg)
    print(f"amount of images in this batch: {amount}")

    # -------- generate images --------
    batch_generate( 
        folder,
        pipe,

        seeds,
        
        intros,
        beauties,
        objects,
        styles,

        amount,

        NEGATIVE_PROMPT,
        steps,
        cfg,
        WIDTH,
        HEIGHT
    )
   
    # ----------- generate grid -------------
    if rows is None:
        generate_linear_grid(
            folder,
            cols,          # "seed", "intro", "beauty", ...
            NUM_COLS,
            WIDTH, HEIGHT
        )
    elif cols is None:
        generate_linear_grid(
            folder,
            rows,          # "seed", "intro", "beauty", ...
            NUM_COLS,
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
        rows, cols, NUM_COLS,
        seeds, intros, beauties, objects, styles,
        amount,
        NEGATIVE_PROMPT,
        MANIPULATION, MANIPULATION,
        GEN_STEPS, GEN_GUIDANCESCALE,
        WIDTH, HEIGHT
    )
    

# =======================================================================
#                                    RUN
# =======================================================================

if __name__ == "__main__":

    seeds_all = load_seeds(SEEDS_PATH)

    intro_lines_all = load_lines(INTRO_PATH)
    beauty_lines_all = load_lines(BEAUTY_PATH)
    object_lines_all = load_lines(OBJECT_PATH)
    style_lines_all = load_lines(STYLE_PATH)

    print("Promptfiles and Seeds ready")

    from generate_images import load_pipeline
    pipeline = load_pipeline()
    print("Pipeline ready")

    from generate_images import batch_generate
    from create_grid import *

    focus = [
        "seed", #0
        "intro", #1
        "beauty", #2
        "object", #3
        "style", #4
        "manipulation_type", #5
        "manipulation", #6
        "steps", #7 
        "cfg" #8
    ]

    """
    # 100 seeds,
    # a portrait of a beautiful person, professional photography
    run(
        focus[0], None,   # seeds

        seeds_all, None,

        intro_lines_all, [1],
        beauty_lines_all, [3],
        object_lines_all, [1],
        style_lines_all, [1],

        pipeline,

        "100_seeds" #Folder Name
    )

    # 10 seeds, 10 beauties
    run(
        focus[2], focus[0],   # col x rows : beauty x seed

        seeds_all, range(10,100+1, 10), # range 10,20,30,...

        intro_lines_all, [1],
        beauty_lines_all, range(1,10+1), # range 1-10
        object_lines_all, [1],
        style_lines_all, [1],

        pipeline,

        "beauty_x_seeds" #Folder Name
    )
    """

    # cfg x steps test
    run(
        focus[7], focus[8],   # col x rows

        seeds_all, [1], 

        intro_lines_all, [1],
        beauty_lines_all, [3], 
        object_lines_all, [1],
        style_lines_all, [1],

        pipeline,

        "steps_x_cfg", #Folder Name
        [10, 20], #steps
        [4, 6, 8] #cfg
    )

    # --- zip folder ---
    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)
    print(f"✅ Zip created: {OUTPUT_DIR}.zip")



# =========================================================================