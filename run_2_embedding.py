import shutil
from datetime import datetime
from pathlib import Path

from embedding_experiments import load_pipeline, batch_generate_embeddings
from create_grid import *

# --- timestamp ---
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"./output/{ts}_run_2_embedding"

NEGATIVE_PROMPT="watermark, text, frame"

GEN_STEPS=30
GEN_CFG=7

WIDTH=512
HEIGHT=744

NUM_COLS=8


# --------------------------------------------------
# MAIN RUN FUNCTION
# --------------------------------------------------

def run_embedding_scale(
    seeds,
    intros,
    beauties,
    objects,
    styles,
    scale_values,
    pipe,
    folder_name
):

    folder = f"{OUTPUT_DIR}/{folder_name}"
    Path(folder).mkdir(parents=True, exist_ok=True)

    amount = len(seeds)*len(intros)*len(beauties)*len(objects)*len(styles)*len(scale_values)
    print(f"Images to generate: {amount}")

    batch_generate_embeddings(
        folder,
        pipe,

        seeds,
        intros,
        beauties,
        objects,
        styles,

        manipulation_type=1,
        manipulation_values=scale_values,

        negative_prompt=NEGATIVE_PROMPT,
        steps=GEN_STEPS,
        cfg=GEN_CFG,
        w=WIDTH,
        h=HEIGHT
    )

    generate_linear_grid(
        folder,
        "manipulation",
        NUM_COLS,
        WIDTH,
        HEIGHT
    )



# --------------------------------------------------
# ENTRY
# --------------------------------------------------

if __name__ == "__main__":

    pipe = load_pipeline()

    seeds = [1234]
    intros=[(1,"a portrait of a")]
    beauties=[(1,"beautiful")]
    objects=[(1,"woman")]
    styles=[(1,"cinematic photography")]

    scale_values = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]

    run_embedding_scale(
        seeds,
        intros,
        beauties,
        objects,
        styles,
        scale_values,
        pipe,
        "embedding_scale_test"
    )

    shutil.make_archive(OUTPUT_DIR, 'zip', OUTPUT_DIR)