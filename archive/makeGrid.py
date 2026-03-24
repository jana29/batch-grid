import os
import csv
from PIL import Image
from math import ceil

from config_default import *
"""try:
    from config_local import *
    print(f"✅ loading images from locally defined OUTPUT_DIR {OUTPUT_DIR}")
except ImportError:
    print(f"⚠️ using default OUTPUT_DIR {OUTPUT_DIR}")"""

folder = OUTPUT_DIR


# --- get prompt -----
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_seeds(path="prompt/00_seed.txt"):
    with open(path, "r") as f:
        return [int(line.strip()) for line in f if line.strip()]


def generate_grid (
    cols = NUM_COLS
):
    """
    intro_lines = load_lines("prompt/01_intro.txt")
    beauty_lines = load_lines("prompt/02_beauty.txt")
    object_lines = load_lines("prompt/03_object.txt")
    style_lines = load_lines("prompt/04_style.txt")
    """
    intro_lines = ["a portrait of a"]
    beauty_lines = ["beautiful"]
    object_lines = ["person"]
    style_lines = ["professional photography"]

    # ------

    grid = {}

    files = sorted([
        f for f in os.listdir(folder)
        if f.endswith(".png") and f.split("_")[0].isdigit()
    ])
    

    print(f"cols: {cols}")

    rows = ceil(len(files) / cols)

    csv_rows = [["" for _ in range(cols)] for _ in range(rows)]


    # ---- image PARAMS ----
    thumb_w = int(WIDTH/2)
    thumb_h = int(HEIGHT/2)

    canvas = Image.new(
        "RGB",
        (cols * thumb_w, rows * thumb_h),
        "white"
    )

    # ---- create grid + csv ----
    for idx, fname in enumerate(files):

        r = idx // cols
        c = idx % cols

        path = os.path.join(folder, fname)

        img = Image.open(path).resize((thumb_w, thumb_h))

        canvas.paste(img, (c * thumb_w, r * thumb_h))


        # -- CSV - get infos from filename ---
        parts = fname.replace(".png", "").split("_")
        seed = parts[0]

        try:
            i_intro = int(parts[1]) - 1
            i_beauty = int(parts[2]) - 1
            i_object = int(parts[3]) - 1
            i_style = int(parts[4]) - 1
        except:
            i_intro = i_beauty = i_object = i_style = 0

        intro = intro_lines[i_intro]
        beauty = beauty_lines[i_beauty]
        obj = object_lines[i_object]
        style = style_lines[i_style]

        suffix = f"{parts[7]}, {parts[8]}"
        
        prompt = f"{intro} {beauty} {obj}, {style}"
        cell_text = f"{prompt}; seed: {seed}; {suffix}"

        csv_rows[r][c] = cell_text


    # ---- save grid ----

    canvas.save(f"{folder}/grid.png")
    canvas.save(f"{folder}/grid.pdf")
    print(f"✅ grid png and pdf saved to {OUTPUT_DIR}")

    # ---- save CSV ----

    with open(f"{folder}/grid.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)c
    print(f"{len(csv_rows)} rows x {len(csv_rows[0])} cols")
    print(f"✅ grid table saved to {OUTPUT_DIR}")




if __name__ == "__main__":
    generate_grid()