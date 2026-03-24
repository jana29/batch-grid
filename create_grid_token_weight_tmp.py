import os
import csv
from PIL import Image


# ------------------------------------------------------------
# CONFIG — adjust only this if needed
# ------------------------------------------------------------

FOLDER = "output/20260324_155359_run_2_embedding/token_weight_intro_beauty_object"

WIDTH = 512
HEIGHT = 744
THUMB_SCALE = 0.5

ROUND_DECIMALS = 4   # VERY IMPORTANT for stable grid


# ------------------------------------------------------------
# PARSER
# ------------------------------------------------------------

def parse_token_weight_filename(fname):
    """
    Expected filename format:

    seed_intro_beauty_object_style_6_axis-weight_steps_cfg.png
    """

    parts = fname.replace(".png", "").split("_")

    try:
        seed = int(parts[0])
        manipulation_type = int(parts[5])

        packed = parts[6]

        axis_str, rest = packed.split("-", 1)

        if packed.count("--"):
            weight = -float(rest)
        else:
            weight = float(rest)

        axis = int(axis_str)

        weight = round(weight, ROUND_DECIMALS)

        return {
            "seed": seed,
            "axis": axis,
            "weight": weight,
            "file": fname,
        }

    except Exception as e:
        print("Parse fail:", fname, e)
        return None


# ------------------------------------------------------------
# LOAD FILES
# ------------------------------------------------------------

if not os.path.exists(FOLDER):
    raise FileNotFoundError(f"Folder not found: {FOLDER}")

files = [
    f for f in os.listdir(FOLDER)
    if f.endswith(".png") and f.split("_")[0].isdigit()
]

meta = []

for f in files:
    m = parse_token_weight_filename(f)
    if m:
        meta.append(m)

if not meta:
    print("❌ No valid images parsed")
    exit()

print(f"✅ Parsed {len(meta)} images")


# ------------------------------------------------------------
# BUILD GRID COORDINATES
# ------------------------------------------------------------

axes = sorted(set(m["axis"] for m in meta))
weights = sorted(set(m["weight"] for m in meta))

rows = len(axes)
cols = len(weights)

print(f"Grid size → {rows} rows × {cols} cols")

lookup = {
    (m["axis"], m["weight"]): m["file"]
    for m in meta
}


# ------------------------------------------------------------
# CREATE CANVAS
# ------------------------------------------------------------

thumb_w = int(WIDTH * THUMB_SCALE)
thumb_h = int(HEIGHT * THUMB_SCALE)

canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")


# ------------------------------------------------------------
# RENDER GRID
# ------------------------------------------------------------

missing = 0

for r, axis in enumerate(axes):
    for c, weight in enumerate(weights):

        fname = lookup.get((axis, weight))

        if fname:
            img = Image.open(os.path.join(FOLDER, fname))
            img = img.resize((thumb_w, thumb_h))
            canvas.paste(img, (c * thumb_w, r * thumb_h))
        else:
            missing += 1

print(f"Missing cells: {missing}")


# ------------------------------------------------------------
# SAVE GRID
# ------------------------------------------------------------

out_png = os.path.join(FOLDER, "grid_token_weight.png")
out_pdf = os.path.join(FOLDER, "grid_token_weight.pdf")

canvas.save(out_png)
canvas.save(out_pdf, "PDF", resolution=300)

print("✅ Grid saved:")
print(out_png)
print(out_pdf)


# ------------------------------------------------------------
# SAVE CSV METADATA
# ------------------------------------------------------------

csv_path = os.path.join(FOLDER, "grid_token_weight.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["axis", "weight", "filename"])

    for m in sorted(meta, key=lambda x: (x["axis"], x["weight"])):
        writer.writerow([m["axis"], m["weight"], m["file"]])

print("✅ CSV saved:")
print(csv_path)