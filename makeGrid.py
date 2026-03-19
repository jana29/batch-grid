import os
from PIL import Image

folder = "output/100seeds"
#target_seed = "230872265211790"

grid = {}

for fname in os.listdir(folder):

    if not fname.endswith(".png"):
        continue

    parts = fname.split("_")

    """
    if len(parts) < 7:
        continue

    seed = parts[0]
    if seed != target_seed:
        continue
    """

    try:
        key = tuple(int(p) for p in parts[1:7])   # i1..i5 → ROW KEY
        col = int(parts[0])                       # i6 → COLUMN
    except:
        continue

    grid.setdefault(key, {})
    grid[key][col] = fname


# ---- SORT ROWS ----
row_keys = sorted(grid.keys())

# ---- CREATE OVERVIEW CSV ----
import csv
cols = 10

table_csv = f"{target_seed}_grid_table.csv"

with open(table_csv, "w", newline="", encoding="utf-8") as f:

    writer = csv.writer(f)

    for key in row_keys:

        row_filenames = []

        for col in range(1, cols + 1):

            fname = grid[key].get(col, "")
            row_filenames.append(fname)

        writer.writerow(row_filenames)

print("✅ grid table saved")


# ---- GRID PARAMS ----
cols = 12
thumb_w = 256
thumb_h = 372

canvas = Image.new(
    "RGB",
    (cols * thumb_w, len(row_keys) * thumb_h),
    "white"
)

# ---- PASTE ----
for r_idx, key in enumerate(row_keys):

    for col in range(1, cols + 1):

        fname = grid[key].get(col)
        if fname is None:
            continue

        path = os.path.join(folder, fname)

        img = Image.open(path).resize((thumb_w, thumb_h))

        x = (col - 1) * thumb_w
        y = r_idx * thumb_h

        canvas.paste(img, (x, y))


canvas.save(f"{target_seed}_grid.png")
canvas.save(f"{target_seed}_grid.pdf")

print("✅ grid done")
"""