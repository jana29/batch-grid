import os
import csv
from PIL import Image
from math import ceil
from datetime import datetime


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def parse_axis(x):
    if "," in x:
        return x
    return int(x)


def parse_float(x):
    """
    Supports:
    - normal float: 1.25
    - p-format: 1p25
    - packed token-weight: 3--0.75
    """

    x = x.replace("p", ".")

    # packed axis-weight
    if "-" in x[1:]:
        axis, rest = x.split("-", 1)

        if x.count("--"):
            rest = "-" + rest

        try:
            return round(float(rest), 6)
        except:
            pass

    return round(float(x), 6)


def parse_filename(fname):
    parts = fname.replace(".png", "").split("_")

    try:
        return {
            "seed": int(parts[0]),
            "intro": parse_axis(parts[1]),
            "beauty": parse_axis(parts[2]),
            "object": parse_axis(parts[3]),
            "style": parse_axis(parts[4]),
            "manipulation_type": int(parts[5]),
            "manipulation_value": parse_float(parts[6]),
            "steps": int(parts[7]),
            "cfg": float(parts[8]),
        }
    except Exception as e:
        print("Parse fail:", fname, e)
        return None


def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.endswith(".png") and f.split("_")[0].isdigit()
    ]


# ------------------------------------------------------------
# GRID TYPE 1
# ------------------------------------------------------------

def generate_ab_grid(folder, row_axis, col_axis, width, height, thumb_scale=0.5):

    files = list_images(folder)

    coords = []
    for f in files:
        meta = parse_filename(f)
        if meta:
            coords.append((meta, f))

    if not coords:
        print("No valid images found.")
        return

    row_values = sorted(set(m[row_axis] for m, _ in coords))
    col_values = sorted(set(m[col_axis] for m, _ in coords))

    rows = len(row_values)
    cols = len(col_values)

    print(f"Generating A×B grid: rows={row_axis} ({rows}) cols={col_axis} ({cols})")

    thumb_w = int(width * thumb_scale)
    thumb_h = int(height * thumb_scale)

    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")

    csv_rows = [["" for _ in range(cols)] for _ in range(rows)]

    lookup = {}
    for meta, fname in coords:
        key = (meta[row_axis], meta[col_axis])
        lookup[key] = fname

    for r, rv in enumerate(row_values):
        for c, cv in enumerate(col_values):

            fname = lookup.get((rv, cv))

            if fname:
                img = Image.open(os.path.join(folder, fname))
                img = img.resize((thumb_w, thumb_h))
                canvas.paste(img, (c * thumb_w, r * thumb_h))
                csv_rows[r][c] = fname
            else:
                csv_rows[r][c] = "missing"

    filename = f"grid_{row_axis}_x_{col_axis}"
    canvas.save(os.path.join(folder, f"{filename}.png"))
    canvas.save(os.path.join(folder, f"{filename}.pdf"), "PDF", resolution=300.0)

    with open(os.path.join(folder, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("✅ A×B grid saved")


# ------------------------------------------------------------
# GRID TYPE 2
# ------------------------------------------------------------

def generate_linear_grid(folder, axis, cols, width, height, thumb_scale=0.5):

    files = list_images(folder)

    coords = []
    for f in files:
        meta = parse_filename(f)
        if meta:
            coords.append((meta, f))

    if not coords:
        print("No valid images found.")
        return

    coords.sort(key=lambda x: x[0][axis])

    total = len(coords)
    rows = ceil(total / cols)

    print(f"Generating linear grid sorted by {axis}: {cols} cols × {rows} rows")

    thumb_w = int(width * thumb_scale)
    thumb_h = int(height * thumb_scale)

    canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")

    csv_rows = [["" for _ in range(cols)] for _ in range(rows)]

    for idx, (meta, fname) in enumerate(coords):

        r = idx // cols
        c = idx % cols

        img = Image.open(os.path.join(folder, fname))
        img = img.resize((thumb_w, thumb_h))
        canvas.paste(img, (c * thumb_w, r * thumb_h))

        csv_rows[r][c] = fname

    filename = f"grid_linear_{axis}"
    canvas.save(os.path.join(folder, f"{filename}.png"))
    canvas.save(os.path.join(folder, f"{filename}.pdf"), "PDF", resolution=300.0)

    with open(os.path.join(folder, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("✅ Linear grid saved")
    