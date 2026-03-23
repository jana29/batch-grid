import os
import csv
from PIL import Image
from math import ceil
from datetime import datetime


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def parse_filename(fname):
    """
    Extract experiment indices from filename.
    Expected format:
    {seed}_{i_intro}_{i_beauty}_{i_object}_{i_style}_{i_manipulation_type}_{Manipulation}_{steps}_{cfg}.png
    """
    parts = fname.replace(".png", "").split("_")

    try:
        return {
            "seed": int(parts[0]),
            "intro": int(parts[1]),
            "beauty": int(parts[2]),
            "object": int(parts[3]),
            "style": int(parts[4]),
            "manipulation_type": int(parts[5]),
            "manipulation": float(parts[6]),
            "steps": int(parts[7]),
            "cfg": float(parts[8]),
        }
    except:
        return None


def list_images(folder):
    return [
        f for f in os.listdir(folder)
        if f.endswith(".png") and f.split("_")[0].isdigit()
    ]


# ------------------------------------------------------------
# GRID TYPE 1
# A x B experiment comparison grid
# ------------------------------------------------------------

def generate_ab_grid(
    folder,
    row_axis,          # "seed", "intro", "beauty", "object", "style"
    col_axis,
    width,
    height,
    thumb_scale=0.5
):
    files = list_images(folder)

    coords = []
    for f in files:
        meta = parse_filename(f)
        if meta:
            coords.append((meta, f))

    if not coords:
        print("No valid images found.")
        return

    # unique sorted axis values
    row_values = sorted(set(m[row_axis] for m, _ in coords))
    col_values = sorted(set(m[col_axis] for m, _ in coords))

    rows = len(row_values)
    cols = len(col_values)

    print(f"Generating A×B grid: rows={row_axis} ({rows}) cols={col_axis} ({cols})")

    thumb_w = int(width * thumb_scale)
    thumb_h = int(height * thumb_scale)

    canvas = Image.new(
        "RGB",
        (cols * thumb_w, rows * thumb_h),
        "white"
    )

    csv_rows = [["" for _ in range(cols)] for _ in range(rows)]

    # build lookup
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

    filename=f"grid_{row_axis}_x_{col_axis}"
    canvas.save(os.path.join(folder, f"{filename}.png"))
    canvas.save(os.path.join(folder, f"{filename}.pdf"), "PDF", resolution=300.0)

    with open(os.path.join(folder, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("✅ A×B grid saved")


# ------------------------------------------------------------
# GRID TYPE 2
# Linear grid (single varying variable)
# ------------------------------------------------------------

def generate_linear_grid(
    folder,
    axis,          # "seed", "intro", "beauty", ...
    cols,
    width,
    height,
    thumb_scale=0.5
):
    files = list_images(folder)

    coords = []
    for f in files:
        meta = parse_filename(f)
        if meta:
            coords.append((meta, f))

    if not coords:
        print("No valid images found.")
        return

    # sort by selected axis
    coords.sort(key=lambda x: x[0][axis])

    total = len(coords)
    rows = ceil(total / cols)

    print(f"Generating linear grid sorted by {axis}: {cols} cols × {rows} rows")

    thumb_w = int(width * thumb_scale)
    thumb_h = int(height * thumb_scale)

    canvas = Image.new(
        "RGB",
        (cols * thumb_w, rows * thumb_h),
        "white"
    )

    csv_rows = [["" for _ in range(cols)] for _ in range(rows)]

    for idx, (meta, fname) in enumerate(coords):
        r = idx // cols
        c = idx % cols

        img = Image.open(os.path.join(folder, fname))
        img = img.resize((thumb_w, thumb_h))
        canvas.paste(img, (c * thumb_w, r * thumb_h))

        csv_rows[r][c] = fname

    filename=f"grid_linear_{axis}"
    canvas.save(os.path.join(folder, f"{filename}.png"))
    canvas.save(os.path.join(folder, f"{filename}.pdf"), "PDF", resolution=300.0)

    with open(os.path.join(folder, f"{filename}.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print("✅ Linear grid saved")


# ------------------------------------------------------------
# settings report txt file
# ------------------------------------------------------------

def write_run_report(
    folder,
    row_axis, col_axis, num_cols,
    seeds, intros, beauties, objects, styles,
    total,
    negative_prompt,
    manipulation_type, manipulation,
    steps, cfg,
    width, height
):

    settings_path = f"{folder}/000_settings_log.txt"

    with open(settings_path, "w") as f:

        f.write("=============== EXPERIMENT REPORT ===============\n\n")

        f.write(f"folder: {folder}\n")
        f.write(f"timestamp: {datetime.now()}\n\n")

        # ---------- SUMMARY ----------
        f.write("=== SUMMARY ===\n")
        f.write(f"total images: {total}\n")
        f.write(f"grid: rows={row_axis}, cols={col_axis}\n")

        # ---------- GRID OVERVIEW ----------
        axis_map = {
            "seed": seeds,
            "intro": [i for i,_ in intros],
            "beauty": [i for i,_ in beauties],
            "object": [i for i,_ in objects],
            "style": [i for i,_ in styles],
            "steps": [steps],
            "cfg": [cfg],
        }
        f.write("\nGRID OVERVIEW\n")

        # a x b grid
        if row_axis is not None and col_axis is not None:       
            row_vals = sorted(set(axis_map.get(row_axis, [])))
            col_vals = sorted(set(axis_map.get(col_axis, [])))
            
            f.write(f"{row_axis} \\ {col_axis}\n")
            
            f.write("             " + " ".join(f"{c:>6}" for c in col_vals) + "\n")

            for rv in row_vals:
                line = f"{rv:>4} | "
                for _ in col_vals:
                    line += "  X  "
                f.write(line + "\n")
        
        # linear grid
        else:
            if row_axis is None:
                vals=sorted(axis_map.get(col_axis, []))
                parameter=col_axis
            elif col_axis is None:
                vals=sorted(axis_map.get(row_axis, []))
                parameter=row_axis
            
            num_rows = math.ceil(len(vals) / num_cols)
            f.write(f"{parameter} in {num_cols}x{num_rows} grid")

            for i, v in enumerate(vals):
                if i % num_cols == 0:
                    f.write("\n")
                f.write(f"{str(v):>6} ")

            f.write("\n")




        f.write("\n")

        # ---------- PROMPT COMPONENTS ----------
        f.write("=== PROMPT COMPONENTS ===\n\n")

        f.write("SEEDS\n")
        for s in seeds:
            f.write(f"  {s}\n")
        f.write("\n")

        def write_component(name, data):
            f.write(f"{name}\n")
            for idx, text in data:
                f.write(f"  {idx}: {text}\n")
            f.write("\n"

        write_component("INTRO", intros)
        write_component("BEAUTY", beauties)
        write_component("OBJECT", objects)
        write_component("STYLE", styles)

        # ---------- MANIPULATION ----------
        f.write("=== MANIPULATION ===\n")
        if manipulation_type == "NONE":
            f.write("manipulation: NONE\n\n")
        else:
            # adapt later
            f.write(f"type: {manipulation_type} {manipulation}\n\n")

        # ---------- SAMPLING ----------
        f.write("=== SAMPLING ===\n")
        f.write(f"steps: {steps}\n")
        f.write(f"cfg: {cfg}\n")
        f.write(f"image_size: {width} x {height}\n\n")

        f.write(f"negative_prompt: {negative_prompt}\n")

    print("✅ settings report written")