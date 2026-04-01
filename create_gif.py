import re
from pathlib import Path
import imageio.v2 as imageio

BASE_DIR = Path("./output")
FPS = 20
FRAME_DURATION = 1 / FPS

number_pattern = re.compile(r"\d+(?:\.\d+)?")

def natural_key(path):
    nums = number_pattern.findall(path.stem)
    return [float(n) if "." in n else int(n) for n in nums]


def menu_select(folders, current_path):
    print(f"\nCurrent folder: {current_path}\n")

    for i, f in enumerate(folders, 1):
        print(f"{i}: {f.name}")

    print("0: Go up")

    while True:
        choice = input("\nSelect folder: ")

        if choice.isdigit():
            idx = int(choice)

            if idx == 0:
                return None

            if 1 <= idx <= len(folders):
                return folders[idx - 1]

        print("Invalid selection")


def collect_images(folder):
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")

    images = [
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
        and p.name != "grid.png"
    ]

    images.sort(key=natural_key)
    return images


def create_outputs(images, folder):

    gif_path = folder / f"{folder.name}.gif"
    mp4_path = folder / f"{folder.name}.mp4"

    print(f"\nProcessing {len(images)} images...")

    frames = [imageio.imread(p) for p in images]

    print("Writing GIF...")
    imageio.mimsave(gif_path, frames, duration=FRAME_DURATION)

    print("Writing MP4...")
    with imageio.get_writer(mp4_path, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)

    print("\nDone!")
    print("GIF:", gif_path)
    print("MP4:", mp4_path)


def navigate(start_dir):

    current = start_dir

    while True:

        subfolders = sorted([p for p in current.iterdir() if p.is_dir()])
        images = collect_images(current)

        if images:
            print(f"\nImages detected in: {current}")
            confirm = input("Generate GIF/MP4 here? (y/n): ").lower()

            if confirm == "y":
                create_outputs(images, current)
                return

        if not subfolders:
            print("No subfolders here.")
            return

        selection = menu_select(subfolders, current)

        if selection is None:
            current = current.parent
            if current == start_dir.parent:
                return
        else:
            current = selection


if __name__ == "__main__":
    navigate(BASE_DIR)