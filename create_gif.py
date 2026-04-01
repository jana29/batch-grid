import re
from pathlib import Path
import imageio.v2 as imageio

BASE_DIR = Path("./output")
FPS = 20
FRAME_DURATION = 1 / FPS

# numeric extraction for natural sorting
number_pattern = re.compile(r"\d+(?:\.\d+)?")

def natural_key(path):
    nums = number_pattern.findall(path.stem)
    return [float(n) if "." in n else int(n) for n in nums]


def select_folder():
    folders = sorted([p for p in BASE_DIR.iterdir() if p.is_dir()])

    if not folders:
        print("No subfolders found in ./output")
        exit()

    print("\nAvailable folders:\n")

    for i, folder in enumerate(folders, 1):
        print(f"{i}: {folder.name}")

    while True:
        choice = input("\nSelect folder number: ")

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(folders):
                return folders[idx]

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

    # load frames
    frames = [imageio.imread(img) for img in images]

    print("Writing GIF...")
    imageio.mimsave(gif_path, frames, duration=FRAME_DURATION)

    print("Writing MP4...")
    with imageio.get_writer(mp4_path, fps=FPS) as writer:
        for frame in frames:
            writer.append_data(frame)

    print("\nDone!")
    print("GIF:", gif_path)
    print("MP4:", mp4_path)


def main():
    folder = select_folder()
    print(f"\nSelected: {folder.name}")

    images = collect_images(folder)

    if not images:
        print("No images found")
        return

    create_outputs(images, folder)


if __name__ == "__main__":
    main()