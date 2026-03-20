# --- image batch settings ---

OUTPUT_DIR = "./output/default"

SEEDS_PATH="prompt/00_seed.txt"

INTRO_PATH="prompt/01_intro.txt"
BEAUTY_PATH="prompt/02_beauty.txt"
OBJECT_PATH="prompt/03_object.txt"
STYLE_PATH="prompt/04_style.txt"

NEGATIVE_PROMPT="watermark, text, picture frame, face card, multiple faces"

GEN_STEPS=20
GEN_GUIDANCESCALE=7

HEIGHT = 744
WIDTH = 512

AMOUNT=20

COMMENT="GEN_STEPS x GUIDANCESCALE"

# --- grid settings ----

NUM_COLS=6


# -----------------------------------------
# --- run generation of images + grid ––––
if __name__ == "__main__":
    from batch_generate_sdxl import batch_generate_from_files
    batch_generate_from_files()
    """
    batch_generate_from_files( 
        OUTPUT_DIR,

        SEEDS_PATH,
        
        INTRO_PATH,
        BEAUTY_PATH,
        OBJECT_PATH,
        STYLE_PATH,

        NEGATIVE_PROMPT,
        GEN_STEPS,
        GEN_GUIDANCESCALE,
        WIDTH,
        HEIGHT
    )
    batch_generate( 
        OUTPUT_DIR,

        SEEDS_PATH,
        
        INTRO_PATH,
        BEAUTY_PATH,
        OBJECT_PATH,
        STYLE_PATH,

        NEGATIVE_PROMPT,
        GEN_STEPS,
        GEN_GUIDANCESCALE,
        WIDTH,
        HEIGHT,

        COMMENT
    )
    """
    
    from makeGrid import generate_grid
    generate_grid()