# --- image batch settings ---

OUTPUT_DIR = "./output/steps_cfg_test2"

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

NUM_COLS=7


# -----------------------------------------
# --- run generation of images + grid ––––
if __name__ == "__main__":

    """
    from batch_generate_sdxl import batch_generate_from_files
    batch_generate_from_files()
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

    from batch_generate_sdxl import load_pipeline
    pipeline = load_pipeline()
    from batch_generate_sdxl import batch_generate
    
    steps_values = range(1,20,2)
    cfg_values = range(1,32,5)

    total_runs = len(steps_values) * len(cfg_values)
    run_index = 0

    for steps in steps_values:
        for cfg in cfg_values:
            run_index += 1
            print(f"\nRun {run_index} / {total_runs}")
            batch_generate( 
                OUTPUT_DIR,
                pipeline,
        
                [908172635401928, 639201847561029],
                
                ["a portrait of a"],
                ["beautiful"],
                ["person"],
                ["professional photography"],
        
                NEGATIVE_PROMPT,
                steps,
                cfg,
                WIDTH,
                HEIGHT,
    
                f"_steps:{steps:02d}_cfg:{cfg:02d}", # add to filename
        
                COMMENT
            )
    
    
    from makeGrid import generate_grid
    generate_grid()

    import shutil
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"{OUTPUT_DIR}_{ts}"
    shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)
    print(f"✅ Zip created: {zip_path}.zip")