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
# --------------- MODES  –----------------–––

def steps_X_cfg():
    # --- get images ---
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
    
                f"_steps:{steps:02d}_cfg:{cfg:02d}", # add suffix to filename
        
                COMMENT
            )

    NUM_COLS = len(cfg_values)
   
    # --- grid ---
    from makeGrid import generate_grid
    generate_grid(NUM_COLS)

    # --- zip folder ---
    import shutil
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_path = f"{OUTPUT_DIR}_{ts}"
    shutil.make_archive(zip_path, 'zip', OUTPUT_DIR)
    print(f"✅ Zip created: {zip_path}.zip")

def get_images_default():
    from batch_generate_sdxl import batch_generate_from_files
    #batch_generate_from_files()
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

# --------- embedding modes ------------
def embedding_scale_mode():

    from batch_generate_sdxl import load_pipeline
    pipe = load_pipeline()

    from generators.embedding_experiments import generate_embedding_scale_grid

    scales = [-1.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    generate_embedding_scale_grid(
        pipe=pipe,
        seeds=[908172635401928, 639201847561029],
        beauty=["beautiful", "ugly"],
        objects=["person"],
        styles=["professional photography"],
        scales=scales,
        negative_prompt=NEGATIVE_PROMPT,
        steps=30,
        cfg=7,
        width=WIDTH,
        height=HEIGHT,
        output_dir=OUTPUT_DIR
    )

    from makeGrid import generate_grid
    generate_grid(len(scales))

    def embedding_interpolation_mode():

    from batch_generate_sdxl import load_pipeline
    pipe = load_pipeline()

    from generators.embedding_experiments import interpolate_embeddings

    interpolate_embeddings(
        pipe,
        prompt_a="a portrait of a beautiful person",
        prompt_b="a portrait of an ugly person",
        steps_list=[0,0.25,0.5,0.75,1],
        negative_prompt=NEGATIVE_PROMPT,
        seeds=[12345],
        steps=30,
        cfg=7,
        width=WIDTH,
        height=HEIGHT,
        output_dir=OUTPUT_DIR
    )


# -----------------------------------------
# --------------- RUN  –----------------–––

if __name__ == "__main__":

    #steps_X_cfg()
    #get_images_default()
    embedding_scale_mode()



