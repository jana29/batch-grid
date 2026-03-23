import torch
import clip
from PIL import Image
import numpy as np


class AestheticScorer:

    def __init__(self, device="cuda"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=device)

        # pretrained linear head weights (public aesthetic model)
        self.linear = torch.nn.Linear(768, 1)
        self.linear.load_state_dict(torch.load("aesthetic_head.pt"))

        self.model.eval()
        self.linear.eval()

    def score(self, image_path):

        img = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model.encode_image(img)
            features = features / features.norm(dim=-1, keepdim=True)
            score = self.linear(features)

        return float(score.item())


"""
## --- call in other file: ----

scores = []

for fname in images:
    s = scorer.score(fname)
    scores.append((fname, s))

scores.sort(key=lambda x: x[1], reverse=True)

## should give smth like: "BEST IMAGE = cfg 6.5 steps 24 seed 908172..."

"""