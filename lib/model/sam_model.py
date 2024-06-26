from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

def load_sam_predictor(sam_checkpoint = "lib/model/sam_vit_h_4b8939.pth", model_type = "vit_h", device = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # print(f"The device is {device}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor
