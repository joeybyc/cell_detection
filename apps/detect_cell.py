import os


from lib.model.sam_model import load_sam_predictor
from lib.image_processing.loading import load_image
from lib.chember_segmentation.sam_detection import extract_chamber_mask
from lib.image_processing.thresholding import isodata_threshold
from lib.image_processing.mask import merge_masks, filter_objects_by_size, extract_objects_by_area, get_centroids
from lib.cell_detection.filter import filter_candidate_cells
from lib.model.ann_cell_identification import Net
import argparse
from lib.common.config import cfg
from lib.common.drawing import draw_square
from torchvision import transforms
import torch
import cv2



def asoct_cell_detection(image_path, predictor, model, transform, params):
    threshold_alpha = params["threshold_alpha"]
    lower_bound = params["lower_bound"]
    upper_bound = params["upper_bound"]
    device = params["device"]

    # Load image
    image = load_image(image_path)
    # Extract chamber masks and related information from the image using the predictor
    chamber_mask, mask_image_anterior_segment, point_prompts, prompt_labels, chamber_mask_candidates = extract_chamber_mask(image, predictor, return_all=True)

    # Apply isodata thresholding to the image and also return the threshold value
    image_threshold, _ = isodata_threshold(image, return_thresh=True, alpha=threshold_alpha)

    merge_image_mask = merge_masks(mask_image_anterior_segment, image_threshold)
    # Filter objects in the merged mask by size to remove cornea-related regions
    detected_objects = filter_objects_by_size(merge_image_mask, lower_bound=lower_bound, upper_bound=upper_bound)
    # Extract the final cell mask by considering only objects within the chamber mask area
    candidate_cell_mask = extract_objects_by_area(chamber_mask, detected_objects)
    cell_mask = filter_candidate_cells(image, candidate_cell_mask, model, device, transform=transform)

    results = {
        'chamber_mask': chamber_mask,
        'mask_image_anterior_segment': mask_image_anterior_segment,
        'point_prompts': point_prompts,
        'chamber_mask_candidates': chamber_mask_candidates,
        'candidate_cell_mask': candidate_cell_mask,
        'cell_mask': cell_mask
    }
    return results

if __name__ == "__main__":
    image_name = 'example1'
    image_ext = 'jpeg'
    image_folder = f'data/example'

    output_folder = f'data/output/{image_name}'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_path = f'{image_folder}/{image_name}.{image_ext}'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, default="./configs/default_settings.yaml")
    args = parser.parse_args()
    cfg.merge_from_file(args.config)
    sam_checkpoint = cfg.sam_model.ckpt
    model_type = cfg.sam_model.model_type
    model_path = cfg.cell_classifier.ckpt
    # Check if CUDA is available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    L = cfg.cell_classifier.size_L
    transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((L, L)), transforms.ToTensor()])

    params = {
        "threshold_alpha": cfg.infer_threshold.threshold_alpha,
        "lower_bound": cfg.infer_threshold.lower_bound,
        "upper_bound": cfg.infer_threshold.upper_bound,
        "device": device
    }
    print(f"The device is {device}")

    # SAM for AC area segmentation
    predictor = load_sam_predictor(sam_checkpoint = sam_checkpoint, model_type=model_type)

    # load cell classifier
    model = Net(size_img=L)  # Initialize model
    # Load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = asoct_cell_detection(image_path, predictor, model, transform, params)

    chamber_mask = results['chamber_mask']
    mask_image_anterior_segment = results['mask_image_anterior_segment']
    point_prompts = results['point_prompts']
    chamber_mask_candidates = results['chamber_mask_candidates']
    candidate_cell_mask = results['candidate_cell_mask']
    cell_mask = results['cell_mask']
    cell_centroids = get_centroids(cell_mask)
    image_with_cell_boxes = draw_square(image_path, cell_centroids)
    candidate_cell_centroids = get_centroids(candidate_cell_mask)
    image_with_candidate_cell_boxes = draw_square(image_path, candidate_cell_centroids)

    # Save the masks as an image file
    cv2.imwrite(f'{output_folder}/chamber_mask.png', chamber_mask)
    cv2.imwrite(f'{output_folder}/mask_image_anterior_segment.png', mask_image_anterior_segment)
    cv2.imwrite(f'{output_folder}/chamber_mask_candidates.png', chamber_mask_candidates)
    cv2.imwrite(f'{output_folder}/candidate_cell_mask.png', candidate_cell_mask)
    cv2.imwrite(f'{output_folder}/cell_mask.png', cell_mask)
    image_with_cell_boxes.save(f'{output_folder}/image_with_cell_bounding_boxes.png')
    image_with_candidate_cell_boxes.save(f'{output_folder}/image_with_candidate_cell_bounding_boxes.png')







