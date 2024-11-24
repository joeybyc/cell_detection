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
from lib.common.drawing import draw_square, overlay_transparent_mask
from torchvision import transforms
import torch
import cv2
from PIL import Image
import numpy as np


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


def extract_masked_region(image_path, mask_path, output_path=None):
    """
    Extract the region of interest from an image using a binary mask and add transparency.
    
    Args:
        image_path (str): Path to the source image (can be PIL Image or path)
        mask_path (str): Path to the binary mask image (can be PIL Image or path)
        output_path (str, optional): Path to save the output image. If None, returns the image object
        
    Returns:
        If output_path is None, returns PIL Image object
        If output_path is provided, saves the image and returns None
    """
    # Load images
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    
    if isinstance(mask_path, str):
        mask = Image.open(mask_path)
    else:
        mask = mask_path

    # Convert mask to binary if it isn't already
    mask = np.array(mask)
    if len(mask.shape) > 2:
        mask = mask[:,:,0]  # Take first channel if multichannel
    
    # Convert to binary
    mask = (mask > 0).astype(np.uint8) * 255
    
    # Convert source image to RGBA if it isn't already
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Convert to numpy arrays for processing
    image_array = np.array(image)
    
    # Create alpha channel from mask
    alpha_channel = mask
    
    # Create new RGBA image
    result = np.zeros((image_array.shape[0], image_array.shape[1], 4), dtype=np.uint8)
    result[:,:,:3] = image_array[:,:,:3]
    result[:,:,3] = alpha_channel
    
    # Convert back to PIL Image
    result_image = Image.fromarray(result)
    
    if output_path:
        result_image.save(output_path, format='PNG')
        return None
    return result_image


if __name__ == "__main__":
    image_name = 'example2'
    image_ext = 'png'
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
    print(f"Loading SAM ...")
    # SAM for AC area segmentation
    predictor = load_sam_predictor(sam_checkpoint = sam_checkpoint, model_type=model_type)

    # load cell classifier
    model = Net(size_img=L)  # Initialize model
    # Load the model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Detecting Cell ...")
    results = asoct_cell_detection(image_path, predictor, model, transform, params)

    chamber_mask = results['chamber_mask']
    mask_image_anterior_segment = results['mask_image_anterior_segment']
    point_prompts = results['point_prompts']
    chamber_mask_candidates = results['chamber_mask_candidates']
    candidate_cell_mask = results['candidate_cell_mask']
    cell_mask = results['cell_mask']
    cell_centroids = get_centroids(cell_mask)
    image_with_cell_boxes = draw_square(image_path, cell_centroids, cropped_output_folder=f'{output_folder}/image_with_cell_bounding_boxes')
    candidate_cell_centroids = get_centroids(candidate_cell_mask)
    image_with_candidate_cell_boxes = draw_square(image_path, candidate_cell_centroids, cropped_output_folder=f'{output_folder}/image_with_candidate_cell_bounding_boxes')




    # Save the masks as an image file
    cv2.imwrite(f'{output_folder}/chamber_mask.png', chamber_mask)
    cv2.imwrite(f'{output_folder}/mask_image_anterior_segment.png', mask_image_anterior_segment)
    cv2.imwrite(f'{output_folder}/chamber_mask_candidates.png', chamber_mask_candidates)
    cv2.imwrite(f'{output_folder}/candidate_cell_mask.png', candidate_cell_mask)
    cv2.imwrite(f'{output_folder}/cell_mask.png', cell_mask)
    image_with_cell_boxes.save(f'{output_folder}/image_with_cell_bounding_boxes.png')
    image_with_candidate_cell_boxes.save(f'{output_folder}/image_with_candidate_cell_bounding_boxes.png')

    # Extract masked regions with transparency
    chamber_mask_path = f'{output_folder}/chamber_mask.png'
    
    # Process original image
    extract_masked_region(
        image_path,
        chamber_mask_path,
        f'{output_folder}/original_masked.png'
    )
    
    # Process image with cell bounding boxes
    extract_masked_region(
        f'{output_folder}/image_with_cell_bounding_boxes.png',
        chamber_mask_path,
        f'{output_folder}/cell_boxes_masked.png'
    )
    
    # Process image with candidate cell bounding boxes
    extract_masked_region(
        f'{output_folder}/image_with_candidate_cell_bounding_boxes.png',
        chamber_mask_path,
        f'{output_folder}/candidate_boxes_masked.png'
    )

    # Create a semi-transparent blue overlay of the chamber mask
    overlay_transparent_mask(
        image_path,
        chamber_mask,
        f'{output_folder}/image_with_blue_overlay.png',
        color=(255, 255, 0),  # Blue in BGR format
        alpha=0.5          # 30% opacity
    )
