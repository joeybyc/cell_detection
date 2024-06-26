import cv2
import numpy as np
import pandas as pd



# Function to load ground truth data from CSV
def load_ground_truth(csv_file):
    """
    Loads ground truth data from a CSV file. The CSV should contain columns for image name, cell annotations, 
    and the dimensions of the images.

    Parameters:
    - csv_file (str): Path to the CSV file containing ground truth data.

    Returns:
    - dict: A dictionary where each key is an image name and the value is a dictionary containing the cell annotations 
            and the dimensions of the image.
    """
    df = pd.read_csv(csv_file)
    ground_truth = {}
    for _, row in df.iterrows():
        image_name = row['image_name']
        cell_annotation = eval(row['cell_annotation'])
        ground_truth[image_name] = {
            'cell_annotation': cell_annotation,
            'image_width': row['image_width'],
            'image_height': row['image_height']
        }
    return ground_truth

# Function to convert cell centroid locations to match the image size
def convert_cell_location(cell_coords, original_width, original_height, image_width, image_height):
    """
    Converts cell centroid locations from original image dimensions to the dimensions of a processed image.

    Parameters:
    - cell_coords (list of tuples): List of (x, y) coordinates of cell centroids.
    - original_width (int): Width of the original image.
    - original_height (int): Height of the original image.
    - image_width (int): Width of the processed image.
    - image_height (int): Height of the processed image.

    Returns:
    - list of tuples: Adjusted list of (x, y) coordinates of cell centroids according to the processed image dimensions.
    """
    scale_x = image_width / original_width
    scale_y = image_height / original_height
    return [(x * scale_x, y * scale_y) for x, y in cell_coords]

# Strict Evaluation for Precision
def get_precision_strict_eval(mask, ground_truth_centroids, return_details=False):
    """
    Calculates the strict precision of the segmentation model. Precision is defined as the ratio of true positives 
    (predicted centroids that correctly identify ground truth centroids) to the total number of predicted objects.

    Parameters:
    - mask (numpy.ndarray): Binary mask image where non-zero pixels represent detected objects.
    - ground_truth_centroids (list of tuples): List of ground truth centroids (x, y).
    - return_details (bool, optional): Flag to indicate if the function should return detailed results. 
                                       Default is False.

    Returns:
    - float: The precision of the model.
    - int: The number of true positives (only if return_details is True).
    - int: The total number of predicted objects (only if return_details is True).
    """
    num_labels, labels = cv2.connectedComponents(mask)
    true_positives = 0

    for i in range(1, num_labels):  # Start from 1 to exclude the background
        object_mask = (labels == i)
        
        for gt_centroid in ground_truth_centroids:
            x, y = int(gt_centroid[0]), int(gt_centroid[1])
            if object_mask[y, x]:
                true_positives += 1
                break  # Move to next object once a match is found

    precision = true_positives / (num_labels - 1) if num_labels > 1 else 1

    if return_details:
        return precision, true_positives, num_labels - 1
    else:
        return precision

# Strict Evaluation for Recall
def get_recall_strict_eval(mask, ground_truth_centroids, return_details=False):
    """
    Calculates the strict recall of the segmentation model. Recall is defined as the ratio of true positives 
    (ground truth centroids correctly identified by predicted centroids) to the total number of ground truth objects.

    Parameters:
    - mask (numpy.ndarray): Binary mask image where non-zero pixels represent detected objects.
    - ground_truth_centroids (list of tuples): List of ground truth centroids (x, y).
    - return_details (bool, optional): Flag to indicate if the function should return detailed results. 
                                       Default is False.

    Returns:
    - float: The recall of the model.
    - int: The number of true positives (only if return_details is True).
    - int: The total number of ground truth objects (only if return_details is True).
    """
    true_positives = 0

    for gt_centroid in ground_truth_centroids:
        x, y = int(gt_centroid[0]), int(gt_centroid[1])
        if mask[y, x]:
            true_positives += 1

    recall = true_positives / len(ground_truth_centroids) if ground_truth_centroids else 1

    if return_details:
        return recall, true_positives, len(ground_truth_centroids)
    else:
        return recall

# Function to process mask image and find object centroids
def process_mask_image(mask_image):
    """
    Processes a mask image to find centroids of detected objects. Assumes that the mask image is a binary image 
    where objects are represented by non-zero pixels.

    Parameters:
    - mask_image (numpy.ndarray): The mask image to be processed.

    Returns:
    - list of tuples: A list of centroids (x, y) of the detected objects.
    """
    num_labels, labels = cv2.connectedComponents(mask_image)
    centroids = []
    for i in range(1, num_labels):  # Start from 1 to exclude the background
        ys, xs = np.where(labels == i)
        centroid_x = np.mean(xs)
        centroid_y = np.mean(ys)
        centroids.append((centroid_x, centroid_y))
    return centroids

# Tolerated Evaluation for Precision
def get_precision_tolerated_eval(predicted_centroids, ground_truth_centroids, L, return_details=False):
    """
    Calculates the tolerated precision of the segmentation model, allowing for some leniency in matching predicted 
    centroids with ground truth. Precision is defined as the ratio of true positives to the total number of predicted objects.

    Parameters:
    - predicted_centroids (list of tuples): List of centroids (x, y) predicted by the model.
    - ground_truth_centroids (list of tuples): List of ground truth centroids (x, y).
    - L (int): Tolerance parameter, defining the size of the square around the predicted centroid for overlap checking.
    - return_details (bool, optional): Flag to indicate if the function should return detailed results. 
                                       Default is False.

    Returns:
    - float: The tolerated precision of the model.
    - int: The number of true positives (only if return_details is True).
    - int: The total number of predicted objects (only if return_details is True).
    """
    true_positives = 0

    def check_overlap(predicted, ground_truth, L):
        x_min, x_max = predicted[0] - L/2, predicted[0] + L/2
        y_min, y_max = predicted[1] - L/2, predicted[1] + L/2
        return x_min <= ground_truth[0] <= x_max and y_min <= ground_truth[1] <= y_max

    for pred_centroid in predicted_centroids:
        if any(check_overlap(pred_centroid, gt_centroid, L) for gt_centroid in ground_truth_centroids):
            true_positives += 1

    precision = true_positives / len(predicted_centroids) if predicted_centroids else 1

    if return_details:
        return precision, true_positives, len(predicted_centroids)
    else:
        return precision

# Tolerated Evaluation for Recall
def get_recall_tolerated_eval(predicted_centroids, ground_truth_centroids, L, return_details=False):
    """
    Calculates the tolerated recall of the segmentation model, allowing for some leniency in matching predicted 
    centroids with ground truth. Recall is defined as the ratio of true positives to the total number of ground truth objects.

    Parameters:
    - predicted_centroids (list of tuples): List of centroids (x, y) predicted by the model.
    - ground_truth_centroids (list of tuples): List of ground truth centroids (x, y).
    - L (int): Tolerance parameter, defining the size of the square around the predicted centroid for overlap checking.
    - return_details (bool, optional): Flag to indicate if the function should return detailed results. 
                                       Default is False.

    Returns:
    - float: The tolerated recall of the model.
    - int: The number of true positives (only if return_details is True).
    - int: The total number of ground truth objects (only if return_details is True).
    """
    true_positives = 0

    def check_overlap(predicted, ground_truth, L):
        x_min, x_max = predicted[0] - L/2, predicted[0] + L/2
        y_min, y_max = predicted[1] - L/2, predicted[1] + L/2
        return x_min <= ground_truth[0] <= x_max and y_min <= ground_truth[1] <= y_max

    for gt_centroid in ground_truth_centroids:
        if any(check_overlap(pred_centroid, gt_centroid, L) for pred_centroid in predicted_centroids):
            true_positives += 1

    recall = true_positives / len(ground_truth_centroids) if ground_truth_centroids else 1

    if return_details:
        return recall, true_positives, len(ground_truth_centroids)
    else:
        return recall

# Function to calculate F1 score
def get_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)
