import os
import shutil
import glob
import random

def construct_dataset(original_data_folder, dataset_folder, ratio):
    """
    Constructs datasets for training, validation, and testing based on a specified ratio.

    Parameters:
    - original_data_folder (str): Path to the folder containing original images.
    - dataset_folder (str): Path to the folder where the dataset will be constructed.
    - ratio (tuple): A tuple indicating the proportion of train, val, and test datasets (e.g., (4, 1, 5)).
    """
    # Ensure dataset_folder exists
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    # Create subfolders for train, val, and test
    for subfolder in ['train', 'val', 'test']:
        subfolder_path = os.path.join(dataset_folder, subfolder)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path)

    # Load a list of image file paths
    image_paths = glob.glob(os.path.join(original_data_folder, '*'))

    # Shuffle the list
    random.shuffle(image_paths)

    # Calculate the proportions for each dataset part
    total_ratio = sum(ratio)  # Total of the ratio values
    train_prop, val_prop, test_prop = [r / total_ratio for r in ratio]

    # Calculate the split indices
    total_images = len(image_paths)
    train_end = int(total_images * train_prop)
    val_end = train_end + int(total_images * val_prop)

    # Split the list
    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:val_end]
    test_images = image_paths[val_end:]

    # Function to copy images to their respective folders
    def copy_images(image_paths, target_folder):
        for path in image_paths:
            shutil.copy(path, os.path.join(dataset_folder, target_folder, os.path.basename(path)))
        print(f'{target_folder} finished!')

    # Copy images to their respective datasets
    copy_images(train_images, 'train')
    copy_images(val_images, 'val')
    copy_images(test_images, 'test')

    print(f"Dataset construction completed: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images.")

def copy_images_to_exp_dataset_cell(source_folder, exp_dataset_folder, exp_dataset_cell_folder):
    """
    Copies images from a source folder to the exp_dataset_cell_folder based on the name in exp_dataset.
    For example.
    In the source_folder, there are a list of image that are binary mask of the cells
    In the exp_dataset_folder, which has three sub-folders: train, test, valid. They have the original OCT image.
    This function will copy the cell mask image into the exp_dataset_cell_folder, The name of cell mask image in each sub folder are same as exp_dataset_folder.

    Parameters:
    - source_folder (str): The folder where the original cell images are stored (e.g., 'cell_1_25_90').
    - exp_dataset_folder (str): The folder of the existing dataset (e.g., 'exp_dataset'). It has three sub-folders train, val, test
    - exp_dataset_cell_folder (str): The folder where the new cell dataset will be created (e.g., 'exp_dataset_cell'). It has three sub-folders train, val, test
    """
    # Ensure exp_dataset_cell_folder exists
    if not os.path.exists(exp_dataset_cell_folder):
        os.makedirs(exp_dataset_cell_folder)

    # Iterate over train, val, and test folders
    for subfolder in ['train', 'val', 'test']:
        source_subfolder_path = os.path.join(exp_dataset_folder, subfolder)
        target_subfolder_path = os.path.join(exp_dataset_cell_folder, subfolder)

        # Ensure target subfolders exist
        if not os.path.exists(target_subfolder_path):
            os.makedirs(target_subfolder_path)

        # List all images in the exp_dataset subfolder
        for image_name in os.listdir(source_subfolder_path):
            # Construct the source and destination paths
            source_image_path = os.path.join(source_folder, image_name)
            destination_image_path = os.path.join(target_subfolder_path, image_name)

            # Check if the image exists in the source folder
            if os.path.exists(source_image_path):
                # Copy the image to the destination
                shutil.copy(source_image_path, destination_image_path)
            else:
                print(f"Image {image_name} not found in {source_folder}.")
