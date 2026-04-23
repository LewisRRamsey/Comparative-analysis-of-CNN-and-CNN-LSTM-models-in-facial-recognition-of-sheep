import os
import shutil
import random

# function for moving dataset files from their original folder to one in either train, validation or test folders, keeping them in their respective class folders
def file_moving_function(source_folder, destination_folders, probabilities = [0.6, 0.1, 0.3]):
    # creating destination folders if they do not already exist
    for folder in destination_folders:
        os.makedirs(folder, exist_ok=True)

    # getting a list of all the images in the source folder
    images = [f for f in os.listdir(source_folder)]

    # randomly moving each image to one of the destination folders
    for image in images:
        source_path = os.path.join(source_folder, image)
        destination_folder = random.choices(destination_folders, weights=probabilities, k=1)[0]
        destination_path = os.path.join(destination_folder, image)
        shutil.copy2(source_path, destination_path)

# iterating over all classes for the three breeds with 8 classes each in the dataset
for sheep_type in ["mc", "su", "ws"]:
    for class_index in range(1, 8):
        # setting source and destination folders for moving files
        source_folder = f"C:/University/Year 3 project/Project files/Proper dataset/{sheep_type}sheep{class_index}"
        destination_folders = [
            f"C:/University/Year 3 project/Project files/Proper working dataset/train/{sheep_type}sheep{class_index}",
            f"C:/University/Year 3 project/Project files/Proper working dataset/validation/{sheep_type}sheep{class_index}",
            f"C:/University/Year 3 project/Project files/Proper working dataset/test/{sheep_type}sheep{class_index}"
        ]
        file_moving_function(source_folder, destination_folders)

# making seperate loop for poll dorset sheep which have 2 more classes than the other three breeds
for class_index in range(1, 10):
        # setting source and destination folders for moving files
        source_folder = f"C:/University/Year 3 project/Project files/Proper dataset/pdsheep{class_index}"
        destination_folders = [
            f"C:/University/Year 3 project/Project files/Proper working dataset/train/pdsheep{class_index}",
            f"C:/University/Year 3 project/Project files/Proper working dataset/validation/pdsheep{class_index}",
            f"C:/University/Year 3 project/Project files/Proper working dataset/test/pdsheep{class_index}"
        ]
        file_moving_function(source_folder, destination_folders)