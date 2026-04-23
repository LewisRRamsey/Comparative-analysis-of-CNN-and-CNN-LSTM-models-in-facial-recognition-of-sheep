import os
import shutil

# putting data into sequences of 2 images for each class in train test and validation folders
for dataset_type in ["train", "validation", "test"]:
    # iterating through all sheep breeds with 8 classes each
    for sheep_type in ["mc", "su", "ws"]:
        # iterating through each class for the current breed
        for class_index in range(1, 9):
            # setting image count and index for sequences
            sequence_image_count = 0
            sequence_index = 1
            for image in os.listdir(f"C:/University/Year 3 project/Project files/Proper working dataset/{dataset_type}/{sheep_type}sheep{class_index}"):
                # setting destination folder for moving files
                destination_folder = f"C:/University/Year 3 project/Project files/Final dataset/{dataset_type}/{sheep_type}sheep{class_index}/sequence{sequence_index}"
                # creates output directories if they do not exist
                os.makedirs(destination_folder, exist_ok=True)
                # copying image to sequence folder
                shutil.copy2(f"C:/University/Year 3 project/Project files/Proper working dataset/{dataset_type}/{sheep_type}sheep{class_index}/{image}", destination_folder)
                sequence_image_count += 1
                # if 2 images have been added to the sequence, move on to the next sequence
                if sequence_image_count == 2:
                    sequence_index += 1
                    sequence_image_count = 0


# putting data into sequences of 2 images for each class in train test and validation folders
for dataset_type in ["train", "validation", "test"]:
    # iterating through all sheep breeds with 10 classes each
    for sheep_type in ["pd"]:
        # iterating through each class for the current breed
        for class_index in range(1, 11):
            # setting image count and index for sequences
            sequence_image_count = 0
            sequence_index = 1
            for image in os.listdir(f"C:/University/Year 3 project/Project files/Proper working dataset/{dataset_type}/{sheep_type}sheep{class_index}"):
                # setting destination folder for moving files
                destination_folder = f"C:/University/Year 3 project/Project files/Final dataset/{dataset_type}/{sheep_type}sheep{class_index}/sequence{sequence_index}"
                # creates output directories if they do not exist
                os.makedirs(destination_folder, exist_ok=True)
                # copying image to sequence folder
                shutil.copy2(f"C:/University/Year 3 project/Project files/Proper working dataset/{dataset_type}/{sheep_type}sheep{class_index}/{image}", destination_folder)
                sequence_image_count += 1
                # if 2 images have been added to the sequence, move on to the next sequence
                if sequence_image_count == 2:
                    sequence_index += 1
                    sequence_image_count = 0
