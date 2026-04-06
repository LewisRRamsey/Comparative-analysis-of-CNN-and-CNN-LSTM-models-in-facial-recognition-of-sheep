from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import img_to_array, load_img
import os
import shutil
import numpy.random as nr
import random


# function for generating noise for images (put as preprocessing function in ImageDataGenerator) 
def generate_gaussian_noise(image):
    image = image.astype('float32')
    noise = nr.normal(loc = 0.0, scale = 10, size = image.shape)
    noisy_image = image + noise
    return noisy_image

# horizontal flip image generation process
datagen_horizontal = ImageDataGenerator(
        horizontal_flip = True)

# noise generator process for images
datagen_noise = ImageDataGenerator(
        preprocessing_function = generate_gaussian_noise)

# main augmentation process for all previously generated images including suitable rotation, brightness adjustment and shearing (orientation change simulation)
datagen_main = ImageDataGenerator(
        rotation_range = 30,
        brightness_range = (0.02, 1.75),
        shear_range = 10,
        zoom_range = 0.12,
        width_shift_range = 0.12,
        height_shift_range = 0.12)

def image_processing_function(input_directory, filename):
        image_path = os.path.join(input_directory, filename)
        # Loading each image 
        image = load_img(image_path) 
        # Converting the image to an array
        image_array = img_to_array(image)
        # Reshaping the image
        image_array = image_array.reshape((1, ) + image_array.shape)
        return image_array

def file_moving_function(source, destination):
        allfiles = os.listdir(source)
        for file in allfiles:
                source_path = os.path.join(source, file)
                destination_path = os.path.join(destination, file)
                os.rename(source_path, destination_path)
        

# input directory for where original image is accessed from, changes with every class in dataset (after every augmentation run)
for sheep_type in ["mc", "pd", "su", "ws"]:
        for image_index in range(1, 26):


                # horzintal flip image augmentation generation

                # setting input and output directories for horizontal flip image generation
                input_directory = f"C:/University/Year 3 project/Project files/sheep_face_dataset/{sheep_type}sheep{image_index}" 
                output_directory = f"C:/University/Year 3 project/Project files/sheep_face_dataset/{sheep_type}sheep{image_index}"

                # creates output directories if they do not exist
                os.makedirs(output_directory, exist_ok=True)

                for filename in os.listdir(input_directory):
                        # processing image into array
                        image_array = image_processing_function(input_directory, filename)
                        i = 0
                        for batch in datagen_horizontal.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = f'{sheep_type}sheep{image_index}hflip', save_format = 'jpg'):
                                i += 1
                                if i == 1:
                                        break


                # all generated images augmentation (rotation, brightness adjustment, shearing, zoom, width shift, height shift)
                
                # setting input and output directories for height shifted image generation
                input_directory = f"C:/University/Year 3 project/Project files/sheep_face_dataset/{sheep_type}sheep{image_index}" 
                output_directory = f"C:/University/Year 3 project/Project files/sheep_face_dataset/{sheep_type}sheep{image_index}"

                for filename in os.listdir(input_directory):
                        image_array = image_processing_function(input_directory, filename)
                        # creating the noisy augmented image, stopping at 1 augmentations per image
                        i = 0
                        for batch in datagen_main.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = f'{sheep_type}sheep{image_index}', save_format = 'jpg'):
                                i += 1
                                if i == 25:
                                        break


                # noisy image augmentation generation
                
                # setting input and output directories for height shifted image generation
                input_directory = f"C:/University/Year 3 project/Project files/sheep_face_dataset/{sheep_type}sheep{image_index}" 
                output_directory = f"C:/University/Year 3 project/Project files/sheep_face_dataset/{sheep_type}sheep{image_index}"

                os.makedirs(output_directory, exist_ok=True)

                for filename in os.listdir(input_directory):
                        image_array = image_processing_function(input_directory, filename)
                        # creating the noisy augmented image, stopping at 1 augmentations per image
                        i = 0
                        num = random.randint(1, 10)
                        if num == 4:
                                for batch in datagen_noise.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = f'{sheep_type}sheep{image_index}noise', save_format = 'jpg'):
                                        i += 1
                                        if i == 1:
                                                break