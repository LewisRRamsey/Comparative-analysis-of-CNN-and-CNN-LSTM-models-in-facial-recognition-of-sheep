from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.utils import img_to_array, load_img
import os
import numpy.random as nr

def image_processing_function(input_directory, filename):
        image_path = os.path.join(input_directory, filename)
        # Loading each image 
        image = load_img(image_path) 
        # Converting the image to an array
        image_array = img_to_array(image)
        # Reshaping the image
        image_array = image_array.reshape((1, ) + image_array.shape)
        return image_array

def generate_gaussian_noise(image):
    image = image.astype('float32')
    noise = nr.normal(loc = 0.0, scale = 10, size = image.shape)
    noisy_image = image + noise
    return noisy_image

# horizontal flip image generation process
datagen_horizontal = ImageDataGenerator(
        horizontal_flip = True)

# zoomed in image generation process
datagen_zoomin = ImageDataGenerator(
        zoom_range = (0.8, 0.8))

# zoomed out image generation process
datagen_zoomout = ImageDataGenerator(
        zoom_range = (1.2, 1.2))

# rotated image generation process
datagen_rotate = ImageDataGenerator(
        rotation_range = 30)

# brightness adjusted image generation process
datagen_bright = ImageDataGenerator(
        brightness_range = (0.02, 2))

# width shfited image generation process
datagen_width = ImageDataGenerator(
        width_shift_range = 0.2)

# height shfited image generation process
datagen_height = ImageDataGenerator(
        height_shift_range = 0.2)

# noise injetced image generation process
datagen_noise = ImageDataGenerator(
        preprocessing_function = generate_gaussian_noise)

#####                                                                                                                                                        #####
# comment out all bar one blocks of code below for wanted example in specified directory in copy of dataset, delete augmented images when done on each iteration #
#####                                                                                                                                                        #####
'''
# setting input and output directories for horizontal flip image generation
input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_horizontal.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1hflip', save_format = 'jpg'):
                i += 1
                if i == 1:
                        break
     
# setting input and output directories for zoomed in image generation
input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_zoomin.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1zoomin', save_format = 'jpg'):
                i += 1
                if i == 1:
                        break
            
# setting input and output directories for zoomed out image generation
input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_zoomout.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1zoomout', save_format = 'jpg'):
                i += 1
                if i == 1:
                        break

input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_rotate.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1rotate', save_format = 'jpg'):
                i += 1
                if i == 5:
                        break
           
input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_bright.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1bright', save_format = 'jpg'):
                i += 1
                if i == 5:
                        break

input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_width.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1width', save_format = 'jpg'):
                i += 1
                if i == 5:
                        break

input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_height.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1height', save_format = 'jpg'):
                i += 1
                if i == 5:
                        break
'''
input_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1" 
output_directory = "C:/University/Year 3 project/Project files/sheep_face_dataset_copy/mcsheep1"

for filename in os.listdir(input_directory):
        # processing image into array
        image_array = image_processing_function(input_directory, filename)
        i = 0
        for batch in datagen_noise.flow(image_array, batch_size = 1, save_to_dir = output_directory, save_prefix = 'mcsheep1noise', save_format = 'jpg'):
                i += 1
                if i == 1:
                        break