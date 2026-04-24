import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import CNN_train_2seq as train
import CNN_LSTM_train_2seq as train_lstm
import numpy as np
import tracemalloc
import time
from PIL import Image
import os
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score

# class used to create the dataset for the model by loading images in sequences of 2, as no builtin method exists for this, based on [36] https://www.codegenes.net/blog/create-dataset-of-images-pytorch/ 
class SequenceDataset(Dataset):
    def __init__(self, dataset_root_path, transform = None):

        # initialising the dataset root path, transformations, list of class samples and dictionary that maps the class names to their assigned index
        self.root = dataset_root_path
        self.transform = transform
        self.class_samples = []
        self.class_to_index = {}

        # Building classes from folder names in the root directory
        # assign class names to an integer index, which will be used as the label for the sequences of images in that class
        classes = sorted([folder for folder in os.listdir(dataset_root_path) if os.path.isdir(os.path.join(dataset_root_path, folder))])
        self.class_to_index = {class_name: index for index, class_name in enumerate(classes)}

        # Collect all sequences and their corresponding labels
        for class_name in classes:
            class_path = os.path.join(dataset_root_path, class_name)

            # iterating through the sequences in each class, creating a path for each
            for sequence_name in os.listdir(class_path):
                sequence_path = os.path.join(class_path, sequence_name)
                # after checking if the path is a folder, adding the path and indexed label to the list of individuals
                if os.path.isdir(sequence_path):
                    self.class_samples.append((sequence_path, self.class_to_index[class_name]))

    def __len__(self):
            # allowing dataloader to get the length of the sequences (in this case is always 2, as model cannot process variable length seuqences)
            return len(self.class_samples)

    def __getitem__(self, index):
            # allowing dataloader to get the sequences and their corresponding labels
            sequence_path, label = self.class_samples[index]

            # Load the two images in sorted order for the LSTM (other model) to be able to learn from the sequence of images, but done here to maintain consistency
            image_files = [file for file in sorted(os.listdir(sequence_path))]
            images = []

            # iterating through the 2 images in each sequence
            for image_name in image_files:
                # getting the path to the image and loading it from the disk (ensuring in colour format)
                img_path = os.path.join(sequence_path, image_name)
                image = Image.open(img_path).convert("RGB")

                # applying the transformations to the image, listed in next code block outside class
                if self.transform:
                    image = self.transform(image)

                # adding the image to the list of images for the sequence
                images.append(image)

            # Stacking the images to a single tensor, so they can be processed by the models
            sequence = torch.stack(images, dim = 0)

            # returning the tensor of image sequence and corresponding class index label
            return sequence, label


#transform for test data same as training data
test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

# loading in the test dataset and creating a dataloader for it
test_dataset_path = "Final dataset/test"
test_dataset = SequenceDataset(test_dataset_path, transform = test_transform)


# Evaluate the model on the test dataset, gradients not needed for evaluation
def test_model(facial_recognition_model, test_data_loader): 

    with torch.no_grad():

        accuracy = MulticlassAccuracy(num_classes = 100).to('cpu')
        precision = MulticlassPrecision(num_classes = 100, average='macro').to('cpu')
        recall = MulticlassRecall(num_classes = 100, average='macro').to('cpu')
        f1 = MulticlassF1Score(num_classes = 100, average='macro').to('cpu')

        # Iterate through the test data loader
        for test_input, test_label in test_data_loader:

            test_outputs = facial_recognition_model(test_input)

            _, test_prediction_class = torch.max(test_outputs, 1)

            accuracy.update(test_prediction_class, test_label.data)
            precision.update(test_prediction_class, test_label.data)
            recall.update(test_prediction_class, test_label.data)
            f1.update(test_prediction_class, test_label.data)

        test_acc_array = np.array([accuracy.compute().item()], dtype = float)
        test_prec_array = np.array([precision.compute().item()], dtype = float)
        test_recall_array = np.array([recall.compute().item()], dtype = float)
        test_f1_array = np.array([f1.compute().item()], dtype = float)

        return test_acc_array, test_prec_array, test_recall_array, test_f1_array
    
# Load the trained neural network model
facial_recognition_model = train.SheepFaceClassifier()
test_data_loader = DataLoader(test_dataset, batch_size = 6, shuffle = False)

# calculating test accuracy, precision, recall and f1 score for CNN with batch size 6
state_dict = torch.load('./CNN_facial_recognition_model_batch_size_6.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
tracemalloc.start()
start_time = time.perf_counter()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model, test_data_loader)
end_time = time.perf_counter()
current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
elapsed_time_mins = (end_time - start_time) / 60
peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
np.save('CNN_6_test_time', elapsed_time_mins)
np.save('CNN_6_test_peak_mem_usage', peak_mem_usage_MB)
np.save('CNN_6_test_accuracy.npy', test_accuracy)
np.save('CNN_6_test_precision.npy', test_precision)
np.save('CNN_6_test_recall.npy', test_recall)
np.save('CNN_6_test_f1.npy', test_f1)

test_data_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

# calculating test accuracy, precision, recall and f1 score for CNN with batch size 8
state_dict = torch.load('./CNN_facial_recognition_model_batch_size_8.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
tracemalloc.start()
start_time = time.perf_counter()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model, test_data_loader)
end_time = time.perf_counter()
current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
elapsed_time_mins = (end_time - start_time) / 60
peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
np.save('CNN_8_test_time', elapsed_time_mins)
np.save('CNN_8_test_peak_mem_usage', peak_mem_usage_MB)
np.save('CNN_8_test_accuracy.npy', test_accuracy)
np.save('CNN_8_test_precision.npy', test_precision)
np.save('CNN_8_test_recall.npy', test_recall)
np.save('CNN_8_test_f1.npy', test_f1)

# Load the trained neural network model
facial_recognition_model = train_lstm.SheepFaceClassifierCNNLSTM(6)
test_data_loader = DataLoader(test_dataset, batch_size = 6, shuffle = False)

# calculating test accuracy, precision, recall and f1 score for CNN-LSTM with batch size 6
state_dict = torch.load('./CNN_LSTM_facial_recognition_model_batch_size_6.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
tracemalloc.start()
start_time = time.perf_counter()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model, test_data_loader)
end_time = time.perf_counter()
current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
elapsed_time_mins = (end_time - start_time) / 60
peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
np.save('CNN_LSTM_6_test_time', elapsed_time_mins)
np.save('CNN_LSTM_6_test_peak_mem_usage', peak_mem_usage_MB)
np.save('CNN_LSTM_6_test_accuracy.npy', test_accuracy)
np.save('CNN_LSTM_6_test_precision.npy', test_precision)
np.save('CNN_LSTM_6_test_recall.npy', test_recall)
np.save('CNN_LSTM_6_test_f1.npy', test_f1)

facial_recognition_model = train_lstm.SheepFaceClassifierCNNLSTM(8)
test_data_loader = DataLoader(test_dataset, batch_size = 8, shuffle = False)

# calculating test accuracy, precision, recall and f1 score for CNN-LSTM with batch size 8
state_dict = torch.load('./CNN_LSTM_facial_recognition_model_batch_size_8.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
tracemalloc.start()
start_time = time.perf_counter()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model, test_data_loader)
end_time = time.perf_counter()
current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
tracemalloc.stop()
elapsed_time_mins = (end_time - start_time) / 60
peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
np.save('CNN_LSTM_8_test_time', elapsed_time_mins)
np.save('CNN_LSTM_8_test_peak_mem_usage', peak_mem_usage_MB)
np.save('CNN_LSTM_8_test_accuracy.npy', test_accuracy)
np.save('CNN_LSTM_8_test_precision.npy', test_precision)
np.save('CNN_LSTM_8_test_recall.npy', test_recall)
np.save('CNN_LSTM_8_test_f1.npy', test_f1)
