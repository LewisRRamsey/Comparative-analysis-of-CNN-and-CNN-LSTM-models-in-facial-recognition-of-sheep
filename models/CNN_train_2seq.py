from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import os
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
import time
import torchvision
import numpy as np
import tracemalloc
from PIL import Image

# defining the path to the dataset
dataset_train_path = "Mini_test_dataset"

class SequenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.class_to_index = {}

        # Building classes from folder names in the root directory
        # assign class names to an integer index, which will be used as the label for the sequences of images in that class
        classes = sorted([folder for folder in os.listdir(root) if os.path.isdir(os.path.join(root, folder))])
        self.class_to_index = {class_name: index for index, class_name in enumerate(classes)}

        # Collect all sequences and their corresponding labels
        for class_name in classes:
            class_path = os.path.join(root, class_name)

            for sequence_name in os.listdir(class_path):
                sequence_path = os.path.join(class_path, sequence_name)
                if os.path.isdir(sequence_path):
                    self.samples.append((sequence_path, self.class_to_index[class_name]))

    def __len__(self):
            # allowing dataloader to get the length of the sequences
            return len(self.samples)

    def __getitem__(self, index):
            # allowing dataloader to get the sequences and their corresponding labels
            sequence_path, label = self.samples[index]

            # Load the two images in sorted order for the LSTM to be able to learn from the sequence of images
            image_files = sorted(os.listdir(sequence_path))
            images = []

            for image_name in image_files:
                img_path = os.path.join(sequence_path, image_name)
                image = Image.open(img_path).convert("RGB")

                if self.transform:
                    image = self.transform(image)

                images.append(image)

            # Stack into shape
            sequence = torch.stack(images, dim = 0)

            return sequence, label



# normalising the images using ImageNet mean and standard deviation for ResNet50 model, as well as resizing the images and converting them to tensors
dataset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])
        
train_dataset = SequenceDataset(dataset_train_path, transform=dataset_transform)

# creating a dataloader for the train set of images
def set_train_loader(batch_size):
    return DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

# creating model class
class SheepFaceClassifier(nn.Module):
    # defining model architecture
    def __init__(self):
        super().__init__()
        # loading in ResNet50 model
        self.pretrained_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # changing the number of output features in the fully connected layer to 100, as there are 100 classes in the dataset
        self.pretrained_model.fc = nn.Linear(in_features=512, out_features=100, bias=True)
        # freezing layers 1,2 and 3 in the pretrained model
        for parameters in self.pretrained_model.layer1.parameters():
            parameters.requires_grad = False
        for parameters in self.pretrained_model.layer2.parameters():
            parameters.requires_grad = False
        for parameters in self.pretrained_model.layer3.parameters():
            parameters.requires_grad = False
        for parameters in self.pretrained_model.layer4.parameters():
            parameters.requires_grad = True
        for parameters in self.pretrained_model.fc.parameters():
            parameters.requires_grad = True
    
    def forward(self, x):

        batch_size, seq_len, c, h, w = x.shape
        # flattens sequence for CNN feature extraction, allows the CNN to still extract spatial features from the images in the sequence
        x = x.view(batch_size * seq_len, c, h, w)
        # passing the input through the ResNet18 model to get the output for each image in the sequence
        frame_logits = self.pretrained_model(x)
        frame_logits = frame_logits.view(batch_size, seq_len, -1)
        # averaging the output from the two images in the sequence to get the output for the sequence of images
        sequence_logits = frame_logits.mean(dim = 1) 
        return sequence_logits            

    def training_model(self, epochs, train_loader):
        # defining the loss function and the optimiser
        loss_criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr = 0.00001)
        scheduler = StepLR(optimiser, step_size = 4, gamma = 0.2)
        accuracies = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zeroing the gradients of the optimiser
                optimiser.zero_grad()
                # passing the inputs through the model to get the outputs
                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                # backpropagating the loss and updating the weights
                epoch_loss += loss.item() * labels.size(0)
                loss.backward()
                optimiser.step()
                # calculating training accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
            if epoch % 4 == 0:
                scheduler.step()
            # returning loss and accuracy for epoch
            accuracy = (correct / total) * 100
            accuracies.append(accuracy)
            print(f"Epoch [{epoch}] Train Loss: {epoch_loss / total}, Accuracy: {accuracy}%")
        return accuracies
        

# ensures training is only done when this script is run directly
# prevents training from being done when this script is imported as a module, e.g for testing
if __name__ == '__main__':

    # setting train_loader for batch size of 6
    train_loader = set_train_loader(batch_size = 6)

    # creating an instance of the model with batch size of 6
    model_6 = SheepFaceClassifier()
    # training the model and measuring time and memory usage
    tracemalloc.start()
    start_time = time.perf_counter()
    accuracies_6 = model_6.training_model(epochs = 15, train_loader = train_loader)
    end_time = time.perf_counter()
    current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
    elapsed_time_mins = (end_time - start_time) / 60
    peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
    # converitng elapsed time and peak memory usage to numpy arrays
    train_time_array = np.array([elapsed_time_mins], dtype = float)
    memory_usage_array = np.array([peak_mem_usage_MB], dtype = float)
    # saving accuracies, train time and memory usage to numpy files
    np.save('CNN_model_6_train_accuracies.npy', accuracies_6)
    np.save('CNN_model_6_peak_mem_usage.npy', memory_usage_array)
    np.save('CNN_model_6_train_time.npy', train_time_array)

    # saving models weights and biases
    torch.save(model_6.state_dict(), './CNN_facial_recognition_model_6_batch_size.pth')
    print('Training is complete for batch size 6 model')


    # setting train_loader for batch size of 16
    train_loader = set_train_loader(batch_size = 16)

    # creating an instance of the model with batch size of 16
    model_16 = SheepFaceClassifier()
    # training the model and measuring time and memory usage
    tracemalloc.start()
    start_time = time.perf_counter()
    accuracies_16 = model_16.training_model(epochs = 15, train_loader = train_loader)
    end_time = time.perf_counter()
    current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
    elapsed_time_mins = (end_time - start_time) / 60
    peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
    # converitng elapsed time and peak memory usage to numpy arrays
    train_time_array = np.array([elapsed_time_mins], dtype = float)
    memory_usage_array = np.array([peak_mem_usage_MB], dtype = float)
    # saving accuracies, train time and memory usage to numpy files
    np.save('CNN_model_16_train_accuracies.npy', accuracies_16)
    np.save('CNN_model_16_peak_mem_usage.npy', memory_usage_array)
    np.save('CNN_model_16_train_time.npy', train_time_array)

    # saving models weights and biases
    torch.save(model_16.state_dict(), './CNN_facial_recognition_model_16_batch_size.pth')
    print('Training is complete for batch size 16 model')
