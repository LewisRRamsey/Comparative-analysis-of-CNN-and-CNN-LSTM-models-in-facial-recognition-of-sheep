from torch import device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
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
import os

# defining the path to the dataset
dataset_train_path = "Mini_test_dataset"
dataset_validation_path = "Mini_test_dataset"

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

            # iterating through the sequences in each class and adding the path to the sequence and its corresponding label to the samples list
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
            image_files = [file for file in sorted(os.listdir(sequence_path)) if file.lower().endswith((".jpg"))]
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
validation_dataset = SequenceDataset(dataset_validation_path, transform=dataset_transform)


# creating a dataloader for the train set of images
def set_train_loader(batch_size):
    return DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

# creating a dataloader for the validation set of images
def set_validation_loader(batch_size):
    return DataLoader(validation_dataset, batch_size = batch_size, shuffle = False, num_workers = 4)

# creating model class
class SheepFaceClassifierCNNLSTM(nn.Module):
    # defining model architecture
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # loading in ResNet18 model
        self.pretrained_model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # removing the fully connected layer from the pretrained model
        self.pretrained_model.fc = nn.Identity()
        # setting the ResNet18 feature size to 512, as the output from the ResNet18 model will be used as the input to the LSTM layer
        self.feature_dim = 512
        # defining the lstm layer
        self.lstm = nn.LSTM(input_size = self.feature_dim, hidden_size = 200, num_layers = 2, batch_first = True, bidirectional = True)
        # creating classifier layer to get the output for 34 classes
        self.classifier_layer = nn.Linear(in_features = 200*2, out_features = 34, bias = True)
        # freezing all layers initially in the pretrained model
        for parameters in self.pretrained_model.parameters():
            parameters.requires_grad = False
        # unfreezing batchnorm layers, layer 4 in the pretrained model to allow it to be trained on the dataset, allowing the model to learn from the dataset while still benefiting from the pretrained weights in the frozen layers
        for m in self.pretrained_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                for p in m.parameters():
                    p.requires_grad = True
        for parameters in self.pretrained_model.layer4.parameters():
            parameters.requires_grad = True


    def forward(self, x):
        
        batch_size, sequence_length, channels, height, width = x.shape

        # Merge batch and time for CNN, so that it can percieve the sequence of images as a batch of images
        x = x.view(batch_size * sequence_length, channels, height, width)

        # CNN feature extraction
        features = self.pretrained_model(x)       

        # Reshape back to (batch, time, features), allows the LSTM to still extract spatial features from the images
        feats = features.view(batch_size, sequence_length, -1)    
        
        # extracting information from the LSTM layer
        lstm_output, _ = self.lstm(feats)      
        lstm_output = lstm_output[:, -1, :]                 

        return self.classifier_layer(lstm_output)


    def training_model(self, epochs, train_loader, validation_loader):
        # defining the loss function and the optimiser
        loss_criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr = 0.0001)
        scheduler = StepLR(optimiser, step_size = 4, gamma = 0.2)
        training_accuracies = []
        validation_accuracies = []
        best_validation_accuracy = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0
            self.train()
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

            # every 4 epochs reducing learning rate
            if epoch % 4 == 0:
                scheduler.step()

            # returning training loss and accuracy for epoch
            train_accuracy = (correct / total) * 100
            training_accuracies.append(train_accuracy)
            print(f'Epoch {epoch} - Training Loss: {epoch_loss / total}, Training Accuracy: {train_accuracy:.2f}%')

            # transitioning model to test mode for validation accuracy calculation
            self.eval()
            validation_correct = 0
            validation_total = 0
            validation_loss = 0.0

            # calculating validation accuracy and loss
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = self(inputs)
                    loss = loss_criterion(outputs, labels)

                    validation_loss += loss.item() * labels.size(0)
                    _, predicted = torch.max(outputs, 1)
                    validation_correct += (predicted == labels).sum().item()
                    validation_total += labels.size(0)

            # displaying validation accuracy for epoch
            validation_accuracy = 100 * validation_correct / validation_total
            validation_accuracies.append(validation_accuracy)
            print(f"Epoch {epoch}: Validation Accuracy = {validation_accuracy:.2f}%")

            # saving model if new best validation accuracy is found
            if validation_accuracy > best_validation_accuracy:
                best_validation_accuracy = validation_accuracy
                torch.save(self.state_dict(), f'./CNN_LSTM_facial_recognition_model_batch_size_{self.batch_size}.pth')
                print("Saved new best model")

        # returning array of training accuracies and best validation accuracy
        return training_accuracies, validation_accuracies, best_validation_accuracy


# ensures training is only done when this script is run directly
# prevents training from being done when this script is imported as a module, e.g for testing
if __name__ == '__main__':

    # setting train_loader for batch size of 6
    train_loader = set_train_loader(batch_size = 6)

    # setting validation_loader for batch size of 6
    validation_loader = set_validation_loader(batch_size = 6)

    # creating an instance of the model with batch size of 6
    model_6 = SheepFaceClassifierCNNLSTM(batch_size = 6)
    # training the model and measuring time and memory usage
    tracemalloc.start()
    start_time = time.perf_counter()
    training_accuracies_6, validation_accuracies_6, best_validation_accuracy_6 = model_6.training_model(epochs = 1, train_loader = train_loader, validation_loader = validation_loader)
    end_time = time.perf_counter()
    current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time_mins = (end_time - start_time) / 60
    peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)

    # saving accuracies, train time and memory usage to numpy files
    np.save('CNN_LSTM_model_6_train_accuracies.npy', training_accuracies_6)
    np.save('CNN_LSTM_model_6_validation_accuracies.npy', validation_accuracies_6)
    np.save('CNN_LSTM_model_6_peak_mem_usage.npy', np.array([peak_mem_usage_MB], dtype = float))
    np.save('CNN_LSTM_model_6_train_time.npy', np.array([elapsed_time_mins], dtype = float))
    np.save('CNN_LSTM_model_6_best_validation_accuracy.npy', np.array([best_validation_accuracy_6], dtype = float))
    print('Training is complete for CNN-LSTM with a batch size of 6')





    # setting train_loader for batch size of 16
    train_loader = set_train_loader(batch_size = 16)

    # setting validation_loader for batch size of 16
    validation_loader = set_validation_loader(batch_size = 16)

    # creating an instance of the model with batch size of 16
    model_16 = SheepFaceClassifierCNNLSTM(batch_size = 16)
    # training the model and measuring time and memory usage
    tracemalloc.start()
    start_time = time.perf_counter()
    training_accuracies_16, validation_accuracies_16, best_validation_accuracy_16 = model_16.training_model(epochs = 25, train_loader = train_loader, validation_loader = validation_loader)
    end_time = time.perf_counter()
    current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    elapsed_time_mins = (end_time - start_time) / 60
    peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)

    # saving accuracies, train time and memory usage to numpy files
    np.save('CNN_LSTM_model_16_train_accuracies.npy', training_accuracies_16)
    np.save('CNN_LSTM_model_16_validation_accuracies.npy', validation_accuracies_16)
    np.save('CNN_LSTM_model_16_peak_mem_usage.npy', np.array([peak_mem_usage_MB], dtype = float))
    np.save('CNN_LSTM_model_16_train_time.npy', np.array([elapsed_time_mins], dtype = float))
    np.save('CNN_LSTM_model_16_best_validation_accuracy.npy', np.array([best_validation_accuracy_16], dtype = float))
    print('Training is complete for CNN-LSTM with batch size of 16')
