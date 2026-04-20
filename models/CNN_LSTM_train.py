from torch import device
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
import time
import torchvision
import numpy as np
import tracemalloc

# defining the path to the dataset
dataset_train_path = "sheep_face_working_dataset/train"

# normalising the images using ImageNet mean and standard deviation for ResNet50 model, as well as resizing the images and converting them to tensors
dataset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# loading in the training dataset
train_dataset = datasets.ImageFolder(root = dataset_train_path, transform = dataset_transform)

# creating a dataloader for the train set of images
def set_train_loader(batch_size):
    return DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

# creating model class
class SheepFaceClassifierCNNLSTM(nn.Module):
    # defining model architecture
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        # loading in ResNet50 model
        self.pretrained_model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        # removing the fully connected layer from the pretrained model
        self.pretrained_model.fc = nn.Identity()
        # setting the ResNet18 feature size to 512, as the output from the ResNet18 model will be used as the input to the LSTM layer
        self.feature_dim = 512
        # defining the lstm layer
        self.lstm = nn.LSTM(input_size=512, hidden_size=200, num_layers=2, batch_first=True, bidirectional = True)
        # creating classifier layer to get the output for 100 classes
        self.classifier_layer = nn.Linear(in_features=200*2, out_features=100, bias=True)
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
        features = self.pretrained_model(x)  # (batch*seq_len, 512)
        features = features.view(self.batch_size, 1, 512)

        y, _ = self.lstm(features)

        return self.classifier_layer(y[:, -1, :])


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
            print(f'Epoch {epoch} - Loss: {epoch_loss / total}, Accuracy: {accuracy}%')
        # returning array of accuracies
        return accuracies


# ensures training is only done when this script is run directly
# prevents training from being done when this script is imported as a module, e.g for testing
if __name__ == '__main__':

    # setting train_loader for batch size of 6
    train_loader = set_train_loader(batch_size = 6)

    # creating an instance of the model with batch size of 6
    model_6 = SheepFaceClassifierCNNLSTM(batch_size = 6)
    # training the model and measuring time and memory usage
    tracemalloc.start()
    start_time = time.perf_counter()
    accuracies_6 = model_6.training_model(epochs = 25, train_loader = train_loader)
    end_time = time.perf_counter()
    current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
    elapsed_time_mins = (end_time - start_time) / 60
    peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
    # converitng elapsed time and peak memory usage to numpy arrays
    train_time_array = np.array([elapsed_time_mins], dtype = float)
    memory_usage_array = np.array([peak_mem_usage_MB], dtype = float)
    # saving accuracies, train time and memory usage to numpy files
    np.save('CNN_LSTM_model_6_train_accuracies.npy', accuracies_6)
    np.save('CNN_LSTM_model_6_peak_mem_usage.npy', memory_usage_array)
    np.save('CNN_LSTM_model_6_train_time.npy', train_time_array)

    # saving the models weights and biases 
    torch.save(model_6.state_dict(), './CNN_LSTM_facial_recognition_model_batch_size_6.pth')
    print('Training is complete for CNN-LSTM with a batch size of 6')


    # setting train_loader for batch size of 16
    train_loader = set_train_loader(batch_size = 16)

    # creating an instance of the model with batch size of 16
    model_16 = SheepFaceClassifierCNNLSTM()
    # training the model and measuring time and memory usage
    tracemalloc.start()
    start_time = time.perf_counter()
    accuracies_16 = model_16.training_model(epochs = 25, train_loader = train_loader)
    end_time = time.perf_counter()
    current_mem_usage, peak_mem_usage = tracemalloc.get_traced_memory()
    elapsed_time_mins = (end_time - start_time) / 60
    peak_mem_usage_MB = peak_mem_usage / (1024 ** 2)
    # converitng elapsed time and peak memory usage to numpy arrays
    train_time_array = np.array([elapsed_time_mins], dtype = float)
    memory_usage_array = np.array([peak_mem_usage_MB], dtype = float)
    # saving accuracies, train time and memory usage to numpy files
    np.save('CNN_LSTM_model_16_train_accuracies.npy', accuracies_16)
    np.save('CNN_LSTM_model_16_peak_mem_usage.npy', memory_usage_array)
    np.save('CNN_LSTM_model_16_train_time.npy', train_time_array)

    # saving the models weights and biases
    torch.save(model_16.state_dict(), './CNN_LSTM_facial_recognition_model_batch_size_16.pth')
    print('Training is complete for CNN-LSTM with batch size of 16')
