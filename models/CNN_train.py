import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.optim.lr_scheduler import StepLR
import time
import torchvision
import visualtorch
import matplotlib.pyplot as plt

# defining the path to the dataset
dataset_train_path = "sheep_face_working_dataset/train"

# normalising the images using ImageNet mean and standard deviation for ResNet50 model, as well as resizing the images and converting them to tensors
dataset_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# loading in the training dataset
train_dataset =datasets.ImageFolder(root = dataset_train_path, transform = dataset_transform)

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
        # passing the input through the ResNet18 model
        return self.pretrained_model(x)               

    def train_model(self, epochs, train_loader):
        # defining the loss function and the optimiser
        loss_criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr = 0.001)
        scheduler = StepLR(optimiser, step_size = 3, gamma = 0.1)
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
            if epoch % 3 == 0:
                scheduler.step()
            # returning loss and accuracy for epoch
            accuracy = (correct / total) * 100
            accuracies.append(accuracy)
            print(f"Epoch [{epoch}] Train Loss: {epoch_loss / total}, Accuracy: {accuracy}%")
        # plotting graph of training accuracies
        # plt.plot(accuracies)
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracies')
        # plt.title('Training Accuracies for CNN model')
        # plt.show()
        # print("graph plotted")
        return accuracies[-1]
        

# ensures training is only done when this script is run directly
# prevents training from being done when this script is imported as a module, e.g for testing
if __name__ == '__main__':
    # setting train_loader for batch size of 6
    train_loader = set_train_loader(batch_size = 16)

    # creating an instance of the model for 40 epochs and batch size of 6
    model_40 = SheepFaceClassifier()
    # training the model with 40 epochs
    final_accuracy_50_6 = model_40.train_model(epochs=3, train_loader=train_loader)
    with open("Results_values.txt", "w") as file:
        file.write(str(final_accuracy_50_6))
    torch.save(model_40.state_dict(), './CNN_facial_recognition_model_40_epochs_6_batch_size.pth')
    print('Training is complete for 40 epochs model')
'''
    # creating an instance of the model for 50 epochs and batch size of 6
    model_50 = SheepFaceClassifier()
    # training the model with 50 epochs
    model.train(epochs=50, train_loader=train_loader)
    torch.save(model_50.state_dict(), './CNN_facial_recognition_model_50_epochs_6_batch_size.pth')
    print('Training is complete for 50 epochs model')

    # creating an instance of the model for 60 epochs and batch size of 6
    model_60 = SheepFaceClassifier()
    # training the model with 60 epochs
    model.train(epochs=60, train_loader=train_loader)
    torch.save(model_60.state_dict(), './CNN_facial_recognition_model_60_epochs_6_batch_size.pth')
    print('Training is complete for 60 epochs model')

    # setting train_loader for batch size of 16
    train_loader = set_train_loader(batch_size = 16)

    # creating an instance of the model for 50 epochs and batch size of 16
    model_50 = SheepFaceClassifier()
    # training the model with 50 epochs
    model.train(epochs=50, train_loader=train_loader)
    torch.save(model_50.state_dict(), './CNN_facial_recognition_model_50_epochs_6_batch_size.pth')
    print('Training is complete for 50 epochs model')

    # creating an instance of the model for 60 epochs and batch size of 16
    model_60 = SheepFaceClassifier()
    # training the model with 60 epochs
    model.train(epochs=60, train_loader=train_loader)
    torch.save(model_60.state_dict(), './CNN_facial_recognition_model_60_epochs_6_batch_size.pth')
    print('Training is complete for 60 epochs model')
'''
    # creating image of model structure
    #vt = visualtorch.lenet_view(model = model, input_shape = (1, 3, 224, 224), to_file = "CNN_model_LeNet_view.png")
    #print("LeNet view saved as CNN_model_LeNet_view.png")
