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
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers = 4)

# creating model class
class SheepFaceClassifier(nn.Module):
    # defining model architecture
    def __init__(self):
        super().__init__()
        # loading in ResNet50 model
        self.pretrained_model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        # changing the number of output features in the fully connected layer to 100, as there are 100 classes in the dataset
        self.pretrained_model.fc = nn.Linear(in_features=2048, out_features=100, bias=True)
        self.new_layers = [self.pretrained_model.layer4, self.pretrained_model.fc]

    def forward(self, x):
        # passing the input through the ResNet50 model
        return self.pretrained_model(x)
    
    def fine_tune(self):
        # initially freezing all the layers in the pretrained model
        for parameters in self.pretrained_model.parameters():
            parameters.requires_grad = False
        # unfreezing the layers specified to have weights trained
        for layer in self.new_layers:   
            for parameters in layer.parameters():
                parameters.requires_grad = True

    def train(self, epochs, train_loader):
        # defining the loss function and the optimiser
        loss_criterion = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(self.parameters(), lr = 0.001)
        scheduler = StepLR(optimiser, step_size = 5, gamma = 0.1)
        running_loss = 0.0
        number_of_batches = 0
        for epoch in range(1, epochs + 1):
            for inputs, labels in train_loader:
                # zeroing the gradients of the optimiser
                optimiser.zero_grad()
                # passing the inputs through the model to get the outputs
                outputs = self(inputs)
                loss = loss_criterion(outputs, labels)
                # backpropagating the loss and updating the weights
                running_loss = running_loss + loss.item()
                number_of_batches += 1
                loss.backward()
                optimiser.step()
                scheduler.step()
            print(f"Epoch [{epoch}] Train Loss: {running_loss / number_of_batches}")
        return running_loss / number_of_batches


    
# creating an instance of the model
model = SheepFaceClassifier()

# ensures training is only done when this script is run directly
# prevents training from being done when this script is imported as a module, e.g for testing
if __name__ == '__main__':
    # model.train(epochs=10, train_loader=train_loader)
    # torch.save(model.state_dict(), './CNN_facial_recognition_model.pth')
    # print('Training is complete')

    vt = visualtorch.lenet_view(model = model, input_shape = (1, 3, 224, 224), to_file = "CNN_model_LeNet_view.png")
    print("LeNet view saved as CNN_model_LeNet_view.png")

