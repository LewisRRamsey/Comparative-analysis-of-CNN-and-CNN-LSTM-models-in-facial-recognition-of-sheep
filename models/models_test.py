import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import CNN_train as train


#transform for test data same as training data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# loading in the test dataset and creating a dataloader for it
test_dataset_path = "C:/University/Year 3 project/Project files/sheep_face_working_dataset/test"
test_dataset = datasets.ImageFolder(root = test_dataset_path, transform = test_transform)
test_data_loader = DataLoader(test_dataset, batch_size = 6, shuffle = False)

# Load the trained neural network model
facial_recognition_model = train.SheepFaceClassifier()
facial_recognition_model.load_model('./CNN_facial_recognition_model.pth')
facial_recognition_model.eval()

#specify the same loss function as used in training
loss_criterion = nn.CrossEntropyLoss()

# Evaluate the model on the test dataset, gradients not needed for evaluation
with torch.no_grad():

    correct_test_classifications = 0

    # Iterate through the test data loader
    for test_input, test_label in test_data_loader:

        test_outputs = facial_recognition_model(test_input)

        batch_loss = loss_criterion(test_outputs, test_label)

        _, test_prediction_class = torch.max(test_outputs, 1)

        correct_test_classifications += torch.sum(test_prediction_class == test_label.data)

    test_accuracy = correct_test_classifications.float() / len(test_dataset)
    print('Test accuracy:', test_accuracy.item() * 100, '%')

