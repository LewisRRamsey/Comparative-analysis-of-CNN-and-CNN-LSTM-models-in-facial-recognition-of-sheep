import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import CNN_train as train


#transform for validation data same as training data
validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# loading in the validation dataset and creating a dataloader for it
validation_dataset_path = "sheep_face_working_dataset/validation"
validation_dataset = datasets.ImageFolder(root = validation_dataset_path, transform = validation_transform)
validation_data_loader = DataLoader(validation_dataset, batch_size = 6, shuffle = False)

# Evaluate the model on the validation dataset, gradients not needed for evaluation
def validate_model(model):
    #specify the same loss function as used in validation
    loss_criterion = nn.CrossEntropyLoss()

    with torch.no_grad():

        correct_validation_classifications = 0

        # Iterate through the validation data loader
        for validation_input, validation_label in validation_data_loader:

            validation_outputs = model(validation_input)

            batch_loss = loss_criterion(validation_outputs, validation_label)

            _, validation_prediction_class = torch.max(validation_outputs, 1)

            correct_validation_classifications += torch.sum(validation_prediction_class == validation_label.data)

        validation_accuracy = correct_validation_classifications.float() / len(validation_dataset)
        return (validation_accuracy.item() * 100)
    
# Load the trained neural network model
facial_recognition_model = train.SheepFaceClassifier()

state_dict = torch.load('./CNN_facial_recognition_model_6_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()

print(f'Validation accuracy: {validate_model(facial_recognition_model)}%')