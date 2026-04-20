import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import CNN_train as train
import CNN_LSTM_train as train_lstm
import numpy as np
from torchmetrics.classification import MulticlassAccuracy


#transform for validation data same as training data
validation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# loading in the validation dataset and creating a dataloader for it
validation_dataset_path = "sheep_face_working_dataset/validation"
validation_dataset = datasets.ImageFolder(root = validation_dataset_path, transform = validation_transform)
validation_data_loader = DataLoader(validation_dataset, batch_size = 16, shuffle = False)

# Evaluate the model on the validation dataset, gradients not needed for evaluation
def validate_model(model):

    with torch.no_grad():

        correct_validation_classifications = 0

        accuracy = MulticlassAccuracy(num_classes = 100).to('cpu')

        # Iterate through the validation data loader
        for validation_input, validation_label in validation_data_loader:

            validation_outputs = model(validation_input)

            _, validation_prediction_class = torch.max(validation_outputs, 1)

            accuracy.update(validation_prediction_class, validation_label.data)

        accuracy_array = np.array([accuracy.compute().item()], dtype = float)

        return accuracy_array

    
# Load the trained neural network model
facial_recognition_model = train.SheepFaceClassifier()

# validation accuracy calculation for CNN model used with batch size 6
state_dict = torch.load('./CNN_facial_recognition_model_6_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()

validation_accuracy_CNN_6_array = validate_model(facial_recognition_model)
np.save('CNN_6_validation_accuracy.npy', validation_accuracy_CNN_6_array)

# validation accuracy calculation for CNN model used with batch size 16
state_dict = torch.load('./CNN_facial_recognition_model_16_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()

validation_accuracy_CNN_16_array = validate_model(facial_recognition_model)
np.save('CNN_16_validation_accuracy.npy', validation_accuracy_CNN_16_array)


facial_recognition_model = train_lstm.SheepFaceClassifierCNNLSTM()

# validation accuracy calculation for CNN LSTM model used with batch size 6
state_dict = torch.load('./CNN_LSTM_facial_recognition_model_batch_size_6.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()

validation_accuracy_CNN_LSTM_6_array = validate_model(facial_recognition_model)
np.save('CNN_LSTM_6_validation_accuracy.npy', validation_accuracy_CNN_LSTM_6_array)

# validation accuracy calculation for CNN LSTM model used with batch size 16
state_dict = torch.load('./CNN_LSTM_facial_recognition_model_batch_size_16.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()

validation_accuracy_CNN_LSTM_16_array = validate_model(facial_recognition_model)
np.save('CNN_LSTM_16_validation_accuracy.npy', validation_accuracy_CNN_LSTM_16_array)


