import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import CNN_train as train
import CNN_LSTM_train as train_lstm
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score


#transform for test data same as training data
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),       
    transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

# loading in the test dataset and creating a dataloader for it
test_dataset_path = "C:/University/Year 3 project/Project files/sheep_face_working_dataset/test"
test_dataset = datasets.ImageFolder(root = test_dataset_path, transform = test_transform)
test_data_loader = DataLoader(test_dataset, batch_size = 16, shuffle = False)

# Evaluate the model on the test dataset, gradients not needed for evaluation
def test_model(facial_recognition_model): 

    with torch.no_grad():

        accuracy = MulticlassAccuracy(num_classes = 100).to('cpu')
        precision = MulticlassPrecision(num_classes = 100, average='macro').to('cpu')
        recall = MulticlassRecall(num_classes = 100, average='macro').to('cpu')
        f1 = MulticlassF1Score(num_classes = 100, average='macro').to('cpu')

        # Iterate through the test data loader
        for test_input, test_label in test_data_loader:

            test_outputs = facial_recognition_model(test_input)

            _, test_prediction_class = torch.max(test_outputs, 1)

            correct_test_classifications += torch.sum(test_prediction_class == test_label.data)

            accuracy.update(test_prediction_class, test_label.data)
            precision.update(test_prediction_class, test_label.data)
            recall.update(test_prediction_class, test_label.data)
            f1.update(test_prediction_class, test_label.data)

        test_acc_array = np.array([accuracy.comput().item()], dtype = float)
        test_prec_array = np.array([precision.compute().item()], dtype = float)
        test_recall_array = np.array([recall.compute().item()], dtype = float)
        test_f1_array = np.array([f1.compute().item()], dtype = float)

        return test_acc_array, test_prec_array, test_recall_array, test_f1_array
    
# Load the trained neural network model
facial_recognition_model = train.SheepFaceClassifier()

# calculating test accuracy, precision, recall and f1 score for CNN with batch size 6
state_dict = torch.load('./CNN_facial_recognition_model_6_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model)
np.save('CNN_6_test_accuracy.npy', test_accuracy)
np.save('CNN_6_test_precision.npy', test_precision)
np.save('CNN_6_test_recall', test_recall)
np.save('CNN_6_test_f1', test_f1)
'''

# calculating test accuracy, precision, recall and f1 score for CNN with batch size 16
state_dict = torch.load('./CNN_facial_recognition_model_16_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model)
np.save('CNN_16_test_accuracy.npy', test_accuracy)
np.save('CNN_16_test_precision.npy', test_precision)
np.save('CNN_16_test_recall', test_recall)
np.save('CNN_16_test_f1', test_f1)


# Load the trained neural network model
facial_recognition_model = train_lstm.SheepFaceClassifierCNNLSTM()

# calculating test accuracy, precision, recall and f1 score for CNN-LSTM with batch size 6
state_dict = torch.load('./CNN_LSTM_facial_recognition_model_6_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model)
np.save('CNN_LSTM_6_test_accuracy.npy', test_accuracy)
np.save('CNN_LSTM_6_test_precision.npy', test_precision)
np.save('CNN_LSTM_6_test_recall', test_recall)
np.save('CNN_LSTM_6_test_f1', test_f1)

# calculating test accuracy, precision, recall and f1 score for CNN-LSTM with batch size 16
state_dict = torch.load('./CNN_LSTM_facial_recognition_model_16_batch_size.pth', map_location='cpu')
facial_recognition_model.load_state_dict(state_dict)
facial_recognition_model.eval()
test_accuracy, test_precision, test_recall, test_f1 = test_model(facial_recognition_model)
np.save('CNN_LSTM_16_test_accuracy.npy', test_accuracy)
np.save('CNN_LSTM_16_test_precision.npy', test_precision)
np.save('CNN_LSTM_16_test_recall', test_recall)
np.save('CNN_LSTM_16_test_f1', test_f1)
'''