import numpy as np
import matplotlib.pyplot as plt


# loading in the training accuracies for CNN with batch size 6 and 16
CNN_6_train_accuracies = np.load('CNN_model_6_train_accuracies.npy')
CNN_16_train_accuracies = np.load('CNN_model_16_train_accuracies.npy')

# plotting training accuracy graph for CNN with batch size 6 and 16 over course of epochs
epochs = range(1, len(CNN_6_train_accuracies) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, CNN_6_train_accuracies, marker='o', label='CNN batch size 6 Accuracy')
plt.plot(epochs, CNN_16_train_accuracies, marker='o', label='CNN batch size 16 Accuracy')
plt.title("Training Accuracy of CNN Models with Batch Sizes 6 and 16")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# loading training accuracies for CNN-LSTM with batch size 6 and 16
CNN_LSTM_6_train_accuracies = np.load('CNN_LSTM_model_6_train_accuracies.npy')
CNN_LSTM_16_train_accuracies = np.load('CNN_LSTM_model_16_train_accuracies.npy')

# plotting training accuracy graph for CNN-LSTM with batch size 6 and 16 over course of epochs
epochs = range(1, len(CNN_LSTM_6_train_accuracies) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, CNN_LSTM_6_train_accuracies, marker='o', label='CNN-LSTM batch size 6 Accuracy')
plt.plot(epochs, CNN_LSTM_16_train_accuracies, marker='o', label='CNN-LSTM batch size 16 Accuracy')
plt.title("Training Accuracy of CNN-LSTM Models with Batch Sizes 6 and 16")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# creating a table to compare the validation accuracy, training time and training peak memory usage for the CNN and CNN-LSTM models with batch sizes of 6 and 16

data = [
    ["CNN (bs = 6)", f"{np.load('CNN_model_6_train_time.npy')[0]:.2f}", f"{np.load('CNN_model_6_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_6_validation_accuracy.npy')[0] * 100:.2f}"],
    ["CNN (bs = 16)", f"{np.load('CNN_model_16_train_time.npy')[0]:.2f}", f"{np.load('CNN_model_16_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_16_validation_accuracy.npy')[0] * 100:.2f}"],
    ["CNN-LSTM (bs = 6)", f"{np.load('CNN_LSTM_model_6_train_time.npy')[0]:.2f}", f"{np.load('CNN_LSTM_model_6_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_LSTM_6_validation_accuracy.npy')[0] * 100:.2f}"],
    ["CNN-LSTM (bs = 16)", f"{np.load('CNN_LSTM_model_16_train_time.npy')[0]:.2f}", f"{np.load('CNN_LSTM_model_16_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_LSTM_16_validation_accuracy.npy')[0] * 100:.2f}"]
]

columns = ["Model", "Training Time (minutes)", "Peak Memory Usage (MB)", "Validation Accuracy (%)"]

fig, ax = plt.subplots()
fig.figsize = (15, 6)
ax.axis("off")
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center"
)

table.auto_set_font_size(True)
table.scale(1.25, 1.25)
plt.show()

# creating a table to compare the test accuracy, precision, recall and F1 score for the CNN and CNN-LSTM models with batch sizes of 6 and 16
data = [
    ["CNN (bs = 6)", f"{np.load('CNN_6_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_6_test_precision.npy')[0]:.2f}", f"{np.load('CNN_6_test_recall.npy')[0]:.2f}", f"{np.load('CNN_6_test_f1.npy')[0]:.2f}"],
    ["CNN (bs = 16)", f"{np.load('CNN_16_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_16_test_precision.npy')[0]:.2f}", f"{np.load('CNN_16_test_recall.npy')[0]:.2f}", f"{np.load('CNN_16_test_f1.npy')[0]:.2f}"],
    ["CNN-LSTM (bs = 6)", f"{np.load('CNN_LSTM_6_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_LSTM_6_test_precision.npy')[0]:.2f}", f"{np.load('CNN_LSTM_6_test_recall.npy')[0]:.2f}", f"{np.load('CNN_LSTM_6_test_f1.npy')[0]:.2f}"],
    ["CNN-LSTM (bs = 16)", f"{np.load('CNN_LSTM_16_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_LSTM_16_test_precision.npy')[0]:.2f}", f"{np.load('CNN_LSTM_16_test_recall.npy')[0]:.2f}", f"{np.load('CNN_LSTM_16_test_f1.npy')[0]:.2f}"]
]

columns = ["Model", "Test Accuracy (%)", "Test Precision (%)", "Test Recall (%)", "Test F1 Score (%)"]

fig, ax = plt.subplots()
fig.figsize = (15, 6)
ax.axis("off")
table = ax.table(
    cellText=data,
    colLabels=columns,
    loc="center"
)

table.auto_set_font_size(True)
table.scale(1.25, 1.25)
plt.show()