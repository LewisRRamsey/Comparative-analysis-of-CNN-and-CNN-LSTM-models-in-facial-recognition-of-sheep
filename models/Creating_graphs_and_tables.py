import numpy as np
import matplotlib.pyplot as plt


# loading in the training accuracies for CNN with batch size 6 and 8
CNN_6_train_accuracies = np.load('CNN_model_6_train_accuracies.npy')
CNN_8_train_accuracies = np.load('CNN_model_8_train_accuracies.npy')
CNN_6_validation_accuracies = np.load('CNN_model_6_validation_accuracies.npy')
CNN_8_validation_accuracies = np.load('CNN_model_8_validation_accuracies.npy')

# plotting training and validation accuracy graph for CNN with batch size 6 and 8 over course of epochs
epochs = range(1, len(CNN_6_train_accuracies) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, CNN_6_train_accuracies, marker = 'o', label = 'CNN batch size 6 train accuracy')
plt.plot(epochs, CNN_6_validation_accuracies, marker = 'x', label = 'CNN batch size 6 validation accuracy')
plt.plot(epochs, CNN_8_train_accuracies, marker = 'o', label = 'CNN batch size 8 train accuracy')
plt.plot(epochs, CNN_8_validation_accuracies, marker = 'x', label = 'CNN batch size 8 validation accuracy')
plt.title("Training Accuracy of CNN Models with Batch Sizes 6 and 8")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# loading training accuracies for CNN-LSTM with batch size 6 and 8
CNN_LSTM_6_train_accuracies = np.load('CNN_LSTM_model_6_train_accuracies.npy')
CNN_LSTM_8_train_accuracies = np.load('CNN_LSTM_model_8_train_accuracies.npy')
CNN_LSTM_6_validation_accuracies = np.load('CNN_LSTM_model_6_validation_accuracies.npy')
CNN_LSTM_8_validation_accuracies = np.load('CNN_LSTM_model_8_validation_accuracies.npy')

# plotting training accuracy graph for CNN-LSTM with batch size 6 and 8 over course of epochs
epochs = range(1, len(CNN_LSTM_6_train_accuracies) + 1)
plt.figure(figsize=(8, 5))
plt.plot(epochs, CNN_LSTM_6_train_accuracies, marker='o', label='CNN-LSTM batch size 6 train accuracy')
plt.plot(epochs, CNN_LSTM_8_train_accuracies, marker='o', label='CNN-LSTM batch size 8 train accuracy')
plt.plot(epochs, CNN_LSTM_6_validation_accuracies, marker='x', label='CNN-LSTM batch size 6 validation accuracy')
plt.plot(epochs, CNN_LSTM_8_validation_accuracies, marker='x', label='CNN-LSTM batch size 8 validation accuracy')
plt.title("Training Accuracy of CNN-LSTM Models with Batch Sizes 6 and 8")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()



# creating a table to compare the peak validation accuracy, training time and training peak memory usage for the CNN and CNN-LSTM models with batch sizes of 6 and 8

data = [
    ["CNN (bs = 6)", f"{np.load('CNN_model_6_train_time.npy')[0]:.2f}", f"{np.load('CNN_model_6_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_model_6_best_validation_accuracy.npy')[0]:.2f}"],
    ["CNN (bs = 8)", f"{np.load('CNN_model_8_train_time.npy')[0]:.2f}", f"{np.load('CNN_model_8_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_model_8_best_validation_accuracy.npy')[0]:.2f}"],
    ["CNN-LSTM (bs = 6)", f"{np.load('CNN_LSTM_model_6_train_time.npy')[0]:.2f}", f"{np.load('CNN_LSTM_model_6_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_LSTM_model_6_best_validation_accuracy.npy')[0]:.2f}"],
    ["CNN-LSTM (bs = 8)", f"{np.load('CNN_LSTM_model_8_train_time.npy')[0]:.2f}", f"{np.load('CNN_LSTM_model_8_peak_mem_usage.npy')[0]:.2f}", f"{np.load('CNN_LSTM_model_8_best_validation_accuracy.npy')[0]:.2f}"]
]

columns = ["Model", "Training Time (minutes)", "Train Peak Mem Usage (MB)", "Peak validation Accuracy (%)"]

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



# creating a table to compare the test accuracy, precision, recall and F1 score for the CNN and CNN-LSTM models with batch sizes of 6 and 8
data = [
    ["CNN (bs = 6)", f"{np.load('CNN_6_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_6_test_precision.npy')[0]:.2f}", f"{np.load('CNN_6_test_recall.npy')[0]:.2f}", f"{np.load('CNN_6_test_f1.npy')[0]:.2f}"],
    ["CNN (bs = 8)", f"{np.load('CNN_8_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_8_test_precision.npy')[0]:.2f}", f"{np.load('CNN_8_test_recall.npy')[0]:.2f}", f"{np.load('CNN_8_test_f1.npy')[0]:.2f}"],
    ["CNN-LSTM (bs = 6)", f"{np.load('CNN_LSTM_6_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_LSTM_6_test_precision.npy')[0]:.2f}", f"{np.load('CNN_LSTM_6_test_recall.npy')[0]:.2f}", f"{np.load('CNN_LSTM_6_test_f1.npy')[0]:.2f}"],
    ["CNN-LSTM (bs = 8)", f"{np.load('CNN_LSTM_8_test_accuracy.npy')[0] * 100:.2f}", f"{np.load('CNN_LSTM_8_test_precision.npy')[0]:.2f}", f"{np.load('CNN_LSTM_8_test_recall.npy')[0]:.2f}", f"{np.load('CNN_LSTM_8_test_f1.npy')[0]:.2f}"]
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


# creating a table to compare the test time and memory usage for the CNN and CNN-LSTM models with batch sizes of 6 and 8
data = [
    ["CNN (bs = 6)", f"{np.load('CNN_6_test_time.npy'):.2f}", f"{np.load('CNN_6_test_peak_mem_usage.npy'):.2f}"],
    ["CNN (bs = 8)", f"{np.load('CNN_8_test_time.npy'):.2f}", f"{np.load('CNN_8_test_peak_mem_usage.npy'):.2f}"],
    ["CNN-LSTM (bs = 6)", f"{np.load('CNN_LSTM_6_test_time.npy'):.2f}", f"{np.load('CNN_LSTM_6_test_peak_mem_usage.npy'):.2f}"],
    ["CNN-LSTM (bs = 8)", f"{np.load('CNN_LSTM_8_test_time.npy'):.2f}", f"{np.load('CNN_LSTM_8_test_peak_mem_usage.npy'):.2f}"]
]

columns = ["Model", "Test Time (minutes)", "Test Peak Mem Usage (MB)"]

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

