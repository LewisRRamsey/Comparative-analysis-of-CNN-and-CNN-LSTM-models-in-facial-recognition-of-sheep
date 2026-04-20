import numpy as np

loaded_array = np.load('CNN_LSTM_16_validation_accuracy.npy')
print(loaded_array)

loaded_array = np.load('CNN_model_6_peak_mem_usage.npy')
print(loaded_array)

loaded_array = np.load('CNN_model_6_train_time.npy')
print(loaded_array)