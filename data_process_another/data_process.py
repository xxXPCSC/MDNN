import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch

globel_numpy_data = None
global_numpy_label = None
class data_loader(torch.utils.data.Dataset):
	def __init__(self, data, label):
		self.data = torch.from_numpy(data)
		self.label = torch.from_numpy(label)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		genotype = self.data[index].float()
		label = self.label[index].float()
		return genotype, label

def get_train_data(train_data_path):
    print("train_data_path is ",train_data_path)
    df_csv = pd.read_csv(train_data_path)
    # df_csv = pd.read_csv('./data/wheat599/wheat599_pc95.csv')
    # df_csv = pd.read_csv('./data/maize1404/1404cubic.csv')
    numpy_array = df_csv.to_numpy()
    return numpy_array

def pad_to_multiple_of_any(row,index):
    length = len(row)
    pad_length = (index - (length % index)) % index
    padded_row = np.pad(row, (0, pad_length), mode='constant', constant_values=0)
    return padded_row


def find_min_difference_pair(N):
    min_diff = float('inf')
    best_pair = (1, N)

    for a in range(1, int(math.sqrt(N)) + 1):
        if N % a == 0:
            b = N // a
            diff = abs(a - b)
            if diff < min_diff:
                min_diff = diff
                best_pair = (a, b)

    return best_pair

def convert_to_high_demesion(row,index):
    length = len(row)
    assert length % index == 0
    temp_dim = length // index
    second,third = find_min_difference_pair(temp_dim)
    return row.reshape(index, second, third).astype(np.int32)


def reshape_to_3d(data, channel=1, max_diff=3):
    length = len(data)
    length_per_channel = (length + channel - 1) // channel

    best_height, best_width = None, None
    min_diff = float('inf')

    for height in range(1, length_per_channel + 1):
        width = (length_per_channel + height - 1) // height  
        if abs(height - width) <= max_diff and abs(height - width) < min_diff:
            best_height, best_width = height, width
            min_diff = abs(height - width)

    new_size = best_height * best_width * channel
    if new_size > length:
        padding = new_size - length
        data = np.concatenate([data, np.zeros(padding)], axis=0)
    elif new_size < length:
        data = data[:new_size]


    return data.reshape(channel, best_height, best_width)


def to_normal(data):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    return X_scaled


def get_train_data_matrix(train_data_path):
    three_d_matrices = []
    row_data = get_train_data(train_data_path)
    for row in row_data:
        row = row.reshape(1,row.shape[0]).astype(np.float32)
        three_d_matrices.append(row)
    
    
    return np.array(three_d_matrices)


def get_train_label_matrix(train_label_path):
    #1 TKW 0.67 2 TW0.63 3 gl 0.77 4 gw 0.71 5 GH 0.69 6 GP 0.5
    # df_csv = pd.read_csv('./data/wheat2000pca/2000_5_phe.txt')
    df_csv = pd.read_csv(train_label_path)
    numpy_array = df_csv.to_numpy()
    numpy_array = numpy_array.reshape(-1)
    return numpy_array


def get_train_data(percentage,train_data_path,train_label_path):
    numpy_array_data = None
    numpy_array_label = None
    
    if  globel_numpy_data is None and global_numpy_label is None:
        numpy_array_data = get_train_data_matrix(train_data_path)
        numpy_array_label = get_train_label_matrix(train_label_path)
    else:
        numpy_array_data = globel_numpy_data
        numpy_array_label = global_numpy_label

    train_data, test_data, train_label, test_label = train_test_split(numpy_array_data, numpy_array_label,
                                                                      train_size=percentage)
    return train_data, train_label, test_data, test_label

def get_train_data_shape(train_data_path,train_label_path,percentage=0.9):
    global globel_numpy_data,global_numpy_label
    
    numpy_array_data = get_train_data_matrix(train_data_path)
    numpy_array_label = get_train_label_matrix(train_label_path)
    
    globel_numpy_data = numpy_array_data
    global_numpy_label = numpy_array_label
    
    print(numpy_array_data.shape)
    print(numpy_array_label.shape)
    
    idx = np.random.choice(len(numpy_array_data), 32, replace=False)
    batch_data = numpy_array_data[idx]
    
    return batch_data

if __name__=='__main__':
    # get_train_data_by_gan()
    get_train_data(0.8)