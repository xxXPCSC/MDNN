import numpy as np
import os
from datetime import datetime
import re

from process_nn import get_true_nn
# from process_nn_test import get_true_nn
_CONV_STRING = {'FA','a', 'b', 'c', 'd','m'}
_POOL_STRING = {'e', 'f', 'g', 'h','i','z'}

def decode_flow(individuals,Neural_nodes,nodes,_conv,_pool,control,base_inchannel,train_data_path,train_label_path):  
    # getattr(obj, attribute_name)
    matrixs_nns = []
    matrix_nodes = []
    real_nns = []
    for individual in individuals:
        matrix_n= fill_matrix(individual,Neural_nodes,nodes)
        matrixs_nns.append(matrix_n)
    print(matrixs_nns)
    for matrix in matrixs_nns:
        matrix_node = np.zeros((Neural_nodes, Neural_nodes + 2), dtype=object)
        real_nn,_matrix = get_struc(matrix,matrix_node,_conv,_pool,nodes)
        real_nns.append(real_nn)
        matrix_nodes.append(_matrix)
    # Directly return the model and the adjacency matrix.
    print(matrix_nodes)
    return get_true_nn(real_nns,matrix_nodes,control,base_inchannel,train_data_path,train_label_path),matrix_nodes

def fill_matrix(individual,Neural_nodes,nodes):
    matrixs_nn = np.zeros((Neural_nodes, Neural_nodes + 1), dtype=object)
    temp = 0
    for i in range(Neural_nodes):
        col = 0
        for j in range(temp, temp + i + 1):
            matrixs_nn[i][col] = individual[j]
            col += 1
        temp = temp+i+1
    return matrixs_nn

def get_struc(matrix,matrix_node,_conv,_pool,nodes):
    real_nn = []
    # "Used as an adjacency matrix: 0 indicates no connection; \[-1, 1] indicates a connection; -1 denotes pooling; 1 denotes convolution; 2 denotes node."
    matrix_temp = matrix_node
    for i in range(matrix.shape[0]):
        temp_nn = []
        for j in range(matrix.shape[1]):  # Traverse columns
            if matrix[i][j] == 0:
                matrix_temp[i][j] = 2
                break
            if matrix[i][j] in _CONV_STRING:
                nn_of,flag = get_output_nn(matrix[i][j],_conv,j,real_nn)
                temp_nn.append(nn_of)
                matrix_temp[i][j] = flag
            elif matrix[i][j] in _POOL_STRING:
                nn_of, flag = get_output_nn(matrix[i][j], _pool, j, real_nn)
                temp_nn.append(nn_of)
                # matrix_temp[i][j] = flag
                matrix_temp[i][j] = flag if flag==0 else -1
        temp_nn.append(nodes[i])
        real_nn.append(temp_nn)
    return real_nn,matrix_temp

def get_row_nn(operator, name):
    op_nn = getattr(operator, name)
    if isinstance(op_nn, bool):
        return False
    return op_nn

def get_output_nn(flow,operator,j,real_nn):
    if isinstance(get_row_nn(operator,flow),bool) or get_row_nn(operator,flow)=="NoConnection":
        return False,0
    else:
        return get_row_nn(operator, flow), 1

def creat_floder_for_matrix(directory,index):


    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M')

    new_folder_name = f"{current_time}_{index}"
    new_folder_path = os.path.join(directory, new_folder_name)

    os.makedirs(new_folder_path, exist_ok=True)
    return new_folder_path

def creat_floder_for_train(base_dir,new_folder_name):

    folder_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


    number_suffixes = []
    for folder in folder_names:
        match = re.search(r'(\d+)$', folder)
        if match:
            number_suffixes.append(int(match.group(1)))

    if number_suffixes:
        new_number = max(number_suffixes) + 1
    else:
        new_number = 1  

    new_folder = f"{new_folder_name}_{new_number}"
    new_folder_path = os.path.join(base_dir, new_folder)


    os.makedirs(new_folder_path, exist_ok=True)

    return new_folder_path

def save_flag_file(matrix_node, data_list, fitness,save_folder):
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    
    numpy_file_path = os.path.join(save_folder, 'fitness.txt')
    data_array = np.array([fitness])
    np.savetxt(numpy_file_path, data_array, fmt='%.6f')
    print(f'fitness.txt has been save in {numpy_file_path}')
    list_file_path = os.path.join(save_folder, 'flag_nns.txt')
    with open(list_file_path, 'w') as f:
        for item in data_list:
            f.write(f"{item}\n")
        for item in matrix_node:
            f.write(f"{item}\n")
    print(f'flag_nns.txt has been save in {list_file_path}')

    return numpy_file_path, list_file_path


def check_file_existence(train_data_paths, label_file_paths):
    missing_files = []

    for path in train_data_paths:
        if not os.path.isfile(path):
            missing_files.append(path)
    for sublist in label_file_paths:
        for path in sublist:
            if not os.path.isfile(path):
                missing_files.append(path)

    if missing_files:
        print("The following file does not exist:")
        for f in missing_files:
            print(f"- {f}")
        return False
    else:
        print("All files exist.")
        return True

def format_seconds(seconds):
    days = int(seconds) // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hours = int(seconds) // 3600
    seconds %= 3600
    minutes = int(seconds) // 60
    seconds %= 60
    return f"{days}day {hours:02d}hour {minutes:02d}minute {seconds:05.2f}second"


if __name__=="__main__":
    print("this is utils")