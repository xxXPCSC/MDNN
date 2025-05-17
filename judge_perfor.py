import random
import train
import my_utils
from ma_config import _Cell_decode, cell_node
# from ma_config_test import _Cell_decode, cell_node
import sys
import torch
import numpy as np

# from ma_config_one import _Cell_decode, cell_node

in_channel = 16
out_channel = 16
kernel_size_s = 7
padding_s = 3

kernel_size_l = 9
padding_l = 4


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(seed)
    np.random.seed(seed)


def Performance_Estimation_Strategy(individual, Neural_nodes, epoch, _conv, _pool,train_data_path,train_label_path,	control =1):
	nodes = []
	for i in range(Neural_nodes):
		if i % 2 == 0:
			node = cell_node(in_channel, out_channel, kernel_size_s, padding_s)
		else:
			node = cell_node(in_channel, out_channel, kernel_size_l, padding_l)
		# nodes.append(node.hidden_layer)
		nodes.append(node)
	base_inchannel = 16
	# decode flow
	models,matrix_nodes = my_utils.decode_flow(individual, Neural_nodes, nodes, _conv, _pool,control,base_inchannel,train_data_path,train_label_path)
	# get model
	my_model = _Cell_decode(models[0],models[1],matrix_nodes,control=control,base_inchannel=base_inchannel)
	result = train.start_model(my_model, epoch,train_data_path,train_label_path)
	return result,matrix_nodes,my_model

