from torch import nn
import random
import torch.nn.functional as F
import torch
import os
import sys
import time
import math
import numpy as np
from math import sqrt
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

class TreeNode:
	def __init__(self, value,index):
		self.value = value  

		self.index=index
		self.children = [] 

	def add_child(self, child_node):
		self.children.append(child_node)

	def get_node_index(self):
		return self.index
	
	def get_node_value(self):
		return self.value

	def __repr__(self, level=0):
		ret = "\t" * level + repr(self.index)+"\n"
		for child in self.children:
			ret += child.__repr__(level + 1)
		return ret

class ChannelAttention(nn.Module):
	def __init__(self, channel, reduction=8):
		super(ChannelAttention, self).__init__()
		self.fc1 = nn.Linear(channel, channel // reduction, bias=False)
		self.fc2 = nn.Linear(channel // reduction, channel, bias=False)

		# self.fc1 = nn.Conv1d(channel, channel // reduction, 1,bias=False)
		# self.fc2 = nn.Conv1d(channel // reduction, channel, 1,bias=False)

	def forward(self, x):
		avg_pool = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)  # Global Average Pooling
		max_pool = F.adaptive_max_pool1d(x, 1).view(x.size(0), -1)  # Global Max Pooling
		x = avg_pool+max_pool
		x = F.relu(self.fc1(x))
		x = torch.sigmoid(self.fc2(x)).view(x.size(0), -1,  1)  # Reshape to (batch_size, channels, 1)
		return x

class SpatialAttention(nn.Module):
	def __init__(self):
		super(SpatialAttention, self).__init__()
		self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)  # Convolutional layer
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		avg_pool = torch.mean(x, dim=1, keepdim=True)  # Average over channels
		max_pool, _ = torch.max(x, dim=1, keepdim=True)  # Max over channels
		x = torch.cat((avg_pool, max_pool), dim=1)  # Concatenate along channel dimension
		x = self.conv(x)  # Apply convolution
		return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        # Channel Attention
        channel_weights = self.channel_attention(x)
        x = x * channel_weights

        # Spatial Attention
        spatial_weights = self.spatial_attention(x)
        x = x * spatial_weights

        return x

class DepthwiseConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size):
		super(DepthwiseConv2D, self).__init__()
		self.kernel_size=kernel_size
		self.in_channel = in_channels
		self.out_channel =out_channels
		self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels)
		self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

	def get_property(self):
		return [self.in_channel,self.out_channel,self.kernel_size]

	def forward(self, x):
		out = self.depthwise_conv(x)
		out = self.point_conv(out)
		return out

class DilatedConv2D(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, dilation_rate=1):
		super(DilatedConv2D, self).__init__()
		self.kernel_size=kernel_size
		self.in_channel = in_channels
		self.out_channel = out_channels
		# padding = ((kernel_size - 1) * dilation_rate - (kernel_size - 1)) // 2
		self.dilated_conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation_rate)

	def get_property(self):
		return [self.in_channel,self.out_channel,self.kernel_size]
	
	def forward(self, x):
		return self.dilated_conv(x)

class NormalConv1d(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size,stride,padding):
		super(NormalConv1d,self).__init__()
		self.in_channel = in_channels
		self.out_channel = out_channels
		self.kernel_size = kernel_size
		self.padding=padding
		self.my_stride = stride
		self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=False)
		self.relu = nn.ReLU()
		self.bn = nn.BatchNorm1d(out_channels)
	def get_property(self):
		return [self.in_channel,self.out_channel,self.kernel_size,self.my_stride,self.padding]
	
	def forward(self, x):
		out =x 
		out = self.conv(out)
		out = self.bn(out)
		out = self.relu(out)
		return out

# conv space
class _CONV_Space:
	def __init__(self, in_channels, out_channels, dilation_rate=1):
		self.FA = False
		self.a = NormalConv1d(in_channels, out_channels, 5,2,1)
		self.b = NormalConv1d(in_channels, out_channels, 7,3,1)
		self.c = NormalConv1d(in_channels, out_channels, 9,3,0)
		self.d = NormalConv1d(in_channels, out_channels, 9,4,0)
		self.m = NormalConv1d(in_channels, out_channels, 11,5,0)

	def random_select(self):
		attributes = list(self.__dict__.items())
		return random.choice(attributes)

	def random_select_NoFA(self):
		attributes = {k: v for k, v in self.__dict__.items() if k != 'FA'}
		if not attributes:
			return None
		random_attr = random.choice(list(attributes.items()))
		return random_attr

# pool space
class _POOL_Space:
	def __init__(self):
		self.e = "NoConnection"
		self.f = nn.MaxPool1d(kernel_size=1)
		self.g = nn.AvgPool1d(kernel_size=3)
		self.h = nn.MaxPool1d(kernel_size=3)
		self.i = nn.MaxPool1d(kernel_size=5)
		self.z = nn.MaxPool1d(kernel_size=7)

	def random_select(self):
		attributes = list(self.__dict__.items())
		return random.choice(attributes)

	def random_select_NoFA(self):
		attributes = {k: v for k, v in self.__dict__.items() if k != 'FA'}
		if not attributes:
			return None
		random_attr = random.choice(list(attributes.items()))
		return random_attr

class cell_node(nn.Module):
	def __init__(self, in_channels, out_channels, kernel_size, padding,stride=1):
		super(cell_node, self).__init__()
		self.kernel_size=kernel_size
		self.in_channel=in_channels
		self.out_channel =out_channels
		self.padding = padding
		self.CBAM_Attention = CBAM(in_channels,in_channels//2)
		self.hidden_layer = nn.Sequential(
			# nn.BatchNorm1d(self.out_channel),
			nn.Conv1d(in_channels, self.out_channel, kernel_size=kernel_size, padding=padding, stride=stride,bias=False),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
			nn.Dropout(0.3),
		)
		self.block = nn.Sequential(
			nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size , padding=self.padding,bias=False),
			nn.BatchNorm1d(out_channels),
			nn.ReLU(),
			
			nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=self.padding,bias=False),
			nn.ReLU(),
			nn.Dropout(0.3),
		)
		self.relu = nn.ReLU()
	def get_property(self):
		return [self.in_channel,self.out_channel,self.kernel_size,self.padding]

	def forward(self, x):
		out =x
		out = self.CBAM_Attention(out)
		out = self.block(out)
		out = self.relu(x+out)
		return out


class FactorizedReduce(nn.Module):
	def __init__(self, C_in, C_out, affine=True):
		super(FactorizedReduce, self).__init__()
		assert C_out % 2 == 0
		self.relu = nn.ReLU(inplace=False)
		self.in_channel = C_in
		self.seq = nn.Sequential(
			nn.Conv1d(C_in, C_out *2, 5, stride=3, bias=False),
			nn.ReLU(),
			nn.Dropout(0.3),
			nn.BatchNorm1d(num_features=C_out *2, affine=affine),
		)
		

	def get_property(self):
		return [self.in_channel]
	
	def forward(self, x):
		output = self.seq(x)
		return output

class Linear_layer(nn.Module):
	def __init__(self,in_channels,linear_number,device):
		super(Linear_layer, self).__init__()
		self.in_channels = in_channels
		self.linear_number = linear_number
		self.linear_layer = nn.Sequential(
			nn.Dropout(0.3).to(device),
			nn.Flatten().to(device),
			nn.Linear(linear_number, 256).to(device),
			nn.Linear(256,1).to(device)
		)

	def get_property(self):
		return [self.in_channels,self.linear_number]
	
	def forward(self,x):
		return self.linear_layer(x)


class Regular_layer(nn.Module):
	def __init__(self,inchannel,device):
		super(Regular_layer, self).__init__()
		self.in_channel = inchannel
		self.regular_layer = nn.Sequential(
		nn.BatchNorm1d(inchannel).to(device),
		nn.Dropout(0.3).to(device),
		)

	def get_property(self):
		return [self.in_channel]
	
	def forward(self,x):
		out = self.regular_layer(x)
		return out

class NoPlayer(nn.Module):
	def __init__(self):
		super(NoPlayer,self).__init__()
	
	def forward(self,x):
		return x
	
class FusionNet(nn.Module):
	def __init__(self, feature_size):
		super(FusionNet, self).__init__()
		self.fc = nn.Linear(feature_size, feature_size)
		self.relu = nn.ReLU()
	def forward(self, x):
		out = self.fc(x)
		out = self.relu(out)
		return out
	
class _Cell_decode(nn.Module):
	def __init__(self, model_list,ca_model_list,matrix_nodes,control,base_inchannel):
		super(_Cell_decode, self).__init__()
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.set_seed(3)
		self.matrix_nodes = matrix_nodes
		self.control = control
		self.my_layers_list = model_list
		self.my_layers_list.to(self.device)
		self.conv = nn.Conv1d(1,out_channels=base_inchannel,kernel_size=1)
		self.batch_nn = nn.BatchNorm1d(base_inchannel)

		self.my_ca_list = ca_model_list
		self.my_ca_list.to(self.device)

	def pad_tensor(self, my_tensor, max_C, max_H, max_W):
		pad_C = max_C - my_tensor.size(1)
		pad_H = max_H - my_tensor.size(2)
		pad_W = max_W - my_tensor.size(3)
		padding = (0, pad_W, 0, pad_H, 0, pad_C)
		return F.pad(my_tensor, padding, mode='constant', value=0)

	def resize_tensor(self,input_tensor, target_height, target_width):

		batch_size, channel, height, width = input_tensor.shape

		if height > target_height:
			crop_top = (height - target_height) // 2
			input_tensor = input_tensor[:, :, crop_top:crop_top + target_height, :]
		
		if width > target_width:
			crop_left = (width - target_width) // 2
			input_tensor = input_tensor[:, :, :, crop_left:crop_left + target_width]

		_, _, new_height, new_width = input_tensor.shape

		pad_top = max(0, (target_height - new_height) // 2)
		pad_bottom = max(0, target_height - new_height - pad_top)
		pad_left = max(0, (target_width - new_width) // 2)
		pad_right = max(0, target_width - new_width - pad_left)

		if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
			input_tensor = F.pad(input_tensor, (pad_left, pad_right, pad_top, pad_bottom))

		return input_tensor

	def input_pad(self,out,ca_flag=-1):
		B_ = []
		C_ = []
		H_ = []
		W_ = []
		TEMP_H_MAX_MIN =None
		TEMP_W_MAX_MIN =None
		for i, my_tensor in enumerate(out):
			B_.append(my_tensor.size(0))
			C_.append(my_tensor.size(1))
			H_.append(my_tensor.size(2))
			W_.append(my_tensor.size(3))
		B_max = max(B_)
		C_max = max(C_)

		TEMP_H_MAX_MIN = max(H_)
		TEMP_W_MAX_MIN = max(W_)

		result = torch.zeros((B_max, C_max, TEMP_H_MAX_MIN, TEMP_W_MAX_MIN)).to(self.device)
		
		for my_tensor in out:
			batch, channel, h, w = my_tensor.size()
			result[:, :channel, :h, :w] += my_tensor
		return result
	
	def input_pad_1d(self,out,ca_flag=-1):
		if ca_flag==-1:
			concatenated_tensor = torch.cat(out, dim=-1)
			return concatenated_tensor
		B_ = []
		C_ = []
		W_ = []
		TEMP_W_MAX_MIN =None
		for i, my_tensor in enumerate(out):
			B_.append(my_tensor.size(0))
			C_.append(my_tensor.size(1))
			W_.append(my_tensor.size(2))
		B_max = max(B_)
		C_max = max(C_)

		TEMP_W_MAX_MIN = max(W_)

		result = torch.zeros((B_max, C_max,  TEMP_W_MAX_MIN)).to(self.device)
		
		for my_tensor in out:
			batch, channel,  w = my_tensor.size()
			result[:, :channel, :w] += my_tensor
		return result
	def expand_channels(self,input_tensor, target_channels):

		batch_size, original_channels, width = input_tensor.shape

		repeat_times = target_channels // original_channels
		

		expanded_tensor = input_tensor.repeat(1, repeat_times, 1)

		if target_channels % original_channels != 0:
			expanded_tensor = expanded_tensor[:, :target_channels, :]

		return expanded_tensor
	def set_seed(self,seed):
		torch.manual_seed(seed) 
		torch.cuda.manual_seed(seed) 
		torch.cuda.manual_seed_all(seed)  
		torch.backends.cudnn.deterministic = True 
		torch.backends.cudnn.benchmark = False 
		random.seed(seed) 
		np.random.seed(seed)

	def feature_extraction_lsit(self,layers,x,ca_flag=-1):
		out = x
		input_1=[]
		for layer in layers:
			temp_out = layer(out).to(self.device)
			input_1.append(temp_out)
		out = self.input_pad_1d(input_1,-1)
		return out
	
	def forward(self, x):
		out = x
		out = self.conv(x)
		out = self.batch_nn(out)
		for layer in self.my_layers_list:
			if isinstance(layer,nn.ModuleList):
				out = self.feature_extraction_lsit(layer,out)
			elif isinstance(layer,Linear_layer):
				out = layer(out)
			else:
				out = layer(out)
		return out

if __name__=='__main__':	
	test = _CONV_Space(16,16)
	temp_value = test.random_select()
	print(temp_value)