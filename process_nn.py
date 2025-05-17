import torch
# from data_process_another.data_process_wheat import get_train_data_shape
# from data_process_another.data_process_wheat2000PCA import get_train_data_shape
from data_process_another.data_process import get_train_data_shape

from ma_config import FactorizedReduce,Linear_layer,Regular_layer,NoPlayer,cell_node,TreeNode
from torch import nn
import torch.nn.functional as F
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

temp_att_layers=nn.ModuleList()

def get_shape(size):
	height = size[2]
	width = size[3]
	return height * width, size[1]

def get_shape_1d(size):
    inchannel_ = size[1]
    width = size[2]
    return  inchannel_,width

def pad_tensor(my_tensor, max_C, max_H, max_W):
    pad_C = max_C - my_tensor.size(1)
    pad_H = max_H - my_tensor.size(2)
    pad_W = max_W - my_tensor.size(3)
    padding = (0, pad_W, 0, pad_H, 0, pad_C)
    return F.pad(my_tensor, padding, mode='constant', value=0)

def resize_tensor(input_tensor, target_height, target_width):
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

def input_pad(out,ca_flag=-1):
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

    result = torch.zeros((B_max, C_max, TEMP_H_MAX_MIN, TEMP_W_MAX_MIN)).to(device)
    
    for my_tensor in out:
        batch, channel, h, w = my_tensor.size()
        result[:, :channel, :h, :w] += my_tensor
    return result

def input_pad_1d(out,ca_flag=-1):
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

    result = torch.zeros((B_max, C_max,  TEMP_W_MAX_MIN)).to(device)
    
    for my_tensor in out:
        batch, channel,  w = my_tensor.size()
        result[:, :channel, :w] += my_tensor
    return result

def check_channel(layer,inchnnel):
    if layer.get_property()[0] == inchnnel:
        return True
    return False

def assign_nn(layer,inchannel,nn_type):
    model_class = type(layer)
    new_model = None
    propertys = layer.get_property()
    
    # node
    if nn_type == 2:
        # in_channels, out_channels, kernel_size, padding
        new_model = model_class(in_channels=inchannel, out_channels=inchannel, kernel_size=propertys[2],
                                padding=propertys[3])
    # flow
    elif nn_type==1:
        # in_channels, out_channels, kernel_size
        if len(propertys) <4:
            new_model = model_class(in_channels=inchannel, out_channels=inchannel, kernel_size=propertys[2])
        else:
            # Normal2d
            new_model = model_class(in_channels=inchannel, out_channels=inchannel, kernel_size=propertys[2],stride = propertys[3],padding=propertys[4])

    # linear
    elif nn_type == 'LR':
        new_model = model_class(in_channels=inchannel,device=device)
    return new_model

def assign_nn_linear(layer,inchannel,all_number):
    model_class = type(layer)
    new_model = None
    propertys = layer.get_property()
    new_model = model_class(in_channels=inchannel,linear_number=all_number,device=device)
    return new_model

# get new nn
def get_new_nn(layer,out):
    new_layer = layer.to(device)
    in_channel = out.shape[1]
    if isinstance(layer,nn.MaxPool1d) or isinstance(layer,nn.AvgPool1d) or check_channel(layer,in_channel):
        out = new_layer(out)
    else:
        if isinstance(layer,cell_node):
            new_layer = assign_nn(layer,in_channel,2).to(device)
        else:
            new_layer = assign_nn(layer,in_channel,1).to(device)
    return out,new_layer
    

def check_module_sequential(layer,out):
    temp_out = out
    temp_list = nn.ModuleList(list(layer.children()))
    new_layer_list = nn.ModuleList()
    for temp_index,temp_layer in enumerate(temp_list):
        if isinstance(temp_layer,nn.MaxPool1d) or isinstance(temp_layer,nn.AvgPool1d):
            out = temp_layer(out)
            new_layer_list.append(copy.deepcopy(temp_layer))
            continue
        out,new_layer = get_new_nn(temp_layer,out)
        new_layer_list.append(copy.deepcopy(new_layer))

    sequential_model = nn.Sequential(*new_layer_list)

    out = sequential_model(temp_out)
    return out,sequential_model

def feature_extraction_list(layers,x,ca_flag=-1):
    out = x
    input_1=[]
    for layer in layers:
        temp_out = layer(out).to(device)
        input_1.append(temp_out)
    out = input_pad_1d(input_1,-1)
    return out

def creat_tree_from_nn_cell(layers):
    root=None

    # add root node
    root=TreeNode(NoPlayer(),-1)

    for index,inner_layers in enumerate(layers):
        temp_cell_node = inner_layers[len(inner_layers)-1]
        for inner_index,inner_layer in enumerate(inner_layers):
            child_value = nn.ModuleList()
            if isinstance(inner_layer,NoPlayer):
                continue
            if isinstance(inner_layer,cell_node):
                break
            parent_index = inner_index-1
            child_index = index

            child_value.append(inner_layer)
            child_value.append(temp_cell_node)

            parent_nodes = []
            find_parents(root,parent_index,parent_nodes)
            for parent in parent_nodes:
                parent.add_child(TreeNode(child_value,child_index))
    print(root)
    return root

def find_parents(node,parent_index, parent_nodes):
    if node.get_node_index() == parent_index:
        parent_nodes.append(node)
    for child in node.children:
        find_parents(child, parent_index, parent_nodes)

def flatten_module_list(module_list):
    flat_list = nn.ModuleList()  

    for item in module_list:
        if isinstance(item, nn.ModuleList):
            flat_list.extend(flatten_module_list(item))
        elif isinstance(item,NoPlayer):
            continue
        else:
            flat_list.append(item)
    
    return flat_list

def dfs_tree_layer(node, path, leaf_nodes):
    if node is None:
        return
    path.append(node.get_node_value())
    if not node.children:  # leaf node
        flat_list = flatten_module_list(path)
        leaf_nodes.append(nn.Sequential(*flat_list))  # Record the path.
    for child in node.children:
        dfs_tree_layer(child, path, leaf_nodes)

    path.pop()  

def test_is_same(layers):
    print(layers)
    shortest_seq = min(layers, key=lambda x: len(x))
    for index,mod in enumerate(shortest_seq):

        for i, seq in enumerate(layers):
            if seq is shortest_seq:
                continue  #
            for j in range(len(seq)):
                if mod is seq[j]:
                    print(f"Module {index} is the {j}‑th module in module list {i}.")
                else:
                    print(f"Module {index} is not the same module as the one in module list {i}.")

def get_temp_nn(result_nns,matrix_nodes,control,base_inchannel,data_shape):
    origin_nums,origin_inchannel=None,None
    if len(data_shape)==4:
        origin_nums,origin_inchannel = get_shape(data_shape)
    else:
        origin_inchannel,origin_nums = get_shape_1d(data_shape)
    my_layers_list = nn.ModuleList()
    temp_cell_layers = nn.ModuleList()
    
    temp_frcs = nn.ModuleList()
    temp_regular_layers = nn.ModuleList()
    last_in_linear  = origin_inchannel*pow(2,control-1)*round(origin_nums/(pow(2,control-1)*pow(2,control-1)))
    temp_linear_layer = Linear_layer(base_inchannel*pow(2,control-1),last_in_linear,device)

    temp_channel = base_inchannel

    for i in range(control - 1):
        temp_frcs.append(FactorizedReduce(temp_channel, temp_channel))
        temp_regular_layers.append(Regular_layer(temp_channel, device))
        temp_channel = temp_channel * 2

    if control==1:
        temp_frcs.append(FactorizedReduce(base_inchannel, base_inchannel))
        temp_regular_layers.append(Regular_layer(base_inchannel, device))

    # conv->N->FRC->R->RL
    temp_channel = base_inchannel
    for every_control in range(control):
        # base_inchannel=(every_control+1)*base_inchannel
        for index, matrix in enumerate(matrix_nodes):
            # Store the row information of a cell.
            temp_row = nn.ModuleList()
            for i in range(matrix.shape[0]):
                # Store the column information of each row.
                temp_clo = nn.ModuleList()
                for j in range(matrix.shape[1]):
                    if matrix[i][j]==0:
                        temp_clo.append(NoPlayer())
                        continue
                    elif matrix[i][j]==-1:
                        temp_layer = result_nns[index][i][j]
                        temp_clo.append(temp_layer)
                    elif matrix[i][j]==1:
                        temp_layer = result_nns[index][i][j]
                        if check_channel(temp_layer,temp_channel):
                            temp_clo.append(temp_layer)
                        else:
                            new_layer = assign_nn(temp_layer,temp_channel,1)
                            temp_clo.append(new_layer)
                    elif matrix[i][j]==2:
                        temp_layer = result_nns[index][i][j]
                        if check_channel(temp_layer,temp_channel):
                            temp_clo.append(temp_layer)
                        else:
                            new_layer = assign_nn(temp_layer,temp_channel,2)
                            temp_clo.append(new_layer)
                        break
                temp_row.append(temp_clo)
            tree_layer = creat_tree_from_nn_cell(temp_row)
            leaf_nodes = nn.ModuleList()
            dfs_tree_layer(tree_layer,[],leaf_nodes)
            leaf_nodes.to(device)
            # test_is_same(leaf_nodes)
            temp_cell_layers.append(leaf_nodes)
            # first is N then is R the cell channel need times 2
            temp_channel = temp_channel*2
    # NR ->regular
    if control==1:
        my_layers_list.append(temp_cell_layers[0])
        # my_layers_list.append(temp_frcs[0])
        # my_layers_list.append(temp_regular_layers[0])

    for i in range(control-1):
        my_layers_list.append(temp_cell_layers[i])
        my_layers_list.append(temp_frcs[i])
        # my_layers_list.append(temp_regular_layers[i])
        my_layers_list.append(temp_cell_layers[i+1])

    my_layers_list.append(temp_linear_layer)

    for var in list(locals()):
        if var.startswith('temp'):
            del locals()[var]
    return my_layers_list

def get_true_nn(real_nns,matrix_nodes,control,base_inchannel,train_data_path,train_label_path):
    sample_data = get_train_data_shape(train_data_path,train_label_path)
    sample_data = torch.from_numpy(sample_data).to(device)

    data_shape = sample_data.shape
    batch_size = data_shape[0]
    
    my_layers_list = get_temp_nn(real_nns,matrix_nodes,control,base_inchannel,data_shape)
    temp_layers_list = my_layers_list

    out = sample_data
    temp_conv = nn.Conv1d(1, base_inchannel, kernel_size=1).to(device)
    temp_batch = nn.BatchNorm1d(base_inchannel).to(device)
    out = temp_batch(temp_conv(out))
    origin_out = out
    del temp_conv,temp_batch
    del my_layers_list
    my_layers_list =  nn.ModuleList()
    
    ca_flag = 0
    new_layers = None
    # A master list where each element represents a Cell, with each Cell processed individually.
    for index, layers in enumerate(temp_layers_list):
        if isinstance(layers, nn.ModuleList):
           my_layers_list.append(layers)
           out = feature_extraction_list(layers,out)
        #    temp_fuse_linear = FusionNet(out.shape[2])
        #    my_layers_list.append(temp_fuse_linear)
        elif isinstance(layers,Linear_layer):
            channel,nums = get_shape_1d(out.shape)
            temp_frc=nn.Sequential(
                nn.Conv1d(in_channels=channel, out_channels=channel//2, kernel_size=1, stride=1, padding=0).to(device),
                nn.Conv1d(in_channels=channel//2, out_channels=channel//2, kernel_size=3, stride=4, padding=1).to(device),
            )
            out = temp_frc(out)
            my_layers_list.append(temp_frc)
            channel,nums = get_shape_1d(out.shape)
            

            new_layers=None
            if check_channel(layer=layers,inchnnel=nums*channel) is False:
                # Reassign the Linear layer.
                new_layers = assign_nn_linear(layers,channel,nums*channel).to(device)
            else:
                new_layers=layers
            my_layers_list.append(new_layers)
            new_layers.to(device)
            out = new_layers(out)
        else:
            my_layers_list.append(layers)           
            layers.to(device)
            out = layers(out)	
    my_ca_list = temp_att_layers
    # my_layers_list = flatten_module_list(my_layers_list)
    return my_layers_list,my_ca_list

if __name__=="__main__":
    print('this is assign __nn')