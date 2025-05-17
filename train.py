import warnings
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import numpy as np
import os
import torch.optim as optim
from focal_loss import FocalLoss
import random
from data_process_another.data_process import get_train_data
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
path = os.path.dirname(__file__)


def init_weights(net):
	if type(net) == torch.nn.Linear or type(net) == torch.nn.Conv1d:
		torch.nn.init.xavier_uniform_(net.weight)


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


# In this module, users could train their own datasets on our baseline
class Train:
	def __init__(self, train_data_path,train_label_path,percentage_of_train=0.8, num_workers=8, batch_size=24):
		# ignore warnings
		warnings.filterwarnings("ignore")
		# select which device to train
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
	
		self.percentage_of_train = percentage_of_train
		# the training batch size
		self.batch_size = batch_size
		# workers amount
		self.num_workers = num_workers
		self.train_data_path = train_data_path
		self.train_label_path = train_label_path
  
	def dataloader_to_train(self,percentage_of_train):
		train_data, train_label, test_data, test_label = get_train_data(percentage_of_train,self.train_data_path,self.train_label_path)
		# dataloader of train dataset
		train_dataloader = DataLoader(data_loader(train_data, train_label), batch_size=self.batch_size, shuffle=True,
									  num_workers=self.num_workers)
		# dataloader of test dataset
		test_dataloader = DataLoader(data_loader(test_data, test_label), batch_size=self.batch_size, shuffle=True,
									 num_workers=self.num_workers)
		return train_dataloader, test_dataloader

	
	# train an epoch
	def train_for_epoch(self, train_dataloader, updater,scheduler, loss, net):
		loss_ = 0.0
		for num_data, (genomap, target_trait) in enumerate(train_dataloader):
			# print("num data is ",num_data)
			genomap, target_trait = genomap.to(self.device), target_trait.to(self.device)
			trait_hat = net(genomap)
			loss_for_batch = loss(trait_hat, target_trait)
			loss_ += loss_for_batch

			updater.zero_grad()
			loss_for_batch.backward()
			updater.step()
			scheduler.step()
		return loss_ / (num_data + 1)

	def train_model(self, epoch, weight_decay=1e-5,net=None,label_index=-1):
		epoch_total = epoch
		if net is None:
			return None
		print(f"Your model structure is: \n{net}")
		net.to(self.device)
		# initialize weights
		net.apply(init_weights)
		# set Smooth S1 Loss as the loss function
		loss = torch.nn.SmoothL1Loss()
		updater = torch.optim.Adam(net.parameters(),lr=0.001, weight_decay=weight_decay)
		scheduler = CosineAnnealingWarmRestarts(updater, T_0=20, T_mult=2, eta_min=0)

		eval_dict = {'train_loss': [], 'test_loss': [], 'mse': [], 'pcc': []}
		mse_loss = torch.nn.MSELoss()
		# train and test dataloaders
		train_dataloader, test_dataloader = self.dataloader_to_train(self.percentage_of_train)
		min_pcc = 0.0
		while epoch:
			print(f"Now training epoch {epoch_total - epoch + 1}")
			net.train()
			avg_train_loss = self.train_for_epoch(train_dataloader, updater, scheduler,loss, net)
			net.eval()
			# for the test dataset
			with torch.no_grad():
				mse = 0.0
				loss_test = 0.0
				hat = np.array([])
				truth = np.array([])
				for index, (test_genomap, test_trait) in enumerate(test_dataloader):
					test_genomap, test_trait = test_genomap.to(self.device), test_trait.to(self.device)
					y_hat = net(test_genomap)
					loss_test += loss(y_hat, test_trait)
					hat = np.insert(hat, 0, y_hat.to('cpu').detach().numpy().reshape(len(y_hat), ), axis=0)
					truth = np.insert(truth, 0, test_trait.to('cpu').numpy(), axis=0)
					mse += mse_loss(y_hat, test_trait)
			loss_test = loss_test / (index + 1)
			eval_dict['test_loss'].append(float(loss_test.to('cpu').detach().numpy()))
			eval_dict['train_loss'].append(float(avg_train_loss.to('cpu').detach().numpy()))
			eval_dict['mse'].append(mse)
			mse / (index + 1)
			temp = np.corrcoef(hat, truth)
			pcc = temp[0][1]
			eval_dict['pcc'].append(pcc)
			if pcc >= min_pcc:
				print(temp)
				print("higher pcc")
				min_pcc = eval_dict['pcc'][-1]
			epoch -= 1
		return eval_dict

def start_model(model, epoch,train_data_path,train_label_path):
	t = Train(train_data_path,train_label_path,batch_size=24,percentage_of_train=0.9,num_workers=8)
	result = None
	Max_fitness = 0
	result = t.train_model(epoch=epoch, net=model)
	Max_fitness = max(result['pcc'])
	print(result)
	print(Max_fitness)
	return Max_fitness

def average_pcc(improvements):
    return sum(improvements) / len(improvements)

# an example
if __name__ == "__main__":
	pass
 
	