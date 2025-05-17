import random
from judge_perfor import Performance_Estimation_Strategy
from ma_config import _CONV_Space, _POOL_Space
from my_utils import creat_floder_for_train,creat_floder_for_matrix,save_flag_file,check_file_existence,format_seconds
import os
import torch.nn
import time
from datetime import datetime
import sys

global_train_data_path = None
global_train_label_path = None

pop_size = 4
num_generations = 2
Cros_prob = 0.5
Muta_prob = 0.5	
Local_search_prob = 0.5
Local_search_itr = 2

Neural_nodes = 4
control =1

Epoch = 200

in_channel = 16
out_channel = 16

_CONV_STRING = {'a', 'b', 'c', 'd','m'}
_POOL_STRING = {'e', 'f', 'g', 'h','i','z'}


# init_population
def init_population(pop_size, Neural_nodes, Con_Search_space, Pool_Search_space):
	population = []
	S_flow = int((3 + Neural_nodes) * Neural_nodes / 2)
	s = 0
	while s < pop_size:
		inits = []
		for i in range(2):
			Temp_flows = []
			for i in range(Neural_nodes):
				flows = get_conv_encode_flows(i, Conv_Space=Con_Search_space)
				Temp_flows.extend(flows)
			# if s < S_flow:
			new_flows = check_reassign_flow(Temp_flows, Neural_nodes, Con_Search_space)
			# print("before ")
			# temp = 0
			# for k in range(Neural_nodes):
			# 	if k==0:
			# 		temp = temp + k + 1
			# 		continue
			# 	if new_flows[temp]!='FA':
			# 		raise KeyError
			# 	temp = temp + k + 1

			# true_flows = get_pool_encode_flows(new_flows, Neural_nodes, Pool_Space=Pool_Search_space)子
			true_flows = get_pool_encode_flows(new_flows, Neural_nodes, Pool_Space=Pool_Search_space)

			# print("after ")
			# temp = 0
			# for k in range(Neural_nodes):
			# 	if k==0:
			# 		temp = temp + k + 1
			# 		continue
			# 	if new_flows[temp]!='FA':
			# 		raise KeyError
			# 	temp = temp + k + 1

			inits.append(true_flows)
		population.append(inits)
		s += 1
	return population

def get_pool_encode_flows(flows, Neural_nodes, Pool_Space):
	temp = 0
	for i in range(Neural_nodes):
		_FA = []
		for j in range(temp, temp + i + 1):
			if flows[j] == 'FA':
				_FA.append(j)
		# In the random selection operator, the first neural node is skipped during the elimination of positions with a relative value of zero, as it needs to maintain the connection with the input.
		if i!=0:
			_FA = list(filter(lambda x: x != temp, _FA))
		if len(_FA) != 0:
			flows[random.choice(_FA)] = Pool_Space.random_select()[0]
		temp = temp + i + 1
	return flows

def get_conv_encode_flows(node_i, Conv_Space):
	flows_node = []
	if node_i==0:
		conv = Conv_Space.random_select_NoFA()
		flows_node.append(conv[0])
		return flows_node
	# Half of the positions are selected as information flow. For neural nodes starting from the second one, only preceding nodes can be chosen as inputs. Since 0 denotes the initial input, the selection starts from 1.
	rn = random.sample(range(1, node_i+1), (node_i+1) // 2)
	# Each neural node is assigned an additional position, so that node i has i + 1 positions preceding it.
	for i in range(node_i + 1):
		if i in rn:
			conv = Conv_Space.random_select()
			flows_node.append(conv[0])
		else:
			flows_node.append('FA')
	return flows_node

# make sure one node has one flow
def check_reassign_flow(flows, nodes, Con_Search_space):
	temp = 0
	for i in range(nodes):
		_Fa = []
		for j in range(temp, temp + i + 1):
			if (flows[j] == 'FA'):
				_Fa.append(j)
		if len(_Fa) == i + 1:
			chan_num = random.randrange(temp+1, temp + i + 1)
			flows[chan_num] = Con_Search_space.random_select_NoFA()[0]
		temp = temp + i + 1

	return flows


def calculate_fitness(population,Neural_nodes,_conv,_pool):
	fitness_values = []
	fitness_models = []
	fietness_matrix_nodes = []
	for i in range(len(population)):
		fitness_value,matrix_nodes,model = Performance_Estimation_Strategy(population[i],Neural_nodes,Epoch,_conv,_pool,global_train_data_path,global_train_label_path,control)
		fitness_values.append(fitness_value)
		fietness_matrix_nodes.append(matrix_nodes)
		fitness_models.append(model)
	return fitness_values,fitness_models,fietness_matrix_nodes

def separate_c_p(OF):
	OF_C = []
	OF_P = []
	for i in range(len(OF)):
		if OF[i] == 'FA':
			OF_C.append('FA')
			OF_P.append('FA')
			continue
		if OF[i] in _CONV_STRING:
			OF_C.append(OF[i])
			OF_P.append('FA')
		elif OF[i] in _POOL_STRING:
			OF_P.append(OF[i])
			OF_C.append('FA')
	return OF_C, OF_P

def count_fa(OF):
	OF_FA_INDEX = [index for index, value in enumerate(OF) if value == 'FA']
	return OF_FA_INDEX

# crossover
def crossover(parent1, parent2, Cros_prob):
	temp = 0
	for i in range(Neural_nodes):
		rounded_value = Cros_prob
		while rounded_value == Cros_prob:
			# Generate a random number between 0 and 1.
			random_value = random.uniform(0, 1)
			# Round the random number to one decimal place.
			rounded_value = round(random_value, 1)
		# cross if exceed Cros_prob
		if rounded_value > Cros_prob:
			for j in range(temp, temp + i + 1):
				parent1[j], parent2[j] = parent2[j], parent1[j]
		temp = temp + i + 1

	return parent1, parent2

# mutation
def mutation(OF, Muta_prob, _operator, Neural_nodes, FA_INDEX):
	temp = 0
	for i in range(Neural_nodes):
		for j in range(temp, temp + i + 1):
			if OF[j] == 'FA':
				continue
			rounded_value = Muta_prob
			while rounded_value == Muta_prob:
				# Generate a random number between 0 and 1.
				random_value = random.uniform(0, 1)
				# Round the random number to one decimal place.
				rounded_value = round(random_value, 1)
			# change flow info if exceed Cros_prob otherwise change flow location
			if rounded_value > Muta_prob:
				OF[j] = _operator.random_select_NoFA()[0]
			else:
				OF = move_flow(OF, temp, i, FA_INDEX)
		temp = temp + i + 1
	return OF

def move_flow(OF, temp, i, FACOUNT_INDEX):
	FA_INDEX = []
	FLOW_INDEX = []
	new_OF = OF
	for j in range(temp, temp + i + 1):
		if OF[j] == 'FA':
			FA_INDEX.append(j)
		else:
			FLOW_INDEX.append(j)
	if len(FA_INDEX) == 0:
		return new_OF
	while len(FA_INDEX) != 0 and len(FLOW_INDEX) != 0:
		random_FA_Index = random.choice(FA_INDEX)
		random_FA_value_index = FA_INDEX.index(random_FA_Index)

		random_Flow_Index = random.choice(FLOW_INDEX)
		random_Flow_value_index = FLOW_INDEX.index(random_Flow_Index)
		# Swap.
		if random_FA_Index in FACOUNT_INDEX:
			OF[random_Flow_Index], OF[random_FA_Index] = OF[random_FA_Index], OF[random_Flow_Index]

		FA_INDEX.pop(random_FA_value_index)
		FLOW_INDEX.pop(random_Flow_value_index)
	return new_OF

def merge_c_p(OF_C, OF_P):
	new_of = []
	for i in range(len(OF_C)):
		if OF_C[i]=='FA' and OF_P[i]=='FA':
			new_of.append('FA')
		elif OF_C[i] != 'FA' and OF_P[i] != 'FA':
			try:
				print('Position overlap.')
				print("The overlapping character is.")
				print(OF_C)
				print(OF_P)
				print("The overlapping position is.")
				print(i)
				raise ValueError("A position overlap error occurred.")
			except Exception as e:
				print(e)
		elif OF_C[i]!='FA' or OF_P[i]!='FA':
			new_of.append(OF_P[i] if OF_P[i]!='FA' else OF_C[i])
	return new_of

def merge_n_r(individual,flag,OF):

	if flag in individual:
		origin_index = 1 if individual[0] == flag else 0
	else:
		try:
			raise ValueError("Parameter not found in list")
		except Exception as e:
			print(e)
	return [OF,individual[origin_index]] if origin_index == 1 else [individual[origin_index],OF]


def split_columns(data):
	"""
	Split a 2D list into two lists by columns.

	Parameters:
	data (list of lists): The input 2D list.

	Returns:
	tuple: A tuple containing two lists, where the first list consists of the first column elements from all rows, and the second list consists of the second column elements from all rows.
	"""
	col1 = [row[0] for row in data]
	col2 = [row[1] for row in data]
	col3 = [row[2] for row in data]
	col4 = [row[3] for row in data]
	return col1, col2,col3,col4

def roulette_wheel_selection(population, fitness_values, num_selections):
	"""
	Roulette Wheel Selection Algorithm

	Parameters:

	population (list): The list of individuals in the population.

	fitness_values (list): The list of fitness values corresponding to each individual.

	num_selections (int): The number of individuals to select.

	Returns:

	selected_individuals (list): The list of selected individuals
	"""


	total_fitness = sum(fitness_values)


	selection_probs = [fitness / total_fitness for fitness in fitness_values]


	cumulative_probs = []
	cumulative_sum = 0
	for prob in selection_probs:
		cumulative_sum += prob
		cumulative_probs.append(cumulative_sum)

	selected_individuals = []
	for _ in range(num_selections):
		rand = random.random()  
		for i, cumulative_prob in enumerate(cumulative_probs):
			if rand <= cumulative_prob:
				selected_individuals.append(population[i])
				break

	return selected_individuals

def elitism_selection(population, fitness_scores,all_models,all_matrixs_nodes,elite_size):
	"""
	Elitism Selection Method

	Parameters:

	population (list): The current population containing all individuals.

	fitness_scores (list): The list of fitness scores corresponding to the individuals in the population.

	elite_size (int): The number of elite individuals to retain.

	Returns:

	list: The individuals preserved after elitism selection.
	"""
	sorted_population = sorted(zip(fitness_scores,population,all_models,all_matrixs_nodes), reverse=True)

	elites = sorted_population[:elite_size]

	return split_columns(elites)

def tournament_selection(population, fitness_scores,all_models,all_matrixs_nodes,tournament_size,n):
	"""
	Tournament Selection Method

	Parameters:

	population (list): The current population containing all individuals.

	fitness_scores (list): The list of fitness scores corresponding to the individuals in the population.

	tournament_size (int): The number of individuals participating in each tournament.

	Returns:

	object: The individual selected from the tournament.
	"""
	winner = []
	for i in range(n):
		tournament_indices = random.sample(range(len(population)), tournament_size)
		tournament = [(fitness_scores[i], population[i],all_models[i],all_matrixs_nodes[i]) for i in tournament_indices]
		winner.append(max(tournament, key=lambda x: x[0]))
	return split_columns(winner)

# globle_search
def globle_search(population, all_fitness, _conv,Muta_prob,Neural_nodes):
	OFs = roulette_wheel_selection(population, all_fitness, 2)
	Init_cell = []
	Normal_cell = []
	Reduce_cell = []
	for i in range(len(OFs[0])):
		OF_1, OF_2 = crossover(OFs[0][i], OFs[1][i], Cros_prob)
		OF_1_C, OF_1_P = separate_c_p(OF_1)
		OF_2_C, OF_2_P = separate_c_p(OF_2)
		# Obtain the indices of FA in OF_P to ensure that subsequent mutations involving the movement of convolutional operators do not overlap with the pooling operators in OF_P.
		FAP1_INDEX = count_fa(OF_1_P)
		FAP2_INDEX = count_fa(OF_2_P)

		NEW_OF_1_C = mutation(OF_1_C, Muta_prob, _conv, Neural_nodes, FAP1_INDEX)
		NEW_OF_2_C = mutation(OF_2_C, Muta_prob, _conv, Neural_nodes, FAP2_INDEX)
		OF_1 = merge_c_p(NEW_OF_1_C, OF_1_P)
		OF_2 = merge_c_p(NEW_OF_2_C, OF_2_P)
		if i ==0:
			Normal_cell.append(OF_1)
			Normal_cell.append(OF_2)
		else:
			Reduce_cell.append(OF_1)
			Reduce_cell.append(OF_2)
	for i in range(len(Normal_cell)):
		Init_cell.append([Normal_cell[i],Reduce_cell[i]])
	return Init_cell

# local_search
def local_search(individual, itro,Local_search_prob,_pool,_conv,Neural_nodes,epoch):
	OF_P_best = individual
	fite_ness_best = None
	model_best = None
	matrix_node_best = None
	fite_ness_best,matrix_node_best,model_best = Performance_Estimation_Strategy(OF_P_best,Neural_nodes,epoch,_conv,_pool,global_train_data_path,global_train_label_path,control)
	for indiv in individual:
		OF_C,OF_P = separate_c_p(indiv)
		FAC_INDEX = count_fa(OF_C)
		
		for i in range(itro):
			OF_P_temp = mutation(OF_P, Local_search_prob, _pool, Neural_nodes, FAC_INDEX)
			OF_P_temp = merge_c_p(OF_C, OF_P_temp)
			new_indivi = merge_n_r(individual,indiv,OF_P_temp)
			fite_ness_temp,matrix_nodes_temp,model_temp =Performance_Estimation_Strategy(new_indivi,Neural_nodes,epoch,_conv,_pool,global_train_data_path,global_train_label_path,control)
			if fite_ness_temp >= fite_ness_best:
				OF_P_best = new_indivi
				fite_ness_best = fite_ness_temp
				model_best = model_temp
				matrix_node_best = matrix_nodes_temp

	return OF_P_best,fite_ness_best,model_best,matrix_node_best

def get_new_population(population,fitness,all_models,all_matrixs_nodes,size):
	new_population = []
	new_fitness = []
	new_models = []
	new_matrixs_nodes = []

	half_befor = int(size/2)

	els_fit,els_pop,els_model,els_matrixs_node = elitism_selection(population,fitness,all_models,all_matrixs_nodes,half_befor)
	tour_fit,tour_pop,tour_model,tour_matrixs_node = tournament_selection(population,fitness,all_models,all_matrixs_nodes,half_befor,half_befor)

	new_population.extend(els_pop)
	new_population.extend(tour_pop)
	

	new_fitness.extend(els_fit)
	new_fitness.extend(tour_fit)

	new_models.extend(els_model)
	new_models.extend(tour_model)

	new_matrixs_nodes.extend(els_matrixs_node)
	new_matrixs_nodes.extend(tour_matrixs_node)

	return new_population,new_fitness,new_models,new_matrixs_nodes

def get_save_population(population,fitness,all_models,all_matrixs_nodes,size):
	new_population = []
	new_fitness = []
	new_models = []
	new_matrixs_nodes = []


	els_fit,els_pop,els_model,els_matrixs_node = elitism_selection(population,fitness,all_models,all_matrixs_nodes,size)
	new_population.extend(els_pop)
	new_fitness.extend(els_fit)
	new_models.extend(els_model)
	new_matrixs_nodes.extend(els_matrixs_node)

	return new_population,new_fitness,new_models,new_matrixs_nodes


def memetic_algorithm(pop_size, Neural_nodes, num_generations,train_data_path = None,train_label_path = None):
	global global_train_data_path,global_train_label_path
	global_train_data_path = train_data_path
	global_train_label_path = train_label_path
 
	_conv = _CONV_Space(in_channel, out_channel)
	_pool = _POOL_Space()

	population = init_population(pop_size, Neural_nodes, _conv, _pool)
	new_population = population
	original_pop = population

	all_fitness,all_models,all_matrixs_nodes = calculate_fitness(new_population,Neural_nodes,_conv,_pool)

	new_fitness = all_fitness
	new_models = all_models
	new_matrixs_nodes = all_matrixs_nodes
	for generation in range(num_generations):
		print(f"generation is : {generation}")
		print("Now, start the global search.")
		individuals = globle_search(new_population, new_fitness, _conv,Muta_prob,Neural_nodes)
		print("Now, end the global search.")

		print("Now, start the local search.")
		for individual in individuals:
			OF_P_best,fitness_best,model_best,matrix_node_best = local_search(individual,Local_search_itr,Local_search_prob,_pool,_conv,Neural_nodes,Epoch)
			original_pop.append(OF_P_best)
			all_fitness.append(fitness_best)
			all_models.append(model_best)
			all_matrixs_nodes.append(matrix_node_best)
		print("Now, end the local search.")
		new_population,new_fitness,new_models,new_matrixs_nodes = get_new_population(original_pop,all_fitness,all_models,all_matrixs_nodes,pop_size)
	print("Now, end the MA,start save file\n")
	new_population,new_fitness,new_models,new_matrixs_nodes = get_save_population(original_pop,all_fitness,all_models,all_matrixs_nodes,pop_size)
 
	print("New fitness is ",new_fitness)
	folder_path = creat_floder_for_train("matrixs","matrixs_train")
	for index,individual in enumerate(new_population):
		file_save_path = creat_floder_for_matrix(folder_path,index+1)
		print(f'Now save {index} information')
		save_flag_file(new_matrixs_nodes[index],individual,new_fitness[index],file_save_path)
		os.makedirs(file_save_path, exist_ok=True)  # Create a folder if it does not exist.
		# Save the model's state dictionary.
		save_path = os.path.join(file_save_path, 'cell_decode_model.pth')
		torch.save(new_models[index].state_dict(), save_path)
		print(f"model has been save in {save_path}")
  
# start
if __name__ == "__main__":
	# for label_index in range(6):
	# 	print(f'now start label_index is ',label_index)
	# 	memetic_algorithm(pop_size, Neural_nodes, num_generations,label_index+1)
	train_data_paths = ['./data/tomato332/Indel_pca/332ind_indel_pc95.csv']
 
	label_file_paths = [['./data/tomato332/Indel_pca/tomato_phe.txt']]
	check_file_existence(train_data_paths,label_file_paths)
	log_path = "runtime_log_ne=1-c=1.txt"

	with open(log_path, "a") as f:
		# Write the current time as a delimiter.
		current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		f.write(f"\n===== Run started at {current_time} =====\n")

		for train_data_index,train_data_path in enumerate(train_data_paths):
			print('now train data is ',train_data_path)
   
			outer_start = time.perf_counter()
			f.write(f"=== Outer Loop {train_data_index+1} Start ===\n")
			f.write(f"now train data is {train_data_path}\n")
   
			for train_label_index,train_label_path in enumerate(label_file_paths[train_data_index]):
				print('now train label is ',train_label_path)
    
				f.write(f'now train label is {train_label_path}')
				inner_start = time.perf_counter()
				memetic_algorithm(pop_size,Neural_nodes,num_generations,train_data_path,train_label_path)
				inner_elapsed = time.perf_counter() - inner_start
				inner_elapsed = format_seconds(inner_elapsed)
    
				f.write(f"Inner Loop {train_label_index+1}: {inner_elapsed} seconds\n")

			outer_elapsed = time.perf_counter() - outer_start
			outer_elapsed = format_seconds(outer_elapsed)
			f.write(f"=== Outer Loop {train_data_index+1} End: {outer_elapsed} seconds ===\n\n")
	
