# MDNN
The project structure is as follows:

- data
  - Datasets to be used for training  
- data_process_another
  -  Scripts for dataset processing  

You can start MDNN by running `python Start_MDNN.py`.

To train models with different network architectures, you can modify the values of the `Neural_nodes`, `control`, and `Epoch` fields in the `Start_MDNN.py` file.

After execution, a folder named `matrixs` will be generated, where the trained model weights, fitness values, and network architectures will be saved.
