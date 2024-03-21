# Diffeomorphism-Neural-Operator
Diffeomorphism-Neural-Operator is a neural operator framework for solving PDEs with various domain and also parameters. It can learn an opertor on generic domain wich is a series domains witch is  diffeomorphically mapped from various physics domains expressed by the same PDE.
Diffeomorphism-Neural-Operator is strong generalization ability to size and shape of domains, and resolution invariant

For more information, please refer to the following paper.

[Diffeomorphism Neural Operator for various domains and parameters of partial differential equations](https://arxiv.org/abs/2402.12475) arXiv preprint arXiv:2402.12475



## Data download
[Darcy flow datasets](https://drive.google.com/drive/folders/1z8s25cKcF6nprngf8lcTXp5F1ECp7h6W?usp=drive_link)

[Pipe flow datasets](https://drive.google.com/drive/folders/1vtgPTYd83bQq-shw4fQizHURF7uJxcMF?usp=drive_link)

[Airfoil flow datasets](https://drive.google.com/drive/folders/1pmYZ1B_c1zVkOmeoksf6kTSKOGMSf2l1?usp=drive_link)

## Data Generation

If you want to try generating data yourself, we provide programs for generating data, including diffeomorphism methods and PDE solving processes. Using a pentagon with Darcy flow as an example, we demonstrate how to generate data, which mainly involves the following steps:

1. Generate the geometric domain and mesh it. Here, we use the **obj** format for the geometric domain, which can be generated using any meshing tool. Some examples are provided in the **part_obj** folder for testing purposes. The various domains of pentagon is genereted by a **MATLAB** process **part_generate.m**

2. Map the physical geometric domain diffeomorphically to a generic domain. Use **create_map.py** to map the mesh of the obj to a **unit rectangle domain**, and sample regularly in the rectangular domain with the solution of 128. Store the geometric coordinates of the sampling points corresponding to the physical domain in the **data** folder, namely **x_data.csv** and **y_data.csv**.

3. Generate physical field data. Based on the geometric domain and parameters, generate input and output fields and sample the input and output fields at the acquired sampling points. This yields the parameter field **C** and the output field **U**, which are stored in the data folder. The PDE solved by MATLAB process **data_generate.m**

Note: The points in **x_data.csv** and **y_data.csv** correspond one-to-one with the values in **C.csv** and **U.csv**.

If you want to get the code of Pipe flow (Generated by **Phiflow**) and Airfoils flow (Generated by **OpenFoam**), please email us. 

## Train

Here are the training and testing programs for Darcy flow. Once the dataset is prepared, you can run train.py using the following command: 
 ```bash
    python train.py
 ```

## Test

After training, the model will be stored in the `model` directory. You can use the provided testing program `test.py` to evaluate the trained model.
 
 ```bash
    python test.py
 ```

You can test the model with datasets of different resolutions, sizes, and shapes. 
The results will be stored in files named `result_train_dataset`, where `'train_dataset'` denotes the name of the training dataset.

## Experiment
Experiments on two statics scenarios (Darcy flow and mechanics) and two dynamic scenarios (pipe flow and airfoil flow) were carried out
### Darcy flow
The DNO of Darcy flow was trained on a pentagon domain and then generalized validation was conducted on varying sizes (5-30) and shapes (pentagon, hexagon, octagon).
#### Generalization on  domains with different size

<img src="experiment_graph/Darcy_flow_size.png" alt="Image" width="520" height="300">

#### Generalization on  domains with different shape

<img src="experiment_graph/Darcy_flow_shape.png" alt="Image" width="520" height="300">

### Pipe flow
The neural operator for pipe flow was trained on domains containing 2-4 baffles and then generalized on domains with 5 baffles.

<img src="experiment_graph/Pipe_flow.gif" alt="Image" width="500" height="300">

### Airfoil flow
The cases of airfoil flow were tested on airfoil of different shapes and then further generalized on larger domains containing different airfoil shapes.

<img src="experiment_graph/Airfoils_flow.gif" alt="Image" width="500" height="300">

## Acknowledgement 
Our framework and code reference the following resources：

[FNO](https://github.com/neuraloperator/neuraloperator)

[The Numerical Tour web site](http://www.numerical-tours.com)

[PhiFlow](https://tum-pbs.github.io/PhiFlow)





