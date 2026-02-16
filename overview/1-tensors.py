import torch
import numpy as np

# --- Defining Tensors ---
# defining a matrix of data
data = [[1, 2], [3, 4]]

# creating tensor using data specified above
x_data = torch.tensor(data)

# creating numpy array using data specified above
np_array = np.array(data)

# creating tensor using numpy array specified above
x_np = torch.from_numpy(np_array)

# print(f"x_data: \n {x_data} \n")
# print(f"x_np: \n {x_np} \n")


# --- Defining Tensors With Tensors ---
# when a new tensor is created using a pre-existing tensor, by default,
# the new tensor will retain the properties (shape & datatype) of the argument tensor,
# this can be explicity overridden

# creates x_ones with same properties as x_data
x_ones = torch.ones_like(x_data)

# creates x_rand with datatype of x_data explicitly overridden
x_rand =  torch.rand_like(x_data, dtype=torch.float) 

# print(f"x_ones: \n {x_ones} \n")
# print(f"x_rand: \n {x_rand} \n")


# --- Defining Tensor Dimentions ---
# The dimentions of tensors can be defined using tupples, 
# think of it like defining the dimentions of a matrix,
# where the first number is rows & the second is columns.
# For example the tensors defined below will all have 3 rows & 5 columns 

shape = (3, 5,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

# print(f"rand_tensor: \n {rand_tensor} \n")
# print(f"ones_tensor: \n {ones_tensor} \n")
# print(f"zeros_tensor: \n {zeros_tensor} \n")


# --- Tensor Attributes ---
# Tensor attribues are feilds on each tensor instance that specify information about the tensor.
# Such as its, shape (diementions), data type, and which device its stored on (ie, cpu or gpu)

tensor = torch.rand(3, 4)

# print(f"tensor shape: {tensor.shape} \n")
# print(f"tensor data type: {tensor.dtype} \n")
# print(f"tensor device: {tensor.device} \n")


# --- Tensor Operations ---
# a brief summary of each some tensor operators is stated below, 
# for more indepth information refer to the torch documentation.
# Tensor operations will be performed much faster on a GPU than a CPU, especially a CUDA enabled GPU.
# The device a tensor is stored on can be switched if needed, but it can only be switched to a present device.

# checks if cuda is available
cuda_available = torch.cuda.is_available()

if(cuda_available):
    # stores tensor on cude instead of initial device
    tensor = tensor.to('cuda')

# print(f"cuda available: {cuda_available}\n")
# print(f"tensor device: {tensor.device}\n")


# --- Tensor Indexing & Slicing ---
# You can access a specific item in a tensor by spcifying its dimention corrdinates in the martrix, with corrdinates being 0 indexed. 
# This is done the same way you would an array (ie, array[1]), exept you need to specify a number for each dimention (ie, tensor[1, 3]).
# If you want to access an entire row or column of a tensor, you can use the ':' character in place row or column number, 
# for example "tensor[:,1]" will select the entire second column of the tensor.

# creates 4x4 tensor of all 1s
tensor = torch.ones(4, 4)

# assigns even item in column 2 to 0
tensor[:,1] = 0

# print(f"tensor: \n {tensor} \n")


# --- Joining Tensors ---
# Tensors can be joined together using various operations, such as torch.cat.
# "torch.cat" concatenates tensors together along a specified dimention.
# It's takes an array consisting of all the tensors you wish to concatenate, 
# and the dimention number on which to join the tensors which is 0 indexed

# concat's 3 tensors together on the row dimention 
t1 = torch.cat([tensor, tensor, tensor], dim=0) 

# print(f"t1: \n {t1} \n")


# --- Multiplying Tensors ---
# As tensors are matricies with special properties, multiplying tensors acts in a similar way,
# for element wise product both tersors must have same dimentions, 
# for matrix multiplication the number of columns in tensor A must be equal to the number of rows in tensor B.
# Note, tensor.T returns the transpose of the tensor (makes the columns the rows & vice versa), 
# this is done for tensor B in the martix multiplication example because it makes sure the tensor A's columns 
# and B's rows meet the conditions required for the operation

# calculates element wise product of 2 tensors
tensor_mult1 = tensor.mul(tensor)

# alternative syntax, does same as above
tensor_mult2 = tensor * tensor

# calculates matrix multiplication of 2 tensors
tensor_matmult1 = tensor.matmul(tensor.T)

# alternate syntax, does same as above
tensor_matmult2 = tensor @ tensor.T

# print(f"tensor_mult1: \n {tensor_mult1} \n")
# print(f"tensor_mult2: \n {tensor_mult2} \n")
# print(f"tensor_matmult1: \n {tensor_matmult1} \n")
# print(f"tensor_matmult2: \n {tensor_matmult2} \n")


# --- In Place Operations ---
# In place operations are operations that end with the '_' suffix.
# In place operations directly modify existing tensors, 
# unlike the operations above they directly edit tensor they are called upon.
# Although they can save memory there use is discouraged as they will remove a tensors imidiate history

# print(f"tensor:\n{tensor}\n")

# Adds 5 to each element in the tensor
tensor.add_(5);

# print(f"tensor:\n{tensor}\n")


# --- Tensors & Numpy ---
# Tensors stored on the CPU and any numpy array created with that tensor or vice versa (a tensor stored on cpu created with a numpy array),
# share their underlying memory location, meaning any change in one will change the other as they are referenceing the same thing

# np array created w/ tensor
t = torch.ones(5) # creates a tensor of all ones
n = t.numpy() # creates numpy array using tensor above 

# print(f"(t->n)t: \n{t}\n")
# print(f"(t->n)n: \n{n}\n")

t.add_(1) # alters tensor, change reflected in numpy array

# print(f"(t->n)t: \n{t}\n")
# print(f"(t->n)n: \n{n}\n")

n = np.ones(5)
t = torch.from_numpy(n)

# print(f"(n->t)t: \n{t}\n")
# print(f"(n->t)n: \n{n}\n")

np.add(n, 1, out=n)

# print(f"(n->t)t: \n{t}\n")
# print(f"(n->t)n: \n{n}\n")

