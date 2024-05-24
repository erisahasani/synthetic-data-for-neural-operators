from symengine import symbols, function_symbol, DenseMatrix, sin, cos, exp
import numpy as np
import torch
import multiprocessing
from sympy import  lambdify



def generate_sine_2d(K):
    x,y = symbols("x y")
    u = function_symbol("u",x,y)
    u = 0*x +0
    for i in range(1,K):
        for j in range(1,K):
            u = u + np.random.normal(0,1/(i+j))*sin(np.pi*i*x)*sin(np.pi*j*y)/np.sqrt(((np.pi*i)**2+(np.pi*j)**2))
    return u

def generate_cosine_2d(K):
    x,y = symbols("x y")
    u = function_symbol("u",x,y)
    u = 0*x +0
    for i in range(1,K):
        for j in range(1,K):
            u = u + np.random.normal(0,1/(i+j))*cos(np.pi*i*x)*cos(np.pi*j*y)/np.sqrt(((np.pi*i)**2+(np.pi*j)**2))
    return u

def get_function(boundary_condition, K):
    u = None
    if boundary_condition.lower() =="dirichlet":
        u = generate_sine_2d(K)
    elif boundary_condition.lower() =="neumann":
        u = generate_cosine_2d(K)
    else:
        raise TypeError("Only Dirichlet or Neumann boundary conditions.")
    return u

def divergence_A_nablau(u,elliptic_matrix):
    x,y = symbols("x y")
    divA_nablau = function_symbol("divA_nablau", x,y)

    a = elliptic_matrix[0]
    b = elliptic_matrix[1]
    c = elliptic_matrix[2]
    d = elliptic_matrix[3]

    a_x = function_symbol("a_x", x,y)
    b_x = function_symbol("b_x", x,y)
    c_y = function_symbol("c_y", x,y)
    d_y = function_symbol("d_y", x,y)   

    a_x = a.diff(x)
    b_x = b.diff(x)
    c_y = c.diff(y)
    d_y = d.diff(y)

    divA_nablau = (a_x + c_y)*u.diff(x) + (b_x+d_y)*u.diff(y) + a*u.diff(x).diff(x) + d*u.diff(y).diff(y) + (b+c)*u.diff(x).diff(y)
    return (-1)*divA_nablau 


def generate_data(dimension, grid_size, truncation_order, elliptic_matrix, nonlinear_term, boundary_condition, idx):
    
    x, y = symbols("x,y")
    f = function_symbol("f",x,y)
    w = symbols("w")
    K = np.random.randint(2,truncation_order+1)
    u = get_function(boundary_condition, K)
    c_u = nonlinear_term.subs({w: u})

    f = divergence_A_nablau(u,elliptic_matrix) + c_u

    cor = np.linspace(0, 1, grid_size)
    X, Y = np.meshgrid(cor, cor)

    X_flat = X.ravel()
    Y_flat = Y.ravel()

    f_func = lambdify((x, y), f)
    u_func = lambdify((x, y), u)

    f_flat = f_func(X_flat, Y_flat)
    u_flat = u_func(X_flat, Y_flat)

    f_reshaped = np.array(f_flat).reshape(X.shape)
    u_reshaped = np.array(u_flat).reshape(X.shape)

    input_data = torch.tensor(f_reshaped)
    output_data = torch.tensor(u_reshaped)

    input_data = input_data[None,:]
    output_data = output_data[None,:]
    

    print("progress:", idx)
    

    return input_data, output_data, idx


def generate_and_enqueue_data(x_data: torch.Tensor, y_data: torch.Tensor, dimension: int, grid_size: int, 
                              truncation_order: int, elliptic_matrix, nonlinear_term, boundary_condition, idx: int):
    
    result1, result2, idx = generate_data(dimension, grid_size, truncation_order, elliptic_matrix, nonlinear_term, boundary_condition, idx)
    x_data[idx,:,:,:] = result1
    y_data[idx,:,:] = result2

def save_data_in_parallel(length: int, dimension: int, grid_size: int, truncation_order: int, elliptic_matrix,
                           nonlinear_term, boundary_condition, x_path, y_path):
    multiprocessing.set_start_method('spawn')
    grid_size = grid_size
    length = length
    input_functions = 1
    x_data = torch.zeros((length,input_functions,grid_size, grid_size))
    y_data = torch.zeros((length,grid_size,grid_size))
    x_data.share_memory_()
    y_data.share_memory_()

    args = [(x_data, y_data, dimension, grid_size, truncation_order, elliptic_matrix,nonlinear_term,
             boundary_condition, idx) for idx in range(length)]
    # torch.set_num_threads(1)
    with multiprocessing.Pool() as pool:
        pool.starmap(generate_and_enqueue_data, args)

    torch.save(x_data, x_path)
    torch.save(y_data, y_path)




if __name__ == '__main__':
    x,y = symbols("x y")
    a = function_symbol("a", x,y)
    b = function_symbol("b", x,y)
    c = function_symbol("c", x,y)
    d = function_symbol("d", x,y)

    w = symbols("w")
    c_u = function_symbol("c_u",w)

    a = 1 + 0*x + 0*y
    b = 0 + 0*x + 0*y
    c = 0 + 0*x + 0*y
    d = 1 + 0*x + 0*y
    c_u = w**2
    length = 200

    save_data_in_parallel(
        length = length,
        dimension= 2,
        grid_size= 85,
        truncation_order = 20,
        elliptic_matrix= [a,b,c,d],
        nonlinear_term= c_u,
        boundary_condition= "dirichlet",
        x_path= f'cosine_1_to_20_positive_div_particularA_norm_{length}_x_test.pt',
        y_path= f'cosine_1_to_20_positive_div_particularA_norm_{length}_y_test.pt'
    )
    