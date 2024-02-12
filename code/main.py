from typing import List
import internals_2d
from symengine import symbols, function_symbol

class DataGenerator: 
    """
    Generates data pairs of (f,u) for second order elliptic equations of the form -div(A nabla u) + b nabla u + cu = f on (0,1)^2 where 
    A is a 2x2 matrix, b is a vector, c is a scalar function and the boundary condition is either "dirichlet" or "neumann".
    """

    def __init__(self, data_points: int = 10000, dimension: int = 2, grid_size: int =85, truncation_order = 20, elliptic_matrix = None, 
                 b_term = None, c_term = None, boundary_condition: str = "dirichlet", x_save_path = None, y_save_path = None) -> None:
        
        self.data_points = data_points
        self.dimension = dimension
        self.grid_size = grid_size
        self.truncation_order = truncation_order
        self.elliptic_matrix = elliptic_matrix
        self.b_term = b_term
        self.c_term = c_term
        self.boundary_condition = boundary_condition
        self.x_save_path = x_save_path
        self.y_save_path = y_save_path
    
        if not self.x_save_path:
            self.x_save_path = f"{self.boundary_condition}_{self.dimension}d_{self.data_points}_points_res_{self.grid_size}by{self.grid_size}_input.pt"

        if not self.y_save_path:
            self.y_save_path = f"{self.boundary_condition}_{self.dimension}d_{self.data_points}_points_res_{self.grid_size}by{self.grid_size}_outout.pt"
    
        if not self.elliptic_matrix:
            x,y = symbols("x y")
            a = function_symbol("a", x,y)
            b = function_symbol("b", x,y)
            c = function_symbol("c", x,y)
            d = function_symbol("d", x,y)
            a = 1 + 0*x + 0*y
            b = 0 + 0*x + 0*y
            c = 0 + 0*x + 0*y
            d = 1 + 0*x + 0*y

            self.elliptic_matrix = [a,b,c,d]

        if not self.b_term:
            x,y = symbols("x y")
            self.b_term = [x + 0*y,0 + 0*x +y]
        
        if not self.c_term:
            x,y = symbols("x y")
            self.c_term = 1 + 0*x + 0*y
        
    def generate(self) -> None:
        internals_2d.save_data_in_parallel(
            self.data_points, 
            self.dimension,
            self.grid_size,
            self.truncation_order,
            self.elliptic_matrix,
            self.b_term,
            self.c_term,
            self.boundary_condition,
            self.x_save_path, 
            self.y_save_path
        )
    
    def print_equation(self):
        print(f'elliptic matrix A is given by: {self.elliptic_matrix}')
        print(f'vector b is given by: {self.b_term}')
        print(f'first order coefficient c is given by: {self.c_term}')

if __name__ == '__main__':
    DataGenerator().generate()
    DataGenerator().print_equation()