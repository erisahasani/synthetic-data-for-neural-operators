from typing import List
import internals_2d
from symengine import symbols, function_symbol

class DataGenerator: 
    """
    Generates data pairs of (f,u) for elliptic nonlinear equations of the form -div(A nabla u) + c(u) = f on (0,1)^2 where 
    A is a 2x2 matrix and c(u) is some nonlinear term in u and boundary condition is either "dirichlet" or "neumann".
    """

    def __init__(self, data_points: int = 25, dimension: int = 2, grid_size: int =85, truncation_order = 20, elliptic_matrix = None, 
                 nonlinear_term = None, boundary_condition: str = "dirichlet", x_save_path = None, y_save_path = None) -> None:
        
        self.data_points = data_points
        self.dimension = dimension
        self.grid_size = grid_size
        self.truncation_order = truncation_order
        self.elliptic_matrix = elliptic_matrix
        self.nonlinear_term = nonlinear_term
        self.boundary_condition = boundary_condition
        self.x_save_path = x_save_path
        self.y_save_path = y_save_path
    
        if not self.x_save_path:
            self.x_save_path = f"synthetic_data_{self.data_points}_x.pt"

        if not self.y_save_path:
            self.y_save_path = f"synthetic_data_{self.data_points}_y.pt"
    
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

        if not self.nonlinear_term:
            w = symbols("w")
            c_u = function_symbol("c_u",w)
            c_u = w**2
            self.nonlinear_term = c_u
        
    def generate(self) -> None:
        internals_2d.save_data_in_parallel(
            self.data_points, 
            self.dimension,
            self.grid_size,
            self.truncation_order,
            self.elliptic_matrix,
            self.nonlinear_term,
            self.boundary_condition,
            self.x_save_path, 
            self.y_save_path
        )

if __name__ == '__main__':
    DataGenerator().generate()