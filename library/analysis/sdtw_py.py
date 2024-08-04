"""
Contains a simple implementation of the soft-dtw algorithm in python.
Note that this functions are not created to by used as a loss functions so they not keep track of gradient. They simply compute the sdtw distance between two inputs.
"""

import numpy as np
from numba import jit

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def soft_dtw_python(x : np.array, y : np.array, gamma : float = 1, p_distance_grade : int = 2) :
    """
    Compute the soft-dtw between x and y. Note that this function compute only the sdtw value but not keep track of gradient.
    """
    distance_matrix = compute_distance_matrix(x, y, gamma, p_distance_grade)

    return distance_matrix[-1, -1]

@jit(nopython = True, parallel = False)
def compute_distance_matrix(x : np.array, y : np.array, gamma : float = 1, p_distance_grade : int = 2) -> np.array :
    """
    Compute the distance matrix between x and y array.
    The value of the soft-dtw are compute in diagonals, starting from the upper left corner and proceding toward the lower right corner.
    The firs diagonal has only 1 element, i.e. the element in posizion (0, 0)
    The second diagonal has 2 elements, i.e. the elements in position (0, 1) and (1, 0)
    """
    
    # Check input parameters
    if len(x) != len(y) : raise ValueError("x and y must have the same length. Current x length = {}, current y length = {}".format(len(x), len(y)))
    if gamma <= 0 : raise ValueError("gamma must be non negative. Current value is {}.".format(gamma))

    # Create distance matrix
    distance_matrix = np.zeros((len(x), len(y)))
    
    # Evaluate the number of diagonals
    n_diagonals = len(x) + len(y) - 1

    # Compute the element in position (0, 0)
    distance_matrix[0, 0] = lp_distance(x[0], y[0], p_distance_grade)
    
    # Variables used during computation
    n_elements_diagonal = 1
    computation_in_the_upper_left_section = True # Used to track if we are in the upper or lower section of the matrix
    shift = 0 # Shift of the indices to use in the lower right section of the matrix
    
    # Compute distance matrix diagonals by diagonals
    for n_1 in range(1, n_diagonals) :
        if n_1 <= n_diagonals / 2 :
            n_elements_diagonal += 1
            computation_in_the_upper_left_section = True
        else :
            n_elements_diagonal -= 1
            computation_in_the_upper_left_section = False

        if computation_in_the_upper_left_section :
            shift = 0
        else :
            shift = 1
        
        # Iterate over elements of the diagonal
        # idx_1 is used for the row of the distance matrix and as index of the x array
        # idx_2 is used for the column of the distance matrix and as index of the y array
        for n_2 in range(n_elements_diagonal) :
            idx_1 = n_2 + shift
            idx_2 = n_elements_diagonal - 1 - n_2 + shift

            # Get the value on the left the one I want to compute
            if idx_1 - 1 < 0 :
                left_value = np.finfo(np.float16).max
            else :
                left_value = distance_matrix[idx_1 - 1, idx_2]

            # Get the value above the one I want to compute
            if idx_2 - 1 < 0 :
                upper_value = np.finfo(np.float16).max
            else :
                upper_value = distance_matrix[idx_1, idx_2 - 1]

            # Get the value above and on the left (i.e. in diagonal) respect the one I want to compute
            if idx_1 - 1 < 0 or idx_2 - 1 < 0 :
                upper_left_value = np.finfo(np.float16).max
            else :
                upper_left_value = distance_matrix[idx_1 - 1, idx_2 - 1]
            
            # Compute the distance between samples of the two array
            samples_distance_value = lp_distance(x[idx_1], y[idx_2], p_distance_grade)
            softmin_value = compute_softmin([left_value, upper_left_value, upper_value], gamma)

            # Save the value in the distance_matrix 
            distance_matrix[idx_1, idx_2] = samples_distance_value + softmin_value

    return distance_matrix

@jit(nopython = True, parallel = False)
def compute_softmin(values_list, gamma : float = 1)  -> float :
    """
    Compute the softmin of the values in values_list
    """
    softmin_value = 0
    for i in range(len(values_list)) :
        softmin_value += np.exp(- values_list[i] / gamma)

    if softmin_value == 0 :
        # Sometimes due to numerical precision the exponential are all round to zero.
        # To avoid error during the computation of the log in this case the min value is returned
        return np.min(values_list)
    else :
        return - gamma * np.log(softmin_value)

@jit(nopython = True, parallel = False)
def lp_distance(a : float, b : float, p_distance_grade : int = 2) -> float:
    """
    Compute the lp norm between a and b.

    @param a : (float)
    @param b : (float)
    @param p_distance_grade : (int) parameter p of the lp distance

    @return : (float) the lp distance between a and b
    """

    return (abs(a - b) ** p_distance_grade) ** (1 / p_distance_grade)
