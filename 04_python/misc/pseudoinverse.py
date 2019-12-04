# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 11:15:17 2019

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def left_pseudo_inv(A):
    """
    Computes the pseudoinverse of a matrix.
    The columns of matrix A must be linearly independent!
    If this is the case (A.T @ A) is invertible.
    
    :param A:       matrix to invert
    :return:        pseudoinverse
    """
    return np.linalg.inv(A.T @ A) @ A.T


def right_pseudo_inv(A):
    """
    Computes the pseudoinverse of a matrix.
    The rows of matrix A must be linearly independent!
    If this is the case (A @ A.T) is invertible.
    
    :param A:       matrix to invert
    :return:        pseudoinverse
    """
    return A.T @ np.linalg.inv(A @ A.T)
    

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    A = np.asarray([[2, 3, 5], [5, 7, 7]])
    print("Matrix A:")
    print(A)
    print()
    
    print("Right pseudoinverse of A:")
    print(A @ right_pseudo_inv(A))
    print()
    
    B = A.T
    print("Matrix B:")
    print(B)
    print()
    
    print("Left pseudoinverse of B:")
    print(left_pseudo_inv(B) @ B)
    