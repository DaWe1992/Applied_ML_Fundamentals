# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 08:30:17 2018

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import math


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

def chi_merge(values, cnt_ivl):
    """
    Dicretizes the data given using chi merge.
    
    :param values:              data to be discretized
    :param cnt_ivl:             number of intervals
    :return:                    merged data
    """
    # 1. INITIALIZATION
    # -------------------------------------------------------------------------
    # sort input by value
    values.sort(key=lambda x: x[0])
    # construct one interval for each value
    data = [[value] for value in values]
    
    # 2. INTERVAL MERGING
    # -------------------------------------------------------------------------
    # determine the set of class values
    classes = list(set(value[1] for value in values))
    
    # while there are more intervals than specified...
    while(len(data) > cnt_ivl):
        K = len(data) - 1
        chi2_list = [0] * K
    
        for k in range(K):
            A = freq_mat([data[k], data[k + 1]], classes)
            E = exp_mat(A)
            
            # calculate chi^2 value
            for i in range(2):
                for j in range(len(classes)):
                    chi2_list[k] += math.pow((A[i][j] - E[i][j]), 2) / E[i][j]
        
        # merge intervalls with lowest chi^2 value
        data = merge(data, find_merge_index(chi2_list))
        
    return data
        

def merge(ivls, index):
    """
    Merges two intervals and returns the new list of intervals.
    
    :param ivls:                list of intervals
    :param index:               index of element that will be merged
                                with its right neighbor
    :return:                    new list of merged intervals
    """
    ivls_new = []
    
    i = 0
    while i < len(ivls):
        if i == index:
            ivls_new.append(ivls[i] + ivls[i + 1])
            i = i + 1
        else:
            ivls_new.append(ivls[i])
        i = i + 1
        
    return ivls_new


def find_merge_index(chi2_list):
    """
    Finds index of element to merge with its right neighbor.
    
    :param chi2_list:           list of chi^2 values
    :return:                    merge index
    """
    min_chi = min(chi2_list)
    for i in range(len(chi2_list)):
        if chi2_list[i] == min_chi:
            return i
        
        
def freq_mat(ivls, classes):
    """
    Calculates the frequencies of classes per interval.
    It returns a matrix of shape (2 x |cls|).
    The intervals are placed in the columns/the classes are placed in the rows.
    
    :param ivls:                intervals
    :param classes:             set of classes
    :return:                    class frequencies per interval
    """
    mat = [[0 for x in range(len(classes))] for y in range(2)]
    
    for i in range(2):
        for j in range(len(classes)):
            mat[i][j] = [pair[1] for pair in ivls[i]].count(classes[j])
            
    return mat


def exp_mat(freq_mat):
    """
    Calculates the matrix of expected values.
    
    :param freq_mat:            frequency matrix as returned by freq_mat(...) function
    :return:                    matrix of expected values
    """
    mat = [[0 for x in range(len(freq_mat[0]))] for y in range(2)]
    
    for i in range(2):
        for j in range(len(mat[0])):
            N_1 = calc_N_i(0, freq_mat)
            N_2 = calc_N_i(1, freq_mat)
            C_j = calc_C_j(j, freq_mat)
            
            mat[i][j] = calc_N_i(i, freq_mat) * (C_j / N_1 + N_2)
            
    return mat
   
         
def calc_N_i(i, freq_mat):
    """
    Calculates the number of elements in the i-th interval.
    
    :param i:                   index of the interval
    :param freq_mat:            frequency matrix
    :return:                    number of elements in the i-th interval
    """
    return sum(freq_mat[i])
    

def calc_C_j(j, freq_mat):
    """
    Calculates the number of occurences of j-th class across all intervals.
    
    :param j:                   index of the class
    :param freq_mat:            frequency matrix
    :return:                    number of times class j occurs across all intervals
    """
    return sum(col[j] for col in freq_mat)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
    
if __name__ == "__main__":
    """
    Main entry point.
    """
    # data to be discretized
    data = [(65, "No"), (68, "Yes"), (69, "Yes"), (70, "Yes"), (71, "No"), (64, "Yes"),
            (72, "No"), (72, "Yes"), (75, "Yes"), (75, "Yes"), (80, "No"), (81, "Yes"),
            (83, "Yes"), (85, "No")]
    
    # number of desired intervals
    cnt_ivl = 4
    merged = chi_merge(data, cnt_ivl)
    
    # print result
    for i in range(cnt_ivl):
        print(merged[i])
        