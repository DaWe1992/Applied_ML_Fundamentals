# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 14:19:46 2017

@author: Daniel Wehner
"""

# definition of training instances
training_data = [
    ["sunny", "hot", "normal", "strong", "warm", "same", "yes"],
    ["sunny", "hot", "high", "strong", "warm", "same", "yes"],
    ["rainy", "cool", "high", "strong", "warm", "change", "no"],
    ["sunny", "hot", "high", "strong", "cool", "change", "yes"]
]

# column names
col_names = [
    "Sky",
    "Temperature",
    "Humidity",
    "Windy",
    "Water",
    "Forecast",
    "Golf"
]

# possible values for each column
col_values = [
    ["sunny", "rainy"],
    ["hot", "cool"],
    ["normal", "high"],
    ["strong", "weak"],
    ["warm", "cool"],
    ["same", "change"]
]

class_labels = ["yes", "no"]

# Function findS
# find the most specific hypothesis
# that covers all training examples
#
# @param training_data
# @return hypothesis
def findS(training_data):
    seen_first_positive = False
    num_attributes = len(training_data[0]) - 1
    
    # initial hypothesis <0, 0, 0, ..., 0>
    hypothesis = ["0"] * (num_attributes - 1)
    
    for i in range(len(training_data)):
        if training_data[i][num_attributes] == class_labels[0]:
            
            # first positive example
            if not seen_first_positive:
                hypothesis = training_data[i][:num_attributes]
                seen_first_positive = True
                
            # consecutive positive example
            else:
                
                # iterate over hypothesis
                for j in range(num_attributes):
                    if training_data[i][j] != hypothesis[j]:
                        hypothesis[j] = "?"
            
        else:
            # do nothing
            pass
    
    return hypothesis


# Function findGSet
# find the most general hypotheses
# that cover all examples
#
# @param training_data
# @return hypotheses
def findGSet(training_data):
    # number of attributes
    num_attr = len(training_data[0]) - 1
    # initial hypotheses <?, ?, ?, ..., ?>
    G = [["?"] * (num_attr - 1)]
    
    # for each training example e
    for e in training_data:
        
        # if e is negative
        # ============================
        if e[-1] == class_labels[1]:
            
            # for all hypotheses h in G that cover e
            for h in [h for h in G if covers(h, e)]:
                G.remove(h) # G = G \ {h}
                
                # for every condition c in e that is not part of h
                for j in range(num_attr - 1):
                    if h[j] == "?":
                        
                        c = e[j]
                        # for all conditions c_ that negate c
                        for c_ in (set(col_values[j]) - {c}):
                            h_ = h[:] # copy list
                            h_[j] = c_ # h_ = h union {c_}
                            prev_pos = [data for data in training_data[:training_data.index(e)]
                                if data[-1] == class_labels[0]]
                            # if h_ covers all previous positive examples
                            if coversAll(h_, prev_pos): G.append(h_) # G = G union {h_}
                                    
        # if e is positive 
        # ============================              
        else:
            # remove all h in G that do not cover e
            for h in range(len(G)):
                if not covers(G[h], e): del G[h]
                    
    return G


# Function: covers
# checks if a hypothesis covers
# a specific example
#
# @param hypothesis
# @param example
# @return boolean
def covers(hypothesis, example):
    tmp = list(set(hypothesis))
    if len(tmp) == 1 and tmp[0] == "?": return True
    else:
        for i in range(len(hypothesis)):
            if hypothesis[i] != "?" and hypothesis[i] != example[i]:
                return False
            
        return True


# Function: coversAll
# checks if a hypothesis covers
# all given examples
#       
# @param hypothesis
# @param examples
# @return boolean
def coversAll(hypothesis, examples):
    return len([example for example in examples
        if covers(hypothesis, example)]) == len(examples)
            
# execute findS algorithm
print(findS(training_data))
print(findGSet(training_data))