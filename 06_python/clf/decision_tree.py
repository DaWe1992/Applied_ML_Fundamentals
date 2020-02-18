# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:00:30 2020

@author: Daniel Wehner
@see: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import numpy as np


# -----------------------------------------------------------------------------
# Class DecisionTree
# -----------------------------------------------------------------------------

class DecisionTree:
    """
    Class DecisionTree.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, X, y, max_depth=10, min_size=10):
        """
        Fits a decision tree classifier to the data.
        
        :param X:               training data (features)
        :param y:               training data (labels)
        :param max_depth:       maximum depth of the tree
        :param min_size:        minimum number of examples per node
        :return:                root node of the tree
        """
        data = np.c_[X, y].tolist()
        self.root = self.__get_split(data)
        self.__split(self.root, max_depth, min_size, 1)
        
    
    def predict(self, data):
        """
        Predicts the label of unseen data.
        
        :param data:            data to predict class labels for
        :return:                predictions
        """
        pred = []
        for row in data:
            node = self.root
            pred.append(self.__get_prediction(node, row))
                    
        return np.asarray(pred)
    
    
    def __get_prediction(self, node, row):
        """
        Gets the prediction for a single example.
        
        :param node:            current decision node
        :param row:             current instance to be classified
        :return:                prediction for current instance
        """
        if row[node["index"]] < node["value"]:
            if isinstance(node["left"], dict):
                return self.__get_prediction(node["left"], row)
            else:
                return node["left"]
        else:
            if isinstance(node["right"], dict):
                return self.__get_prediction(node["right"], row)
            else:
                return node["right"]
        
        
    def __get_split(self, data):
        """
        Selects the best split point for a data set.
        
        :param data:            data set
        :return:                best split index
        """
        classes = list(set(row[-1] for row in data))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        
        for i in range(len(data[0]) - 1):
            for row in data:
                groups = self.__test_split(i, row[i], data)
                gini = self.__gini_index(groups, classes)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = \
                        i, row[i], gini, groups
                    
        return {
            "index": b_index,
            "value": b_value,
            "groups": b_groups
        }
        
        
    def __test_split(self, i, value, data):
        """
        Splits the data set based on an attribute and an attribute value.
        
        :param i:               index of feature         
        :param value:           value for the split
        :param data:            data set
        :return:                split data set (left split, right split)
        """
        left, right = list(), list()
        
        for row in data:
            if row[i] < value:
                left.append(row)
            else:
                right.append(row)
                
        return left, right
        
        
    def __gini_index(self, groups, classes):
        """
        Calculates the gini index for a split.
        
        :param groups:          splits
        :param classes:         class values
        :return:                gini index for the split
        """
        # count all samples at split point
        n_instances = float(sum([len(group) for group in groups]))
        # sum weighted gini index for each group
        gini = 0.0
        
        for group in groups:
            size = float(len(group))
            # avoid division by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                p = [row[-1] for row in group].count(class_val) / size
                score += p * p
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)
            
        return gini
    
    
    def __to_terminal(self, group):
        """
        Creates a terminal node value.
        
        :param group:           instances in node
        :return:                leaf value
        """
        outcomes = [row[-1] for row in group]
        
        return max(set(outcomes), key=outcomes.count)
    
    
    def __split(self, node, max_depth, min_size, depth):
        """
        Creates child splits for a node or produces a terminal
        """
        left, right = node["groups"]
        del(node["groups"])
        # check for a no split
        if not left or not right:
            node["left"] = node["right"] = self.__to_terminal(left + right)
            return
        # check for max depth
        if depth >= max_depth:
            node["left"], node["right"] = self.__to_terminal(left), self.__to_terminal(right)
            return
        
        # process left child
        if len(left) <= min_size:
            node["left"] = self.__to_terminal(left)
        else:
            node["left"] = self.__get_split(left)
            self.__split(node["left"], max_depth, min_size, depth + 1)
            
        # process right child
        if len(right) <= min_size:
            node["right"] = self.__to_terminal(right)
        else:
            node["right"] = self.__get_split(right)
            self.__split(node["right"], max_depth, min_size, depth + 1)
            
            
    def __print_tree(self, node, depth=0):
        """
        Prints the decision tree.
        
        :param node:            node to be printed
        :param depth:           depth of the node in the tree
        """
        if isinstance(node, dict):
            print("%s[X%d < %.3f]" % ((depth * "..", (node["index"] + 1), node["value"])))
            self.__print_tree(node["left"], depth + 1)
            self.__print_tree(node["right"], depth + 1)
        else:
            print("%s[%s]" % ((depth * "..", node)))
            
            
    def visualize(self):
        """
        Visualizes the tree.
        """
        self.__print_tree(self.root)
        