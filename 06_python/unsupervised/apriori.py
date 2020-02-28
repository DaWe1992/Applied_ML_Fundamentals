# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:51:53 2020

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import os
import itertools
import pandas as pd

from prettytable import PrettyTable


# -----------------------------------------------------------------------------
# Class Apriori
# -----------------------------------------------------------------------------

class Apriori:
    """
    Class Apriori.
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    

    def fit(self, X, s_min, c_min, rules_sort_by="support"):
        """
        Derives association rules from the data set.
        
        :param X:               data set
        :param s_min:           minimum support of a rule
        :param c_min:           minimum confidence of a rule
        :param rules_sort_by:   sort key for the rules
        :return:                association rules
        """
        self.X = X
        self.s_min = s_min
        self.c_min = c_min
        
        # step 1: find frequent item sets
        # ---------------------------------------------------------------------
        freq_sets = self.__find_frequent_item_sets()
        # step 2: generate association rules from the frequent item sets
        # ---------------------------------------------------------------------
        self.rules = self.__find_assoc_rules(freq_sets)
        self.rules = self.__calculate_metrics()
        
        return self.rules


    def __find_frequent_item_sets(self):
        """
        Finds all frequent item sets in the data.
        
        :return:                frequent item sets
        """
        # get column names
        k = 1
        C = [[item] for item in self.X.columns.values]
        S = list()
        
        while len(C) != 0:
            # remove all infrequent item sets from C
            S_k = [c for c in C if self.__is_frequent(c)]
            # create all sets with k + 1 elements which can be formed
            # by uniting two elements in S_k
            C = self.__create_new_item_sets(S_k, k)
            if k > 1:
                # remove all item sets from C,
                # where not all subsets of size k are in S_k
                C = [c for c in C if self.__contains_all_subsets(
                        S_k, list(itertools.combinations(c, k)))]
            # add results to S
            S += S_k
            k += 1
            
        return S


    def __create_new_item_sets(self, S_k, k):
        """
        Creates new item sets of length k + 1 by merging.
        
        :param S_k:             frequent item sets to be merged
        :param k:               current value of k
        :return:                new item sets
        """
        prefixes = [elem[:(k - 1)] for elem in S_k]
        
        # list of new item sets of length k + 1
        new_sets = []
        
        # list of already visited prefixes
        visited = []
        for prefix in prefixes:
            if prefix not in visited:
                visited.append(prefix)
                
                # get all postfixes
                postfixes = [S_k[index][-1:] \
                    for (index, elem) in enumerate(prefixes) \
                    if elem == prefix]
                
                # get all combinations of postfixes
                postfixes = list(itertools.combinations(postfixes, 2))
                
                # create new set using prefix and postfix
                for postfix in postfixes:    
                    new_sets.append(prefix + postfix[0] + postfix[1])
                    
        return new_sets
      
      
    def __contains_all_subsets(self, S_k, subsets):
        """
        Checks if all subsets are contained in S_k.
        
        :param S_k:             frequent item sets to be merged
        :param subsets:         list of all subsets
        :return:                true if all subsets are contained, false otherwise
        """
        for subset in subsets:
            subset = list(subset)
            if subset not in S_k:
                return False
            
        return True   


    def __find_assoc_rules(self, freq_sets):
        """
        Finds all association rules from the frequent item sets.
        This method is currently implemented as a brute force search.
        A more intelligent alternative is to use the anti-monotonicity property
        of the confidence to prune the search space.
        
        :param:                 frequent item sets
        :return:                association rules
        """
        rules = []
        
        # find candidate rules
        for item_set in freq_sets:   
            # only look at item sets with at least two elements,
            # since rules like {} => {xyz} or {xyz} => {} don't make any sense 
            if len(item_set) >= 2:        
                # try all different combinations of the items
                # and add them to the rule set
                for m in range(1, len(item_set)):
                    
                    bodies = [set(tpl) \
                        for tpl in list(itertools.combinations(item_set, m))]
                    
                    for body in bodies:    
                        rules.append([list(body), list(set(item_set) - body)])
        
        # check if the candidate rules are confident
        rules = [rule for rule in rules if self.__is_confident(rule)]
        
        return rules


    def __is_frequent(self, item_set):
        """
        Checks if an item set is frequent.
        
        :param item_set:        item set to be checkt for frequency
        :return:                true if item_set is frequent, false otherwise
        """        
        return self.__support(item_set) >= self.s_min
      
        
    def __is_confident(self, rule):
        """
        Checks if a rule is confident.
        
        :param rule:            rule to be checked
        :return:                true if rule is confident, false otherwise
        """
        return self.__confidence(rule) >= self.c_min
    
    
    def __support(self, item_set):
        """
        Calculates the support of an item set.
        
        :param item_set:        item set to be checkt for support
        :return:                support of item_set
        """
        result = self.X[item_set[0]]
        
        # perform a bitwise and of columns
        for item in item_set:
            result = result & self.X[item]
            
        return sum(result) / self.X.shape[0]
    
    
    def __confidence(self, rule):
        """
        Calculates the confidence of a rule.
        
        :param rule:            rule to be checked
        :return:                confidence of rule
        """
        body = rule[0]
        head = rule[1]
        return self.__support(body + head) / self.__support(body)
    
    
    def __lift(self, rule):
        """
        Calculates the lift of a rule.
        
        :param rule:            rule to calculate the lift for
        :return:                lift of rule
        """
        body = rule[0]
        head = rule[1]
        return self.__support(body + head) / \
            (self.__support(body) * self.__support(head))
    
    
    def __leverage(self, rule):
        """
        Calculates the leverage of a rule.
        
        :param rule:            rule to calculate the leverage for
        :return:                leverage of rule
        """
        body = rule[0]
        head = rule[1]
        return self.__support(body + head) - \
            (self.__support(body) * self.__support(head))
    
    
#    def __conviction(self, rule):
#        """
#        Calculates the conviction of a rule.
#        It can be interpreted as the ratio of the expected frequency
#        that the body occurs without the head.
#        
#        :param rule:            rule to calculate the conviction for
#        :return:                conviction of rule
#        """
#        head = rule[1]
#        return (1 - self.__support(head)) / (1 - self.__confidence(rule) + 0.01)


    def __calculate_metrics(self, sort_by="support"):
        """
        Calculates the metrics (support, confidence, lift, ...) for each rule.
        This method requires a call to 'fit()' first.
        
        :param sort_by:         sort key to sort the rules
        :return:                sorted list of rules with metrics
        """
        rules = []
        
        for rule in self.rules:
            body = rule[0]
            head = rule[1]
            
            # calculate metrics for all rules
            rules.append({
                "body":         body,
                "head":         head,
                "support":      "{:.5f}".format(self.__support(body + head)),
                "confidence":   "{:.5f}".format(self.__confidence(rule)),
#                "conviction":   "{:.5f}".format(self.__conviction(rule)),
                "lift":         "{:.5f}".format(self.__lift(rule)),
                "leverage":     "{:.5f}".format(self.__leverage(rule))
            })
        
        # sort rules by attribute specified
        return sorted(rules, key=lambda k: k[sort_by], reverse=True)
    
    
    def show(self):
        """
        Prints the rules in a tabular format.
        """
        # print some statistics about the data set
        print("Number of items:\t", self.X.shape[1])
        print("Number of transactions:\t", self.X.shape[0])
        
        # add table header
        t = PrettyTable([
            "rule", "support", "confidence", "lift", "leverage"
        ])
        
        t.align["rule"] = "l"
        
        # add rows (one for each rule)
        for rule in self.rules:       
            t.add_row([
                "{:45s} {:2s}  {:45s}".format(
                    str(set(rule["body"])), "->", str(set(rule["head"]))
                ),
                rule["support"],
                rule["confidence"],
#                rule["conviction"],
                rule["lift"],
                rule["leverage"]
            ])
                
        print(t)
        

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    """
    Main.
    """
    # items must be in ascending order
    
    # data set 1
    X_raw = {
        "bread":    [1, 0, 1, 0],
        "butter":   [1, 0, 0, 0],
        "coffee":   [0, 1, 1, 1],
        "milk":     [0, 1, 1, 1],
        "sugar":    [1, 1, 1, 0]        
    }
    
#    # data set 2
#    X_raw = {
#        "beer":     [0, 0, 1, 0, 0],
#        "bread":    [1, 0, 0, 1, 1],
#        "butter":   [0, 1, 0, 1, 0],
#        "diapers":  [0, 0, 1, 0, 0],
#        "milk":     [1, 0, 0, 1, 0]      
#    }
#    
    # convert to pandas data frame
    X = pd.DataFrame(data=X_raw)
#    
#    # alternatively: read data from csv file
#    path = os.path.abspath(
#        os.path.join(
#            os.path.dirname(__file__), "..", "data", "market_basket_analysis.csv"
#        )
#    )
#    X = pd.read_csv(path)
    
    apriori = Apriori()
    apriori.fit(X, s_min=0.5, c_min=1.00, rules_sort_by="support")
    apriori.show()
    