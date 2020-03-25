# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:10:06 2020
Code adapted from:
http://www.katrinerk.com/courses/python-worksheets/hidden-markov-models-for-pos-tagging-in-python

@author: Daniel Wehner
"""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------

import nltk

from nltk.corpus import brown


# -----------------------------------------------------------------------------
# Class HMM (Hidden Markov Model)
# -----------------------------------------------------------------------------

class HMM():
    """
    Class HMM.
    
    Given an annotated corpus of sequences, a hiddem Markov model aims
    to find the most probable tag sequence for unseen sequences.
    
    Tags = hidden state (t)
    Sequence elements = observed state (o)
    """
    
    def __init__(self):
        """
        Constructor.
        """
        pass
    
    
    def fit(self, data):
        """
        Fits a hidden Markov model to the data.
        
        :param data:            training data (e.g. corpus comprising n sentences)
                                of the following form (flat list):
                                [(START, START), (Tag{1}, Word{1}), ... (Tag{M}, Word{M}), (END, END), ...]
        """
        self.data = data
        # get a list of unique tags
        self.tagset = set([tag for (tag, word) in data])
        
        # compute emission and transition probabilities
        self.__comp_prob()
    
        
    def predict(self, X=["I", "want", "to", "race"]):
        """
        Predicts the most probable tag sequence for a given sequence.
        
        :param X:               sequence for which to find the best tag sequence
        :return:                most probable tag sequence
        """
        # get the backpointers and the seed
        backpointer, seed = self.__decode(X)
        # invert the list of backpointers
        backpointer.reverse()
        # initialize the best_seq
        best_seq = ["END", seed]
        
        # go through the inverted list of backpointers
        curr_best_tag = seed
        for bp in backpointer:
            best_seq.append(bp[curr_best_tag])
            curr_best_tag = bp[curr_best_tag]
        
        # reverse the sequence
        best_seq.reverse()

        return best_seq[1:-1]
    
    
    def __comp_prob(self):
        """
        Computes the emission and transition probabilities
        using maximum likelihood estimation (MLE).
        """
        # ---------------------------------------------------------------------
        # compute emission probabilities
        # p(o{i}|t{i}) = count(t{i}, o{i}) / count(t{i})
        # ---------------------------------------------------------------------
        self.e = nltk.ConditionalProbDist(
            nltk.ConditionalFreqDist(self.data), nltk.MLEProbDist)

        # ---------------------------------------------------------------------
        # compute transition probabilities
        # p(t{i}|t{i-1}) = count(t{i-1}, t{i}) / count(t{i-1})
        # ---------------------------------------------------------------------
        cfd_tags = nltk.ConditionalFreqDist(
            nltk.bigrams([tag for (tag, word) in self.data]))
        self.t = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
    
    
    def __decode(self, X):
        """
        Performs the decoding step using the Viterbi algorithm.
        
        :param X:               sequence for which to find the best tag sequence
        :return:                list of backpointers and seed element
        """
        viterbi = []; backpointer = []
        
        # go over all words in the sentence
        for w in range(len(X)):
            vt = {}; bp = {}
            # go over all tags in the tagset
            for tag in self.tagset:
                # do not record anything for the START tag
                if tag == "START": continue
                # first word in the sentence
                # -------------------------------------------------------------
                if w == 0:
                    vt[tag] = self.t["START"].prob(tag) * self.e[tag].prob(X[0])
                    bp[tag] = "START"
                # all other words in the sentence
                # -------------------------------------------------------------
                else:
                    prev_vt = viterbi[-1]
                    # if this tag is X and the current word is w,
                    # then find the previous tag Y, which maximizes
                    # prev_viterbi[Y] * p(X|Y) * p(w|X)
                    best_prev = max(
                        prev_vt.keys(), key = lambda prev_tag: \
                            prev_vt[prev_tag] * self.t[prev_tag].prob(tag) * \
                            self.e[tag].prob(X[w]))
            
                    vt[tag] = prev_vt[best_prev] * \
                        self.t[best_prev].prob(tag) * self.e[tag].prob(X[w])
                    bp[tag] = best_prev
        
            viterbi.append(vt)
            backpointer.append(bp)
        
        # END tag
        # ---------------------------------------------------------------------
        prev_vt = viterbi[-1]
        best_prev = max(prev_vt.keys(),
            key = lambda prev_tag: prev_vt[prev_tag] * self.t[prev_tag].prob("END"))
        
        print("Probability of best sequence: {}".format(
            prev_vt[best_prev] * self.t[best_prev].prob("END")))
        
        return backpointer, best_prev


# -----------------------------------------------------------------------------
# Data creation
# -----------------------------------------------------------------------------
        
def get_data_for_pos_tagging():
    """
    Gets the data for POS tagging (brown corpus; included in NLTK).
    
    :return:                data
    """
    # flat list of sentences
    data = []
    
    # go over all sentences in the corpus
    for sent in brown.tagged_sents():
        # add START tag to each sentence
        data.append(("START", "START"))
        # simplify the tag set (only use first two characters)
        data.extend([(tag[:2], word) for (word, tag) in sent])
        # add END tag to each sentence
        data.append(("END", "END"))
        
    return data


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
        
if __name__ == "__main__":
    """
    Main entry point.
    """
    pos_data = get_data_for_pos_tagging()
    
    new_sent = ["I", "like", "food", "."]
    
    # train hidden Markov model
    # -------------------------------------------------------------------------
    hmm = HMM()
    hmm.fit(pos_data)
    pred = hmm.predict(new_sent)
    
    # print prediction
    print("The sentence was:")
    print(new_sent)
    print()
    print("The prediction is:")
    print(pred)
    
"""
-------------------------------------------------------------------------------
Brown tag set (for POS tagging):
-------------------------------------------------------------------------------
Tag     Explanation
-------------------------------------------------------------------------------
.	    sentence (. ; ? *)
(	    left paren
)    	right paren
*   	not, n't
--   	dash
,   	comma
:   	colon
ABL  	pre-qualifier (quite, rather)
ABN 	pre-quantifier (half, all)
ABX 	pre-quantifier (both)
AP  	post-determiner (many, several, next)
AT  	article (a, the, no)
BE  	be
BED  	were
BEDZ	was
BEG	    being
BEM	    am
BEN	    been
BER	    are, art
BBB  	is
CC  	coordinating conjunction (and, or)
CD  	cardinal numeral (one, two, 2, etc.)
CS  	subordinating conjunction (if, although)
DO  	do
DOD	    did
DOZ  	does
DT  	singular determiner/quantifier (this, that)
DTI  	singular or plural determiner/quantifier (some, any)
DTS 	plural determiner (these, those)
DTX 	determiner/double conjunction (either)
EX  	existential there
FW  	foreign word (hyphenated before regular tag)
HL  	word occurring in the headline (hyphenated after regular tag)
HV  	have
HVD  	had (past tense)
HVG	    having
HVN  	had (past participle)
HVZ 	has
IN  	preposition
JJ  	adjective
JJR  	comparative adjective
JJS 	semantically superlative adjective (chief, top)
JJT 	morphologically superlative adjective (biggest)
MD  	modal auxiliary (can, should, will)
NC  	cited word (hyphenated after regular tag)
NN  	singular or mass noun
NN$ 	possessive singular noun
NNS	    plural noun
NNS$ 	possessive plural noun
NP	    proper noun or part of name phrase
NP$ 	possessive proper noun
NPS 	plural proper noun
NPS$	possessive plural proper noun
NR  	adverbial noun (home, today, west)
NRS 	plural adverbial noun
OD  	ordinal numeral (first, 2nd)
PN  	nominal pronoun (everybody, nothing)
PN$  	possessive nominal pronoun
PP$ 	possessive personal pronoun (my, our)
PP$$	second (nominal) possessive pronoun (mine, ours)
PPL 	singular reflexive/intensive personal pronoun (myself)
PPLS 	plural reflexive/intensive personal pronoun (ourselves)
PPO 	objective personal pronoun (me, him, it, them)
PPS 	3rd. singular nominative pronoun (he, she, it, one)
PPSS	other nominative personal pronoun (I, we, they, you)
QL  	qualifier (very, fairly)
QLP 	post-qualifier (enough, indeed)
RB  	adverb
RBR  	comparative adverb
RBT 	superlative adverb
RN  	nominal adverb (here, then, indoors)
RP  	adverb/particle (about, off, up)
TL  	word occurring in title (hyphenated after regular tag)
TO  	infinitive marker to
UH  	interjection, exclamation
VB  	verb, base form
VBD  	verb, past tense
VBG	    verb, present participle/gerund
VBN 	verb, past participle
VBP 	verb, non 3rd person, singular, present
VBZ 	verb, 3rd. singular present
WDT 	wh- determiner (what, which)
WP$ 	possessive wh- pronoun (whose)
WPO 	objective wh- pronoun (whom, which, that)
WPS 	nominative wh- pronoun (who, which, that)
WQL 	wh- qualifier (how)
WRB 	wh- adverb (how, where, when)
"""
