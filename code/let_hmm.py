#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from math import inf
import random
import collections

class LetterHiddenMarkovModel():
    """
    This class defines an HMM for each letter.
    """

    def __init__(self, letter) -> None:
        """
        Initializes the HMM for specified letter.

        num_states: Number of hidden states the HMM has at each time step.
        num_arcs: 
        letter: Identification for which letter this HMM models.
        labels: Labels to use for initializing transition and emission probs.
        """

        # Number of hidden states
        self.num_states = 3
        
        # Number of non-zero probability non-null arcs per step
        self.num_arcs = 5

        # Create an identification for this hmm
        self.letter = letter
   
        # Create and initialize params
        self.init_params()

        # Create arc counters, 256 is the length of the vocab, includes null arc
        self.counter = np.full((self.num_arcs+1,256), fill_value=-inf)


    def init_params(self, train_labels='../data/clsp.trnlbls', sil_labels='../data/clsp.endpts', lbl_names='../data/clsp.lblnames') -> None:
        """
        Initialize transition and emission matrices using unigram frequency 
        of each label in training data.
        """
        # Initialize transition probabilities for 3 states
        # For efficiecy, store as single vector w/ len self.num_arcs + 1
        # Last one is a null transition
        transition = np.log(np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2]))
        self.transition = transition
        
        # Extract all 256 label names
        lblnames = []
        with open(lbl_names, 'r') as n:
            lines = n.readlines()[1:]
            for l in lines:
                lblnames.append(l.strip('\n'))
        lblnames = set(lblnames)
        
        # Extract endpoints for each sample
        endpts = []
        with open(sil_labels, 'r') as s:
            lines = s.readlines()[1:]
            for l in lines:
                endpt = [int(n) for n in l.split(" ")]
                endpts.append(endpt)
        
        # Extract word labels for each sample
        lbls = []
        with open(train_labels, 'r') as t:
            lines = t.readlines()[1:]
            for j in range(len(lines)):
                first = endpts[j][0]
                second = endpts[j][1]
                lbl = lines[j].split(" ")[:-1]
                lbls.extend(lbl[first:second-1])
        
        # Get frequencies of each label
        freqs = collections.Counter(lbls)

        # Add 1 smoothing for all samples not present
        # Without smoothing there are only 152 non-zero labels
        misslbls = lblnames ^ freqs.keys()
        for mlbl in misslbls:
            freqs[mlbl] = 1

        rel_freqs = np.array(list(freqs.values()))
        # Normalize to get relative frequencies
        rel_freqs = rel_freqs / np.sum(rel_freqs)
        # Initialize emissions array
        emission = np.tile(rel_freqs, (self.num_arcs,1))

        # Creat perturbation vector of length vocab, 256
        randoms = np.array(random.sample(range(1, 3000), len(rel_freqs)))
        randoms = randoms - np.mean(randoms)
        randoms = np.tile(randoms, (self.num_arcs,1))
            
        # Choose ls such that emission probs are always positive for all elements
        l = np.array([[-1.], [2.], [-1.3], [1.8], [-1.5]])
        l/=100000000.
        # Perturb the emission matrix while keeping valid probabilities
        emission = np.log(emission + l * randoms)
        # Add final row for null arc, 
        # #TODO make it 0, ie 1, so that when you add to transitions, it doesn't affect anythinhg
        null_row = np.repeat(0., 256)
        self.emission = np.vstack((emission, null_row))
        
        # Initialize transition probabilities for 3 states
        # For efficiecy, store as single vector w/ len self.num_arcs + 1
        # Last one is a null transition
        self.transition = np.log(np.array([0.8, 0.2, 0.8, 0.2, 0.8, 0.2]))

    def collect_counts(self, uncollected, lbl_indices, num_lbls, vocab):
        #print(uncollected)
        #print(np.count_nonzero(np.isnan(uncollected)))
        # For each letter in the alphabet, sum all transition counts that emit that letter
        for vocab_id in vocab.values():#range(256):
            # Create a boolean array same size as the lbl_sequment that indicates where each letter is
            where = np.full((num_lbls), False)
            where[lbl_indices[vocab_id]] = True
            unc = uncollected[:,where]
            if unc.shape[1] == 0:#np.sum(where) == 0:
                continue
            update = np.vstack((self.counter[:,vocab_id], logsumexp(unc, axis=1)))
            self.counter[:,vocab_id] = logsumexp(update, axis=0)

    def update_params(self):
        # Normalizing across rows, which is the same as p(e,state)/p(state)
        # This sums over all counts for all letters for each state, since those are the other options
        # Given the state
        #print(logsumexp(self.transition))
        transition = logsumexp(self.counter, axis=1)
        #print(logsumexp(transition))
        self.emission = self.counter - np.expand_dims(transition, axis=1)
        #print(logsumexp(self.emission, axis=1))

        #print(self.transition[0:2])
        #print(transition[0:2])
        #print(logsumexp(transition[0:2]))
        transition[0:2] -= logsumexp(transition[0:2])
        #print(logsumexp(self.transition[0:2]))
        transition[2:4] -= logsumexp(transition[2:4])
        transition[4:6] -= logsumexp(transition[4:6])
        self.transition = transition
        print(logsumexp(self.transition))

        # Reset counter
        self.counter = np.full((self.num_arcs+1,256), fill_value=-inf)