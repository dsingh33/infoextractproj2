#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from math import inf, log, exp, sqrt
import math
import random
import string
import collections

class SilenceHiddenMarkovModel():
    """
    This class defines an HMM for each letter.
    """

    def __init__(self) -> None:
        """
        Initializes the HMM for silence.

        num_states: Number of hidden states the HMM has at each time step.

        num_arcs: Number of non-null, non-zero probability arcs at each time step.

        tot_arcs: Total number of arcs, including null arcs at each time step.

        letter: Identification for silence.

        This file uses clsp.trnlbls, clsp.endpts, and clsp.lblnames for initialization
        """

        # Number of hidden states
        self.num_states = 5

        # Number of non-zero prob arcs
        self.num_arcs = 17

        # Total arcs including null arcs
        self.tot_arcs = 18

        # Create an identification for this hmm
        self.letter = 'sil'
   
        # Create and initialize params
        self.init_params()

        # Create arc counters, 256 is the length of the vocab, includes null arc
        self.counter = np.full((self.tot_arcs,256), fill_value=-inf)


    def init_params(self, train_labels='../data/clsp.trnlbls', sil_labels='../data/clsp.endpts', lbl_names='../data/clsp.lblnames') -> None:
        """
        Initialize transition and emission matrices using unigram frequency of each label in training data.

        train_labels: Training labels for each utterance, clsp.trnlbls, used to get unigram frequencies.
        sil_labels: Endpoint indices for each training label sequence.
        lbl_names: List of each label name, 256 total.
        
        NOTE: In order to be efficient and not multiply sparse matrices, I am not storing the parameter
        matrices using their usual shape. I will explain more in comments.
        """
        # Initialize transition probabilities for  states
        transition = [0.25]*16
        transition.extend([0.75, 0.25])
        self.transition = np.log(transition)

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

        # Extract silence labels for each sample
        # Extract word labels for each sample
        lbls = []
        with open(train_labels, 'r') as t:
            lines = t.readlines()[1:]
            for j in range(len(lines)):
                first = endpts[j][0]
                second = endpts[j][1]
                lbl = lines[j].split(" ")[:-1]
                lbls.extend(lbl[:first])
                lbls.extend(lbl[second-1:])

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
        # One l for each arc row
        l = np.array([[-0.5], [0.5], [-0.6], [0.6], [-0.7], [0.7], [-0.8], [0.8], [-0.9], [0.9], [-1.], [1.1], [-1.1], [1.2], [-1.2], [1.3], [-0.3]])
        l/=1000000000.
        # Perturb the emission matrix while keeping valid probabilities
        emission = np.log(emission + l * randoms)
        # Add final row for null arc, this is a DUMMY row only for computation purposes
        null_row = np.repeat(0., 256)
        self.emission = np.vstack((emission, null_row))


    def collect_counts(self, uncollected, lbl_indices, num_lbls, vocab):
        """
        This function collects counts, given a matrix that is tot_arcs x number of timesteps.

        uncollected: Matrix that holds the arc probability for each arc for each timestep.

        lbl_indices: Indicates for each label which timesteps have that label.

        num_lbls: Indicates the total number of time_steps, the length of the label sequence.

        vocab: Dictionary that indicates the label and the index that corresponds to each label.
        """
        # For each label in vocab, sum all transition counts that emit that letter
        for vocab_id in vocab.values():#range(256):
            # Create a boolean array same size as the lbl_sequment that indicates where each letter is
            where = np.full((num_lbls), False)
            where[lbl_indices[vocab_id]] = True
            unc = uncollected[:,where]
            # Do not collect counts if this label is not present
            if unc.shape[1] == 0:
                continue
            # If this is the last silence hmm with no null arc, do not update null arc counts
            if unc.shape[0] == self.num_arcs:
                update = np.vstack((self.counter[:-1,vocab_id], logsumexp(unc, axis=1)))
                self.counter[:-1,vocab_id] = logsumexp(update, axis=0) #TODO adding? -inf
            else:
                update = np.vstack((self.counter[:,vocab_id], logsumexp(unc, axis=1)))
                self.counter[:,vocab_id] = logsumexp(update, axis=0)


    def update_params(self):
        """
        This function updates all of the parameters using self.counter.
        """
        # Sum all of the counts across self.counter to get the transition counts
        transition = logsumexp(self.counter, axis=1)
        # Normalize self.counter by transition counts to get the new emission matrix
        self.emission = self.counter - np.expand_dims(transition, axis=1)

        # Normalize transition arc probabilities by all of the transitions that are emitted by the same state
        transition[0:4] -= logsumexp(transition[0:4])
        transition[4:8] -= logsumexp(transition[4:8])
        transition[8:12] -= logsumexp(transition[8:12])
        transition[12:16] -= logsumexp(transition[12:16])
        transition[16:18] -= logsumexp(transition[16:18])
        self.transition = transition

        # Reset counter
        self.counter = np.full((self.num_arcs+1,256), fill_value=-inf)