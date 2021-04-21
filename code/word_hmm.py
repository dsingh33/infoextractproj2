#!/usr/bin/env python3

import numpy as np
import numpy.ma as ma
from scipy.special import logsumexp
from math import inf, log, exp, sqrt
import math
import random
import string
import collections
from let_hmm import LetterHiddenMarkovModel as let_hmm

class WordHiddenMarkovModel():
    """
    This class defines an HMM for each letter.
    """

    def __init__(self, letters, hmms, vocab='../data/clsp.lblnames') -> None:
        """
        Initializes the HMM.

        num_states: Number of hidden states the HMM has at each time step.

        alt_init: Whether or not to use the alternative initialization.

        train_lbl_seq: List of chars that is the training data, NOT the file path.
        """

        # Number of hidden states, 3 for each letter and 5 for each silence
        self.num_states = 3*len(letters) + 10

        # Number of emitting arcs, 5 for each letter and 17 for each silence
        self.num_arcs = 17*2 + 5*len(letters)

        # Number of letter HMMs
        self.num_letters = len(list(letters))

        # Total number of arcs including null arcs
        self.tot_arcs = self.num_arcs + self.num_letters + 1

        # List of component letter HMMs
        graphemes = []
        shmm = hmms.get('sil')
        graphemes.append(shmm)
        for letter in list(letters):
            lhmm = hmms.get(letter)
            graphemes.append(lhmm)
        graphemes.append(shmm)
        self.graphemes = graphemes

        # Create and initialize params
        self.init_params()

        # Create dictionary that maps label names to index 0-255 for the emission matrix
        lblnames = []
        with open(vocab, 'r') as n:
            lines = n.readlines()[1:]
            for l in lines:
                lblnames.append(l.strip('\n'))
        lblnames = set(lblnames)
        lblnames = list(lblnames)

        vocab_index = [num for num in range(256)]
        self.vocab = {lblnames[i] : vocab_index[i] for i in range(len(vocab_index))}


    def init_params(self) -> None:
        """
        Get transition and emission probabilities from component HMMs.
        """
        transition = []
        emission = []
        for hmm in self.graphemes:
            transition.extend(hmm.transition)
            emission.append(hmm.emission)
        # Stack transtion probs on top of each other for arc probabiltiies
        self.transition = np.array(transition[:-1])
        # Vertically stack emission probabilities, TODO remove last null arc row
        self.emission = np.concatenate(tuple(emission), axis=0)[:-1,:]


    def forward(self, lbl_seq):
        """
        Forward alpha probabilities calculated using dynamic programming/token passing.

        lbl_seq: List of chars that is the label sequence of the utterance.
        """
        # Create an alpha for each state and populate with prob 0, -inf in log domain
        # The + 1 extra is because this is an arc-emitting HMM
        alphas = np.full((self.num_states, len(lbl_seq)+1), fill_value=-inf)

        # Repetition array: copy each state's alpha by the number of outgoing arcs
        # Makes computation extremely efficient
        # 4 4s for the first four states, 2 for the last silence state, 3 2s for each letter state
        # And then finally 4 4s and a final 1
        rep = [4]*4
        rep.extend([2])
        rep.extend([2]*(self.num_letters*3))
        rep.extend([4]*4)
        rep.extend([1])
        rep = np.array(rep)

        # Numpy array mask to make computation more efficient
        mask = np.full((self.num_states, self.tot_arcs), fill_value=-inf)
        
        mask[0,0] = 0.
        mask[1,[1,4,8,12]] = 0.
        mask[2,[2,5,9,13]] = 0.
        mask[3,[3,6,10,14]] = 0.
        mask[4,[7,11,15,16]] = 0.

        m=18
        j=5
        for x in range(self.num_letters):
            mask[j,m] = 0.
            mask[j+1,[m+1,m+2]] = 0.
            mask[j+2,[m+3,m+4]] = 0.
            j+=3
            m+=6

        mask[j,m] = 0.
        mask[j+1,[m+1,m+4,m+8,m+12]] = 0.
        mask[j+2,[m+2,m+5,m+9,m+13]] = 0.
        mask[j+3,[m+3,m+6,m+10,m+14]] = 0.
        mask[j+4,[m+7,m+11,m+15,m+16]] = 0.

        # Numpy array mask to make computation more efficient
        mask2 = np.full((self.num_letters+1, self.num_states), fill_value=-inf)
        mask2[0,[4,5]] = 0.
        h=7
        for x in range(self.num_letters):
            mask2[x+1,[h,h+1]] = 0.
            h+=3

        # For each character
        for lbl_id in np.arange(len(lbl_seq)+1):
            # First time step
            if lbl_id == 0:
                # Set intial state alpha to 1 (0 in log)
                # All others 0 (-inf in log)
                initial_state = 0
                alphas[initial_state, lbl_id] = 0.
                #print(alphas[:, lbl_id])
                continue
            
            # Vocab id is this character's index in the emission matrix
            # -1 because uses the prevous emission to calculate this time step's alpha
            vocab_id = self.vocab.get(lbl_seq[lbl_id-1])

            # Add emission probs to transition probs of each nonzero arc
            # This should be length self.num_arcs
            arc_probs = self.transition + self.emission[:,vocab_id]

            #print(np.exp(arc_probs[range(17,self.tot_arcs-17,6)]))

            # Get previous alpha
            a = alphas[:,lbl_id-1]

            # Calculate alphas of next step by adding arc probs to previous alphas
            a = np.repeat(a, rep)
            # zero out the indices corresponding to null arc transitions
            a[range(17,self.tot_arcs-17,6)] = 0.
            intmed = a + arc_probs
            #print(np.exp(intmed[range(17,self.tot_arcs-17,6)]))

            intmed = np.tile(intmed, (self.num_states,1)) + mask
            intmed = logsumexp(intmed, axis=1)

            # Add null probs to relevant nodes in the new alpha vector
            intmed2 = intmed
            intmed2[4]+=arc_probs[17]
            intmed2[range(7,self.num_states-5,3)]+=arc_probs[range(23,self.tot_arcs-17,6)]
            # Combine arc prob to the node the null arc transitions to
            intmed2 = np.tile(intmed2, (self.num_letters+1,1)) + mask2
            intmed[range(5,self.num_states-4,3)] = logsumexp(intmed2, axis=1)

            alphas[:,lbl_id] = intmed

        # Get log-likelihood, which is taken to be the bottom right alpha
        lbl_id = -1 # last char
        final_state = -1
        log_marginal_prob = alphas[final_state,lbl_id]
        
        return log_marginal_prob, alphas


    def backward(self, lbl_seq):
        betas = np.full((self.num_states, len(lbl_seq)+1), fill_value=-inf)

        rep = [1]
        rep.extend([4]*4)
        rep.extend([2]*(self.num_letters*3))
        rep.extend([2])
        rep.extend([4]*4)
        rep = np.array(rep)

        # Numpy array mask to make computation more efficient
        mask = np.full((self.num_states, self.tot_arcs), fill_value=-inf)
        
        mask[0,[0,1,2,3]] = 0.
        mask[1,[4,5,6,7]] = 0.
        mask[2,[8,9,10,11]] = 0.
        mask[3,[12,13,14,15]] = 0.
        mask[4,16] = 0. # add null here
        #mask[0,[0,1,5,9]] = 0.
        #mask[1,[2,6,10,13]] = 0.
        #mask[2,[3,7,11,14]] = 0.
        #mask[3,[4,8,12,15]] = 0.
        #mask[4,16] = 0. # add null here

        m=18
        j=5
        for x in range(self.num_letters):
            mask[j,[m,m+1]] = 0.
            mask[j+1,[m+2,m+3]] = 0.
            mask[j+2,m+4] = 0. # add null here
            m+=6
            j+=3

        mask[j,[m,m+1,m+2,m+3]] = 0.
        mask[j+1,[m+4,m+5,m+6,m+7]] = 0.
        mask[j+2,[m+8,m+9,m+10,m+11]] = 0.
        mask[j+3,[m+12,m+13,m+14,m+15]] = 0.
        mask[j+4,m+16] = 0. # add null here
        #mask[j,[m,m+1,m+5,m+9]] = 0.
        #mask[j+1,[m+2,m+6,m+10,m+13]] = 0.
        #mask[j+2,[m+3,m+7,m+11,m+14]] = 0.
        #mask[j+3,[m+4,m+8,m+12,m+15]] = 0.
        #mask[j+4,m+16] = 0. # add null here

        # Numpy array mask to make computation more efficient
        mask2 = np.full((self.num_letters+1, self.num_states), fill_value=-inf)
        mask2[0,[4,5]] = 0.
        h=7
        for x in range(self.num_letters):
            mask2[x+1,[h,h+1]] = 0.
            h+=3

        # For each character, reverse order
        for lbl_id in np.flip(np.arange(len(lbl_seq)+1)):
            # Final time step
            if lbl_id == len(lbl_seq):
                # Backward prob of last bottom right hidden state is 1
                # All others 0, ie -inf
                betas[-1,lbl_id] = 0.
                #print(betas[:,lbl_id])
                continue

            # Vocab id is this character's position in the emission matrix
            # +1 because uses the next emission to calculate this time step's beta
            vocab_id = self.vocab.get(lbl_seq[lbl_id])
            
            # Add emission probs to transition probs of each nonzero arc
            # This should be length self.num_arcs
            arc_probs = self.transition + self.emission[:,vocab_id]

            b = betas[:,lbl_id+1]
            #print(b)

            b_ext = list(b[0:4])
            b_ext.extend(list(b[1:5])*3)
            b_ext.extend(list(b[4:5]))
            b_ext.extend(list(np.repeat(b[5:self.num_states-5], 2)))
            b_ext.extend(list(b[self.num_states-5:self.num_states-4]))
            b_ext.extend(list(b[self.num_states-5:self.num_states-1]))
            b_ext.extend(list(np.repeat(b[self.num_states-4:self.num_states],3)))
            b_ext.extend(list(b[self.num_states-1:self.num_states]))
            b = np.array(b_ext)

            #b = np.repeat(b, rep)
            #print(b)
            b[range(17,self.tot_arcs-17,6)] = 0.
            intmed = b + arc_probs
            #print(arc_probs)
            #print(intmed)

            intmed = np.tile(intmed, (self.num_states,1)) + mask
            #print(mask[-4])
            #print(mask[-3])
            #print(mask[-2])
            #print(mask[-1])
            intmed = logsumexp(intmed, axis=1)
            #print(intmed2)

            intmed2 = intmed
            intmed2[5]+=arc_probs[17]
            intmed2[range(8,self.num_states-4,3)]+=arc_probs[range(23,self.tot_arcs-17,6)]
            intmed2 = np.tile(intmed2, (self.num_letters+1,1)) + mask2
            intmed[range(4,self.num_states-5,3)] = logsumexp(intmed2, axis=1)

            betas[:,lbl_id] = intmed

        # Get log likelihood, which should be the top left alpha
        lbl_id = 0
        initial_state = 0
        log_marginal_prob = betas[initial_state,lbl_id]
        
        return log_marginal_prob, betas

    def collect_counts(self, lbl_seq, alphas, betas, hmms): 
        # Records indices of all instances of each letter in the text
        # This will be useful when calculating counts for each letter
        self.indices = [np.where(np.array(lbl_seq) == letter)[0] for letter in self.vocab.keys()]

        # Counts at each timestep for each transition
        uncollected = np.full((self.tot_arcs, len(lbl_seq)), fill_value=-inf)

        rep = [4]*4
        rep.extend([2])
        rep.extend([2]*(self.num_letters*3))
        rep.extend([4]*4)
        rep.extend([1])
        rep_a = np.array(rep)

        for lbl_id in np.arange(len(lbl_seq)):
            # Expand alphas to be column vectors and betas to be row
            # Makes sense because alphas correspond to row index, which corresponds the orgin state
            # Betas correspond to col index, which corresponds to the destination state

            # Add log probs of left alpha, right beta, transition, and emission prob for each transition
            # Output should be a num_states x num_states matrix for each character in text
            #uncollected[:,:,char_id] = left_alphas + self.transition + e + right_betas
            vocab_id = self.vocab.get(lbl_seq[lbl_id])
            arc_probs = self.transition + self.emission[:,vocab_id]

            # Left alpha vector and right beta vector
            la = alphas[:,lbl_id]
            rb = betas[:,lbl_id+1]

            la = np.repeat(la, rep_a)

            rb_ext = list(rb[0:4])
            rb_ext.extend(list(rb[1:5])*3)
            rb_ext.extend(list(rb[4:5]))
            rb_ext.extend(list(np.repeat(rb[5:self.num_states-5], 2)))
            rb_ext.extend(list(rb[self.num_states-5:self.num_states-4]))
            rb_ext.extend(list(rb[self.num_states-5:self.num_states-1]))
            rb_ext.extend(list(np.repeat(rb[self.num_states-4:self.num_states],3)))
            rb_ext.extend(list(rb[self.num_states-1:self.num_states]))
            rb = np.array(rb_ext)

            # Add together to get probabilities of going through the arcs in this timestep
            intmed = la + arc_probs + rb

            # Get null counts
            # prb is previous right betas, ones for the same column of states as current alphas
            prb = betas[:,lbl_id]
            rb_ext = list(prb[0:4])
            rb_ext.extend(list(prb[1:5])*3)
            rb_ext.extend(list(prb[4:5]))
            rb_ext.extend(list(np.repeat(prb[5:self.num_states-5], 2)))
            rb_ext.extend(list(prb[self.num_states-5:self.num_states-4]))
            rb_ext.extend(list(prb[self.num_states-5:self.num_states-1]))
            rb_ext.extend(list(np.repeat(prb[self.num_states-4:self.num_states],3)))
            rb_ext.extend(list(prb[self.num_states-1:self.num_states]))
            prb = np.array(rb_ext)

            intmed[range(17,self.tot_arcs-17,6)] = la[range(5,self.num_states-4,3)] + arc_probs[range(17,self.tot_arcs-17,6)] + prb[range(6,self.num_states-3,3)]

            uncollected[:,lbl_id] = intmed

        # Add last column null count to previous column's null counts since there are more columns than timesteps
        # These will all be summed later on to form the overall transition counts so it doesn't matter
        la = alphas[:,-1]
        la = np.repeat(la, rep_a)
        uncollected[range(17,self.tot_arcs-17,6),-1] += la[range(5,self.num_states-4,3)] + arc_probs[range(17,self.tot_arcs-17,6)] + rb[range(6,self.num_states-3,3)]
        
        sil_hmm = self.graphemes[0]
        sil_hmm.collect_counts(uncollected[0:18], self.indices, len(lbl_seq), self.vocab)
        hmms.update({sil_hmm.letter:sil_hmm})
        m=18
        n=24
        # Collect counts for each component HMM
        for hmm in self.graphemes[1:-1]:
            hmm.collect_counts(uncollected[m:n], self.indices, len(lbl_seq), self.vocab)
            # CRITICAL: have to update and return dictionary of component hmms
            hmms.update({hmm.letter:hmm})
            m+=6
            n+=6

        sil_hmm = self.graphemes[-1]
        sil_hmm.collect_counts(uncollected[m:m+17], self.indices, len(lbl_seq), self.vocab)
        hmms.update({sil_hmm.letter:sil_hmm})