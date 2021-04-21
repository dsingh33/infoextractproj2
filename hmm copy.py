#!/usr/bin/env python3

import numpy as np
from scipy.special import logsumexp
from math import inf, log, exp, sqrt
import math
import random
import string

class HiddenMarkovModel():
    """
    This class defines an HMM.
    """

    def __init__(self, num_states: int, alt_init: bool, train_doc: str) -> None:
        """
        Initializes the HMM.

        num_states: Number of hidden states the HMM has at each time step.

        alt_init: Whether or not to use the alternative initialization.

        train_doc: List of chars that is the training data, NOT the file path.
        """

        # Number of hidden states
        self.num_states = num_states

        # Create dictionary that maps alphabet & ' ' to index 0-27 for emission matrix
        alphabet = list(string.ascii_lowercase) + [' ']
        vocab_index = [num for num in range(27)]
        self.vocab = {alphabet[i] : vocab_index[i] for i in range(len(vocab_index))}

        with open(train_doc, 'r') as f:
            train_doc = f.read()
            train_doc = list(train_doc)

        # Records indices of all instances of each letter in the text
        # This will be useful later on when calculating counts for each letter
        self.indices = [np.where(np.array(train_doc) == letter)[0] for letter in self.vocab.keys()]
        
        # Create and initialize params
        self.init_params(alt=alt_init, doc=train_doc)
        # Choose what initial hidden state will be with equal probability
        self.initial_state = random.choice([state for state in range(self.num_states)])


    def init_params(self, alt: bool, doc) -> None:
        """
        This function initializes the transition and emission log probabilities.
        The transitions are initialized as slight pertubations of 1/num_states,
        where all rows sum to 1. Rows indicate source state and columns destination
        state. The emissions can be initialized using the alternative method, which
        makes use of their relative frequencies in the data, or as slight pertubations
        of 1/|vocab| where each row sums to one. The rows are the states and the
        columbs are elements of the alphabet.
        NOTE: I have not actually made it so that the matrices are intialized wrt
        the number of states specified. Here it needs to be done manually.
        """
        if alt:
            # Alternative initialization of emission matrix
            # Calculate relative frequency of each letter in the document
            rel_freq = []
            for vocab_id in self.vocab.values():
                rel_freq.append(np.sum(len(self.indices[vocab_id])) / len(doc))
            rel_freq = np.array(rel_freq)
            
            # Find a random number for each letter and minus their total mean
            randoms = np.array(random.sample(range(1, 200), len(self.vocab)))
            randoms = randoms - np.mean(randoms)
            
            # Choose l such that emission_s0 is always positive for all elements.
            l = 1 / 100000

            emission_s0 = rel_freq - l * randoms
            emission_s1 = rel_freq + l * randoms
            self.emission = np.log(np.array([emission_s0, emission_s1]))
        else:
            # Initializes an emission matrix of size self.num_states x self.vocab
            # Turns this matrix into log probabilities
            emission_row1 = np.concatenate((np.repeat(0.0370, 13), np.repeat(0.0371, 13), np.array([0.0367])))
            emission_row2 = np.concatenate((np.repeat(0.0371, 13), np.repeat(0.0370, 13), np.array([0.0367])))
            self.emission = np.log(np.array([emission_row1, emission_row2]))

        # Initializes a transition matrix of size self.num_states x self.num_states
        # Turns this matrix into log probabilities
        transition = [[0.49, 0.51],
                      [0.51, 0.49]]
        self.transition = np.log(np.array(transition))

        # Initializations for 4 states, uncomment if running 4 states
        #transition = [[0.26, 0.27, 0.23, 0.24],
        #              [0.24, 0.26, 0.27, 0.23],
        #              [0.23, 0.24, 0.26, 0.27],
        #              [0.27, 0.23, 0.24, 0.26]]
        #self.transition = np.log(np.array(transition))

        #emission_row1 = np.concatenate((np.repeat(0.0369, 6), np.repeat(0.0372, 7), np.repeat(0.0369, 7), np.repeat(0.0372, 6), np.array([0.0367])))
        #emission_row2 = np.concatenate((np.repeat(0.0372, 6), np.repeat(0.0369, 7), np.repeat(0.0372, 7), np.repeat(0.0369, 6), np.array([0.0367])))
        #emission_row3 = np.concatenate((np.repeat(0.0370, 6), np.repeat(0.0371, 7), np.repeat(0.0370, 7), np.repeat(0.0371, 6), np.array([0.0367])))
        #emission_row4 = np.concatenate((np.repeat(0.0371, 6), np.repeat(0.0370, 7), np.repeat(0.0371, 7), np.repeat(0.0370, 6), np.array([0.0367])))
        #self.emission = np.log(np.array([emission_row1, emission_row2, emission_row3, emission_row4]))


    def forward(self, doc):
        """
        Forward alpha probabilities calculated using dynamic programming.

        doc: List of chars that is the document.
        """
        # Create an alpha for each state and populate with prob 0, -inf in log domain
        # The + 1 extra is for the states at the final time step, which don't emit anything
        alphas = np.full((self.num_states, len(doc)+1), fill_value=-inf)

        # For each character
        for char_id in np.arange(len(doc)+1):
            # First time step
            if char_id == 0:
                # Set intial state alpha to 1 (0 in log)
                # All others 0 (-inf in log)
                alphas[self.initial_state, char_id] = 0.
                continue
            
            # Vocab id is this character's index in the emission matrix
            # -1 because uses the prevous emission to calculate this time step's alpha
            vocab_id = self.vocab.get(doc[char_id-1])

            # Adding log probabilities of the previous alphas, the transitions and the emission
            # Should be a self.num_states by self.num_states matrix
            # array[0,1] = log(alpha0) + log(p('e'|s0)) + log(p(s1|s0))
            # From s0 to s1 ^^^
            # Need to expand dims to make broadcasting work
            a = np.expand_dims(alphas[:,char_id-1], axis=1)
            e = np.expand_dims(self.emission[:,vocab_id], axis=1)
            intmed = a + self.transition + e

            # Sum all of the probs in the columns, which means logsumexping all of the log probs
            alphas[:,char_id] = logsumexp(intmed, axis=0)

        # Final time step, single end state where all alphas are summed to get log-likelihood
        char_id = -1 # last char
        log_marginal_prob = logsumexp(alphas[:,char_id])
        
        return log_marginal_prob, alphas


    def backward(self, doc):
        betas = np.full((self.num_states, len(doc)+1), fill_value=-inf)

        # For each character, reverse order
        for char_id in np.flip(np.arange(len(doc)+1)):
            # Final time step
            if char_id == len(doc):
                # Backward prob of last hidden state is just 1
                betas[:,char_id] = np.repeat(1., self.num_states)
                continue

            # Vocab id is this character's position in the emission matrix
            vocab_id = self.vocab.get(doc[char_id])
            
            # Adding log probabilities of the next betas, the transitions and the emission
            # Should be a self.num_states by self.num_states matrix
            # array[0,1] = log(beta1) + log(p('e'|s0)) + log(p(s1|s0))
            # From s0 to s1 ^^^
            # Need to expand dims to make broadcasting work
            e = np.expand_dims(self.emission[:,vocab_id], axis=1)
            intmed = betas[:,char_id+1] + self.transition + e

            # Logsumexp across the rows, same as add all probs going out of a state
            betas[:,char_id] = logsumexp(intmed, axis=1)

        # The beta of the intial state should be the same as the log prob
        log_marginal_prob = betas[self.initial_state,0]
        
        return log_marginal_prob, betas


    def update_params(self, log_marginal_prob, alphas, betas, doc) -> None:
        """
        Updates the transition and emission matrices after alphas and betas have
        been obtained by running forward and backward.
        """
        # Counts at each timestep for each transition
        # With 2 states, 2x2x30000 tensor
        uncollected = np.full((self.num_states, self.num_states, len(doc)), fill_value=-inf)
        
        for char_id in np.arange(len(doc)):
            vocab_id = self.vocab.get(doc[char_id])

            # Expand alphas to be column vectors and betas to be row
            # Makes sense because alphas correspond to row index, which corresponds the orgin state
            # Betas correspond to col index, which corresponds to the destination state
            left_alphas = np.expand_dims(alphas[:,char_id], axis=1)
            e = np.expand_dims(self.emission[:,vocab_id], axis=1)
            right_betas = betas[:,char_id+1]

            # Add log probs of left alpha, right beta, transition, and emission prob for each transition
            # Output should be a num_states x num_states matrix for each character in text
            uncollected[:,:,char_id] = left_alphas + self.transition + e + right_betas
        
        # Normalize with marginal prob, unnecessary
        #uncollected -= log_marginal_prob

        # Collect counts for each transition, output should be same size as self.transition
        # This sums counts for each transition over all characters
        transition_counts = logsumexp(uncollected, axis=2)
        # Update transition matrix by normalizing by row sum, since rows of transition matrices
        # Must add up to 1 in normal prob domain, 0 in log domain, same as softmax
        # This divides by all other transitions from the left state, which is represented by rows
        self.transition = transition_counts - np.expand_dims(logsumexp(transition_counts, axis=1), axis=1)
        
        # num_states x num_states x 27
        # Same idea as transition_counts, except for each character
        emission_counts = np.full((self.num_states, self.num_states, len(self.vocab)), fill_value=-inf)

        # For each letter in the alphabet, sum all transition counts that emit that letter
        for vocab_id in self.vocab.values():
            # Create a boolean array same size as the document that indicates where each letter is
            where = np.full((len(doc)), False)
            where[self.indices[vocab_id]] = True
            unc = uncollected[:,:,where]
            emission_counts[:,:,vocab_id] = logsumexp(unc, axis=2)

        # This sums for each row to get c(letter, state) since left states emit letters
        # Result of sum should be num_states x 27, adding counts for all emissions out of a state
        emission_counts = logsumexp(emission_counts, axis=1)
        # Normalizing across rows, which is the same as p(e,state)/p(state)
        # This sums over all counts for all letters for each state, since those are the other options
        # Given the state
        emission_counts_sum_state = np.expand_dims(logsumexp(emission_counts, axis=1), axis=1)
        self.emission = emission_counts - emission_counts_sum_state


    def train(self, iters: int, train_doc: str, test_doc: str):
        """
        This function organizes and run everything.
        iters: Number of iterations.
        train_doc: File path to train document.
        test_doc: File path to test document.

        NOTE: Uncomment the commented lines if using 4 states.
        """
        with open(train_doc, 'r') as f:
            train_doc = f.read()
            train_doc = list(train_doc)

        with open(test_doc, 'r') as f:
            test_doc = f.read()
            test_doc = list(test_doc)

        # Record average log prob per iteration k for plotting
        log_probs_k_train = []
        log_probs_k_test = []
        # Record emission probabilities for all states for letter a for each state
        emission_probs_a_s0 = []
        emission_probs_a_s1 = []
        #emission_probs_a_s2 = []
        #emission_probs_a_s3 = []
        # For letter n
        emission_probs_n_s0 = []
        emission_probs_n_s1 = []
        #emission_probs_n_s2 = []
        #emission_probs_n_s3 = []

        # Run the EM algorithm for k iterations
        for k in range(iters):
            log_prob1, alphas = self.forward(doc=train_doc)
            log_prob2, betas = self.backward(doc=train_doc)
            log_prob_test, alphas2 = self.forward(doc=test_doc)
            # both marginals should be equal, round bc of numerical fluctuations
            #assert (round(log_prob1) == round(log_prob2))
            self.update_params(log_marginal_prob=log_prob1, alphas=alphas, betas=betas, doc=train_doc)

            log_probs_k_train.append(log_prob1 / len(train_doc))
            log_probs_k_test.append(log_prob_test / len(test_doc))

            emission_probs_a_s0.append(np.exp(self.emission[0,0]))
            emission_probs_a_s1.append(np.exp(self.emission[1,0]))
            #emission_probs_a_s2.append(np.exp(self.emission[2,0]))
            #emission_probs_a_s3.append(np.exp(self.emission[3,0]))

            emission_probs_n_s0.append(np.exp(self.emission[0,13]))
            emission_probs_n_s1.append(np.exp(self.emission[1,13]))
            #emission_probs_n_s2.append(np.exp(self.emission[2,13]))
            #emission_probs_n_s3.append(np.exp(self.emission[3,13]))

        print(log_probs_k_train[-1])
        print(log_probs_k_test[-1])

        # Record the final params for each state
        emission_probs_final_s0 = np.exp(self.emission[0,:]).tolist()
        emission_probs_final_s1 = np.exp(self.emission[1,:]).tolist()
        #emission_probs_final_s2 = np.exp(self.emission[2,:]).tolist()
        #emission_probs_final_s3 = np.exp(self.emission[3,:]).tolist()
        print(np.exp(self.transition))
        print(np.exp(self.emission))

        return log_probs_k_train, log_probs_k_test, emission_probs_a_s0, emission_probs_a_s1, emission_probs_n_s0, emission_probs_n_s1, emission_probs_final_s0, emission_probs_final_s1
        #return log_probs_k_train, log_probs_k_test, emission_probs_a_s0, emission_probs_a_s1, emission_probs_a_s2, emission_probs_a_s3, emission_probs_n_s0, emission_probs_n_s1, emission_probs_n_s2, emission_probs_n_s3, emission_probs_final_s0, emission_probs_final_s1, emission_probs_final_s2, emission_probs_final_s3




        # First silence HMM
            alphas[0,lbl_id] = a[0] + arc_probs[0]
            alphas[1,lbl_id] = logsumexp(a[0] + arc_probs[1], a[1] + arc_probs[4], a[2] + arc_probs[8], a[3] + arc_probs[12])
            alphas[2,lbl_id] = logsumexp(a[0] + arc_probs[2], a[1] + arc_probs[5], a[2] + arc_probs[9], a[3] + arc_probs[13])
            alphas[3,lbl_id] = logsumexp(a[0] + arc_probs[3], a[1] + arc_probs[6], a[2] + arc_probs[10], a[3] + arc_probs[14])
            alphas[4,lbl_id] = logsumexp(a[1] + arc_probs[7], a[2] + arc_probs[11], a[3] + arc_probs[15], a[4] + arc_probs[16])

            # Intermediate letter HMMs
            j = 5 # alpha idx tracker
            k = 0 # null prob idx tracker
            m = 17 # arc prob idx tracker
            for x in range(self.num_letters):
                alphas[j,lbl_id] = logsumexp(alphas[j-1,lbl_id] + self.null[k], a[j] + arc_probs[m])
                alphas[j+1,lbl_id] = logsumexp(a[j] + arc_probs[m+1], a[j+1] + arc_probs[m+2])
                alphas[j+2,lbl_id] = logsumexp(a[j+1] + arc_probs[m+3], a[j+2] + arc_probs[m+4])
                j+=3
                k+=1
                m+=5

            # Final silence HMM
            alphas[j,lbl_id] = logsumexp(alphas[j-1,lbl_id] + self.null[k], a[j] + arc_probs[m])
            alphas[j+1,lbl_id] = logsumexp(a[j] + arc_probs[m+1], a[j+1] + arc_probs[m+4], a[j+2] + arc_probs[m+8], a[j+3] + arc_probs[m+12])
            alphas[j+2,lbl_id] = logsumexp(a[j] + arc_probs[m+2], a[j+1] + arc_probs[m+5], a[j+2] + arc_probs[m+9], a[j+3] + arc_probs[m+13])
            alphas[j+3,lbl_id] = logsumexp(a[j] + arc_probs[m+3], a[j+1] + arc_probs[m+6], a[j+2] + arc_probs[m+10], a[j+3] + arc_probs[m+14])
            alphas[j+4,lbl_id] = logsumexp(a[j+1] + arc_probs[m+7], a[j+2] + arc_probs[m+11], a[j+3] + arc_probs[m+15], a[j+4] + arc_probs[m+16])

            #alphas[8] = logsumexp(alphas[7] + self.null[1], a[8] + arc_probs[22])
            #.........
            # First silence HMM
            j = self.num_states # beta idx tracker
            k =  len(self.null) - 1 # null prob idx tracker
            m = self.num_arcs # arc prob idx tracker
            betas[j,lbl_id] = b[j] + arc_probs[m]
            betas[j-1,lbl_id] = logsumexp(b[j] + arc_probs[m-1], b[j-1] + arc_probs[m-4], b[j-2] + arc_probs[m-8], b[j-3] + arc_probs[m-12])
            betas[j-2,lbl_id] = logsumexp(b[j] + arc_probs[m-2], b[j-1] + arc_probs[m-5], b[j-2] + arc_probs[m-9], b[j-3] + arc_probs[m-13])
            betas[j-3,lbl_id] = logsumexp(b[j] + arc_probs[m-3], b[j-1] + arc_probs[m-6], b[j-2] + arc_probs[m-10], b[j-3] + arc_probs[m-14])
            betas[j-4,lbl_id] = logsumexp(b[j-1] + arc_probs[m-7], b[j-2] + arc_probs[m-11], b[j-3] + arc_probs[m-15], b[j-4] + arc_probs[m-16])
            m-=17
            j-=5

            # Intermediate letter HMMs
            for x in range(self.num_letters):
                betas[j,lbl_id] = logsumexp(betas[j+1,lbl_id] + self.null[k], b[j] + arc_probs[m])
                betas[j-1,lbl_id] = logsumexp(b[j] + arc_probs[m-1], b[j-1] + arc_probs[m-2])
                betas[j-2,lbl_id] = logsumexp(b[j-1] + arc_probs[m-3], b[j-2] + arc_probs[m-4])
                j-=3
                k-=1
                m-=5

            # Final silence HMM
            betas[j,lbl_id] = logsumexp(alphas[j+1,lbl_id] + self.null[k], b[j] + arc_probs[m])
            betas[j-1,lbl_id] = logsumexp(b[j] + arc_probs[m-1], b[j-1] + arc_probs[m-4], b[j-2] + arc_probs[m-8], b[j-3] + arc_probs[m-12])
            betas[j-2,lbl_id] = logsumexp(b[j] + arc_probs[m-2], b[j-1] + arc_probs[m-5], b[j-2] + arc_probs[m-9], b[j-3] + arc_probs[m-13])
            betas[j-3,lbl_id] = logsumexp(b[j] + arc_probs[m-3], b[j-1] + arc_probs[m-6], b[j-2] + arc_probs[m-10], b[j-3] + arc_probs[m-14])
            betas[j-4,lbl_id] = logsumexp(b[j-1] + arc_probs[m-7], b[j-2] + arc_probs[m-11], b[j-3] + arc_probs[m-15], b[j-4] + arc_probs[m-16])



            j=0 # arc idx tracker
            m=0 # state idx tracker
            n=0 # beta idx tracker

            uncollected[j,lbl_id] = la[m] + arc_probs[j] + b[n]
            uncollected[j+1,lbl_id] = la[m] + arc_probs[j+1] + b[n+1]
            uncollected[j+2,lbl_id] = la[m] + arc_probs[j+2] + b[n+2]
            uncollected[j+3,lbl_id] = la[m] + arc_probs[j+3] + b[n+3]
            j+=4
            n+=1

            for m in range(1,5):
                uncollected[j,lbl_id] = la[m] + arc_probs[j] + b[n]
                uncollected[j+1,lbl_id] = la[m] + arc_probs[j+1] + b[n+1]
                uncollected[j+2,lbl_id] = la[m] + arc_probs[j+2] + b[n+2]
                uncollected[j+3,lbl_id] = la[m] + arc_probs[j+3] + b[n+3]
                j+=4

            uncollected[j,lbl_id] = la[m] + arc_probs[j] + b[m]
            uncollected_null[0,lbl_id] = alphas[m,lbl_id+1]] + self.null[0] + b[m+1]
            j+=1
            
            n=5
            m=5
            for m in range(5,self.num_arcs-1):
                uncollected[j,lbl_id] = la[m] + arc_probs[j] + b[n]
                uncollected[j+1,lbl_id] = la[m] + arc_probs[j+1] + b[n+1]
                j+=2
                n+=1
            uncollected[j,lbl_id] = la[m] + arc_probs[j] + b[m]

            m=4
            for k in range(len(self.null)):
                uncollected_null[k,lbl_id] = alphas[m,lbl_id+1]] + self.null[k] + b[m+1]
                m+=3


        # Normalize with marginal prob, unnecessary
        #uncollected -= log_marginal_prob

        # Should be length self.num_arcs
        transition_counts = logsumexp(uncollected, axis=1)
        null_counts = logsumexp(uncollected_null, axis=1)
        # Normalize
        transition_counts[0:16] -= np.flatten(logsumexpsoftmax(transition_counts[0:16].reshape(4,4), axis=1))
        transition_counts[16] -= transition_counts[16], null_counts[0]

        j=17
        for x in range(self.num_letters):
            transition_counts[j:j+4] -= np.flatten(softmax(transition_counts[j:j+4].reshape(2,2), axis=1))
            transition_counts[j+5]
            null_counts[k]
            j+=2

        self.transition = transition_counts
        self.null = null_counts

        # Collect counts for each transition, output should be same size as self.transition
        # This sums counts for each transition over all characters
        transition_counts = logsumexp(uncollected, axis=2)
        # Update transition matrix by normalizing by row sum, since rows of transition matrices
        # Must add up to 1 in normal prob domain, 0 in log domain, same as softmax
        # This divides by all other transitions from the left state, which is represented by rows
        self.transition = transition_counts - np.expand_dims(logsumexp(transition_counts, axis=1), axis=1)
        
        # num_states x num_states x 27
        # Same idea as transition_counts, except for each character
        emission_counts = np.full((self.num_arcs, len(self.vocab)), fill_value=-inf)

        # For each letter in the alphabet, sum all transition counts that emit that letter
        for vocab_id in self.vocab.values():
            # Create a boolean array same size as the lbl_sequment that indicates where each letter is
            where = np.full((len(lbl_seq)), False)
            where[self.indices[vocab_id]] = True
            unc = uncollected[:,where]
            emission_counts[:,vocab_id] = logsumexp(unc, axis=1)

        # Normalizing across rows, which is the same as p(e,state)/p(state)
        # This sums over all counts for all letters for each state, since those are the other options
        # Given the state
        self.emission = emission_counts - logsumexp(emission_counts, axis=1)