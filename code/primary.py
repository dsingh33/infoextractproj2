#!/usr/bin/env python3

import argparse
from word_hmm import WordHiddenMarkovModel as word_hmm
from let_hmm import LetterHiddenMarkovModel as let_hmm
from sil_hmm import SilenceHiddenMarkovModel as sil_hmm
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import random
import string
import numpy as np
import json

"""
This file is used to train Word HMMs using the command line.
It defines the following arguments:

--words -> This is a string that forms the file path of the training script, clsp.trnscr

--train -> This is a string that forms the file path of the training data, clsp.trnlbls

--test -> This is a string that forms the file path of the test data, clsp.devlbls

--iters -> This integer is the number of iterations the EM algorithm should run for

"""

ap = argparse.ArgumentParser()
ap.add_argument('--words', default='../data/clsp.trnscr')
ap.add_argument('--train', default='../data/clsp.trnlbls')
ap.add_argument('--test', default='../data/clsp.devlbls')
ap.add_argument('--iters', default=10)

args = ap.parse_args()

# Read in the words
words=[]
with open(args.words, 'r') as s:
    lines = s.readlines()[1:]
    for l in lines:
        words.append(l.strip('\n'))

# Extract labels for each utterance
lbl_seqs = []
with open(args.train, 'r') as t:
    lines = t.readlines()[1:]
    for j in range(len(lines)):
        lbl = lines[j].split(" ")[:-1]
        lbl_seqs.append(lbl)

# Jointly sort both the words and label sequences alphabetically
words, lbl_seqs = [list(tup) for tup in zip(*sorted(zip(words, lbl_seqs)))]

# Create a dict of component hmms that can be shared across word hmms
component_hmms = {}
# This vocabulary does not use k, q, or z
alphabet = list(string.ascii_lowercase)
alphabet.remove('k')
alphabet.remove('q')
alphabet.remove('z')
# Create a letter hmm for each letter in the alphabet
for letter in alphabet:
    lhmm = let_hmm(letter=letter)
    component_hmms.update({letter:lhmm})
# Create a silence hmm, only need one
shmm = sil_hmm()
component_hmms.update({'sil':shmm})

# Record average log prob per iteration k for plotting
log_probs_k_train = []

# TRAINING
# Train the hmm using EM algo for p iterations
for p in range(args.iters):
    # Create a dict of word hmms that can be reused across iterations
    # Automatically updates with new params from prev iter
    word_hmms = {}
    for word in set(words):
        whmm = word_hmm(letters=word, hmms=component_hmms)
        word_hmms.update({word:whmm})
    # Go through each of the 798 words and utterances alphabetically
    total_log_likelihood = []
    for i in range(len(words)):
        word = words[i]
        lbls = lbl_seqs[i]
        # Get word hmm
        whmm = word_hmms.get(word)
        # Do the forward and backward passes
        flog_prob, alphas = whmm.forward(lbl_seq=lbls)
        blog_prob, betas = whmm.backward(lbl_seq=lbls)
        print(flog_prob / len(lbls))
        print(blog_prob / len(lbls))
        # Collect counts for each letter and silence component
        whmm.collect_counts(lbl_seq=lbls, alphas=alphas, betas=betas, hmms=component_hmms)
        # CRITICAL: update the dictionary with this hmm
        word_hmms.update({word:whmm})
        total_log_likelihood.append(flog_prob)

    # At the end of each iteration, update parameters across all component hmms
    for hmm in component_hmms.values():
        hmm.update_params()
        component_hmms.update({hmm.letter:hmm})

    total_log_likelihood = logsumexp(total_log_likelihood)
    # Get the average total_log_likelihood
    log_probs_k_train.append(total_log_likelihood / len(lbl_seqs))

# Print the final log likelihood
print(log_probs_k_train[-1])

# Plot of the average log-likelihood as a function of # iterations
plt.plot(log_probs_k_train, color='red', label='Train')
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title(f'Average Log-Likelihood of Training Data Over {args.iters} Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/avg_log_prob_k.png')

# TESTING
# Extract word labels for each sample
devlbl_seqs = []
with open(args.test, 'r') as t:
    lines = t.readlines()[1:]
    for j in range(len(lines)):
        lbl = lines[j].split(" ")[:-1]
        devlbl_seqs.append(lbl)

word_hmms = {}
for word in words:
    whmm = word_hmm(letters=word, hmms=component_hmms)
    word_hmms.update({word:whmm})

out_file = open("test_results.txt", "w") 

# For each test sample
for i in range(len(devlbl_seqs)):
    lbls = devlbl_seqs[i]
    forward_log_probs = []
    # For each word hmm
    for word in set(words):
        whmm = word_hmms.get(word)
        # Do the forward and backward passes
        flog_prob, alphas = whmm.forward(lbl_seq=lbls)
        forward_log_probs.append(flog_prob / len(lbls))
        print(flog_prob / len(lbls))
    forward_log_probs = np.array(forward_log_probs)
    # Select the word with the max forward prob
    max_forward_log_prob = np.max(forward_log_probs)
    most_likely_word = words[np.argmax(forward_log_probs)]
    # Calculate confidence
    confidence = max_forward_log_prob - logsumexp(forward_log_probs)

    # Write out to file
    out_file.write(f'{most_likely_word}\t{confidence}\n')

out_file.close()