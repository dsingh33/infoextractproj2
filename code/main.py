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
This file is used to train hmm.py using the command line.
It defines the following arguments:

--train -> This is a string that forms the file path of the training data

--test -> This is a string that forms the file path of the test data

--nstates -> This integer is the number of hidden states the HMM should have

--iters -> This integer is the number of iterations the EM algorithm should run for

--altint -> This boolean flag indicates whether you want to use the alternative intialization

NOTE: Uncomment the commented lines if using 4 states, code does not account for this automatically
"""

ap = argparse.ArgumentParser()
#ap.add_argument('--train', default="../data/train.txt")
#ap.add_argument('--test', default="../data/test.txt")
#ap.add_argument('--nstates', default=2)
ap.add_argument('--iters', default=3)
#ap.add_argument('--altinit', action='store_true')

args = ap.parse_args()

words=[]
with open('../data/clsp.trnscr', 'r') as s:
    lines = s.readlines()[1:]
    for l in lines:
        words.append(l.strip('\n'))

# Extract word labels for each sample
lbl_seqs = []
with open('../data/clsp.trnlbls', 'r') as t:
    lines = t.readlines()[1:]
    for j in range(len(lines)):
        lbl = lines[j].split(" ")[:-1]
        lbl_seqs.append(lbl)

# Jointly sort both the words and label sequences alphabetically
#TODO: temporary, eliminate
words = words[0:50]
lbl_seqs = lbl_seqs[0:50]
words, lbl_seqs = [list(tup) for tup in zip(*sorted(zip(words, lbl_seqs)))]

# Create a dict of component hmms that can be shared across word hmms
component_hmms = {}
alphabet = list(string.ascii_lowercase)
alphabet.remove('k')
alphabet.remove('q')
alphabet.remove('z')
for letter in alphabet:
    lhmm = let_hmm(letter=letter)
    component_hmms.update({letter:lhmm})
shmm = sil_hmm()
component_hmms.update({'sil':shmm})

# Record average log prob per iteration k for plotting
log_probs_k_train = []

# Train the hmm using EM algo for p iterations
for p in range(args.iters):
    print(p)
    # Create a dict of word hmms that can be reused across iterations
    # Automatically updates with new params from prev iter
    word_hmms = {}
    for word in words:
        whmm = word_hmm(letters=word, hmms=component_hmms)
        word_hmms.update({word:whmm})
    # Go through each of the 798 words and utterances alphabetically
    total_log_likelihood = []
    for i in range(len(words)):
        word = words[i]
        #print(word)
        lbls = lbl_seqs[i]
        # Get word hmm
        whmm = word_hmms.get(word)
        # Do the forward and backward passes
        flog_prob, alphas = whmm.forward(lbl_seq=lbls)
        blog_prob, betas = whmm.backward(lbl_seq=lbls)
        print(flog_prob)
        #print(blog_prob)
        # Collect counts for each letter and silence component
        whmm.collect_counts(lbl_seq=lbls, alphas=alphas, betas=betas, hmms=component_hmms)
        # CRITICAL: update the dictionary with this hmm
        word_hmms.update({word:whmm})

        total_log_likelihood.append(flog_prob)

    # At the end of each iteration, update parameters across all hmms
    for hmm in component_hmms.values():
        #print(" ")
        #print(hmm.letter)
        hmm.update_params()
        #print(" ")
        component_hmms.update({hmm.letter:hmm})

    #assert (len(total_log_likelihood) == len(lbl_seqs))
    total_log_likelihood = logsumexp(total_log_likelihood)
    log_probs_k_train.append(total_log_likelihood / len(lbl_seqs))

print(log_probs_k_train[-1])

# Plot of the average log-likelihood as a function of # iterations
plt.plot(log_probs_k_train, color='red', label='Train')
#plt.plot(log_probs_k_test, color='blue', label='Test')
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title('Average Log-Likelihood of Training Data Over 100 Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/avg_log_prob_k.png')


# Extract word labels for each sample
devlbl_seqs = []
with open('../data/clsp.devlbls', 'r') as t:
    lines = t.readlines()[1:]
    for j in range(len(lines)):
        lbl = lines[j].split(" ")[:-1]
        devlbl_seqs.append(lbl)

word_hmms = {}
for word in words:
    whmm = word_hmm(letters=word, hmms=component_hmms)
    word_hmms.update({word:whmm})

test_results = {}

for i in range(len(devlbl_seqs)):
    lbls = devlbl_seqs[i]
    # Get word hmm
    forward_log_probs = []
    for word in set(words):
        whmm = word_hmms.get(word)
        # Do the forward and backward passes
        flog_prob, alphas = whmm.forward(lbl_seq=lbls)
        forward_log_probs.append(flog_prob)
        print(flog_prob)
    forward_log_probs = np.array(forward_log_probs)
    max_forward_log_prob = np.max(forward_log_probs)
    most_likely_word = words[np.argmax(forward_log_probs)]
    confidence = max_forward_log_prob - logsumexp(forward_log_probs)
    test_results.update({i:{'word':most_likely_word, 'confidence':confidence}})

out_file = open("test_results.json", "w") 
json.dump(test_results, out_file) 
out_file.close()