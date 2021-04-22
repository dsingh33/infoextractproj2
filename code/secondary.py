#!/usr/bin/env python3

import argparse
from word_hmm import WordHiddenMarkovModel as word_hmm
from let_hmm import LetterHiddenMarkovModel as let_hmm
from sil_hmm import SilenceHiddenMarkovModel as sil_hmm
import matplotlib.pyplot as plt
from scipy.special import logsumexp
import random
import string
import json
import numpy as np

"""
This file is used to train word hmm using the command line. This is for the secondary system.
It is the same the primary system except it divides the data into a held out and training portion.
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

words=[]
with open(args.words, 'r') as s:
    lines = s.readlines()[1:]
    for l in lines:
        words.append(l.strip('\n'))

lbl_seqs = []
with open(args.train, 'r') as t:
    lines = t.readlines()[1:]
    for j in range(len(lines)):
        lbl = lines[j].split(" ")[:-1]
        lbl_seqs.append(lbl)

words = words[0:100]
lbl_seqs = lbl_seqs[0:100]
words, lbl_seqs = [list(tup) for tup in zip(*sorted(zip(words, lbl_seqs)))]

# SPLIT DATSET INTO HELDOUT AND TRAINING PORTION
words_heldout = []
words_train = []
lbl_seqs_heldout = []
lbl_seqs_train = []

words_shuffle = []
lbl_seqs_shuffle = []
prev_word = ''
# For each word and label sequence
for i in range(len(lbl_seqs)):
    word_curr = words[i]
    lbls_curr = lbl_seqs[i]
    # If it is the same as the previous word, add it to words_shuffle
    if word_curr == prev_word:
        words_shuffle.append(word_curr)
        lbl_seqs_shuffle.append(lbls_curr)
        prev_word = word_curr

        # Add the last batch
        if i == len(lbl_seqs) - 1:
            np.random.shuffle(words_shuffle)
            r = int(round(len(words_shuffle))*0.8)
            words_heldout.extend(list(words_shuffle)[0:r])
            words_train.extend(list(words_shuffle[r:len(words_shuffle)]))
            lbl_seqs_heldout.extend(list(lbl_seqs_shuffle)[0:r])
            lbl_seqs_train.extend(list(lbl_seqs_shuffle[r:len(words_shuffle)]))
    # Else if this word is different, shuffle the previous set of words, split them, and add to the dataset
    else:
        np.random.shuffle(words_shuffle)
        r = int(round(len(words_shuffle))*0.8) # Split the set of words so that 80% go to train and 20% to heldout
        words_heldout.extend(list(words_shuffle)[0:r])
        words_train.extend(list(words_shuffle[r:len(words_shuffle)]))
        lbl_seqs_heldout.extend(list(lbl_seqs_shuffle)[0:r])
        lbl_seqs_train.extend(list(lbl_seqs_shuffle[r:len(words_shuffle)]))

        words_shuffle = []
        lbl_seqs_shuffle = []
        words_shuffle.append(word_curr)
        lbl_seqs_shuffle.append(lbls_curr)
        prev_word = word_curr


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
# Set previous accuracy
prev_accuracy = 0.
# Set the continuing condition, which is that the accuracy improved
condition = True
# Iteration counter
p = 0

# TRAINING
# Train the hmm using EM algo until there is no improvement
while condition:
    p+=1
    # Create a dict of word hmms that can be reused across iterations
    # Automatically updates with new params from prev iter
    word_hmms = {}
    for word in set(words_train):
        whmm = word_hmm(letters=word, hmms=component_hmms)
        word_hmms.update({word:whmm})
    # Go through each of the 798 words and utterances alphabetically
    total_log_likelihood = []
    for i in range(len(words_train)):
        word = words_train[i]
        lbls = lbl_seqs_train[i]
        # Get word hmm
        whmm = word_hmms.get(word)
        # Do the forward and backward passes
        flog_prob, alphas = whmm.forward(lbl_seq=lbls)
        blog_prob, betas = whmm.backward(lbl_seq=lbls)
        print(flog_prob)
        print(blog_prob)
        # Collect counts for each letter and silence component
        whmm.collect_counts(lbl_seq=lbls, alphas=alphas, betas=betas, hmms=component_hmms)
        # CRITICAL: update the dictionary with this hmm
        word_hmms.update({word:whmm})

        total_log_likelihood.append(flog_prob)

    # At the end of each iteration, update parameters across all hmms
    for hmm in component_hmms.values():
        hmm.update_params()
        component_hmms.update({hmm.letter:hmm})

    total_log_likelihood = logsumexp(total_log_likelihood)
    log_probs_k_train.append(total_log_likelihood / len(lbl_seqs))

    accuracy = 0.

    # VALIDATION: test each sample in the heldout dataset on each word hmm
    # select the best one and calculate accuracy
    for i in range(len(lbl_seqs_heldout)):
        lbls = lbl_seqs_heldout[i]
        target_word = words_heldout[i]
        forward_log_probs = []
        for word in set(words):
            whmm = word_hmms.get(word)
            # Do the forward and backward passes
            flog_prob, alphas = whmm.forward(lbl_seq=lbls)
            forward_log_probs.append(flog_prob)
            print(flog_prob)
        # Select best word based on forward probability
        forward_log_probs = np.array(forward_log_probs)
        max_forward_log_prob = np.max(forward_log_probs)
        most_likely_word = words[np.argmax(forward_log_probs)]
        confidence = max_forward_log_prob - logsumexp(forward_log_probs)
        # Add one to accuracy if correct
        if target_word == most_likely_word:
            accuracy+=1

    accuracy/=len(words_heldout) 

    print("ACCURACY: ", accuracy) 

    # If this accuracy is not better than the previous, set condition to false
    # Record the number of iterations it took
    if accuracy <= prev_accuracy-0.02:
        n_star = p
        condition = False
        break
    else:
        prev_accuracy = accuracy

print(log_probs_k_train[-1])

# Plot of the average log-likelihood as a function of # iterations
plt.plot(log_probs_k_train, color='red', label='Train')
plt.xlabel('Iterations')
plt.ylabel('Average Log-Likelihood')
plt.title(f'Average Log-Likelihood of Training Data Over {args.iters} Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/avg_log_prob_k.png')

# TESTING AGAIN ON THE ENTIRE DATASET USING N_STAR
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

# Train the hmm using EM algo for n_star iterations
for p in range(n_star):
    word_hmms = {}
    for word in set(words):
        whmm = word_hmm(letters=word, hmms=component_hmms)
        word_hmms.update({word:whmm})
    total_log_likelihood = []
    for i in range(len(words)):
        word = words[i]
        lbls = lbl_seqs[i]
        whmm = word_hmms.get(word)
        flog_prob, alphas = whmm.forward(lbl_seq=lbls)
        blog_prob, betas = whmm.backward(lbl_seq=lbls)
        whmm.collect_counts(lbl_seq=lbls, alphas=alphas, betas=betas, hmms=component_hmms)
        word_hmms.update({word:whmm})
        total_log_likelihood.append(flog_prob)

    # At the end of each iteration, update parameters across all component hmms
    for hmm in component_hmms.values():
        hmm.update_params()
        component_hmms.update({hmm.letter:hmm})

    total_log_likelihood = logsumexp(total_log_likelihood)
    # Get the average total_log_likelihood
    log_probs_k_train.append(total_log_likelihood / len(lbl_seqs))

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

out_file = open("test_results_sec.txt", "w") 

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