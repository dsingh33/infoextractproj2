# infoextractproj2

This is the directory for Information Extraction Project 2.
There are three folders: code, data, and figs.

Data contains all of the files given for the assignment, unchanged.

Code contains five files: let_hmm.py define the letter hmm class, sil_hmm.py defines the silence hmm class, and word_hmm.py defines the word hmm class.
primary.py is the file that calls and runs all of these classes and their methods.
secondary.py also runs these methods, for the constrastive system using held out data.

The figs folder has the following figures:
avg_tot_log_lik_10_iters_primary.png is the log likelihood plot for the primary system.
avg_tot_log_lik_10_iters_secondary_kept.png is the log likelihood plot for the secondary system on the kept data.
avg_tot_log_lik_10_iters_secondary_total.png is the same on the total data, which consists of the kept plus the held out data.

In addition there are three more files:
IE Project 2 Report is the pdf with a dicussion of this project and debugging efforts.

test_results.txt has 393 lines with the predicted word and confidence score for the primary system on the test data.
test_results_sec.txt is the same for the secondary system.
