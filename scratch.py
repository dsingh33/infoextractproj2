import numpy as np
from scipy.special import logsumexp

a = np.array([2,3,4,5])
b = np.array([3,4,5,6])
update = np.vstack((a,b))
print(update.shape)
print(update)
print(logsumexp(update, axis=0))

# Get lists of words and utterance labels
# Plot of the final iteration's emission probabilities for each state
# Uncomment the commented lines if using 4 states
plt.plot(emission_probs_final_s0, color='red', label='State 0 Emissions')
plt.plot(emission_probs_final_s1, color='blue', label='State 1 Emissions')
#plt.plot(emission_probs_final_s2, color='green', label='State 2 Emissions')
#plt.plot(emission_probs_final_s3, color='purple', label='State 3 Emissions')
plt.xlabel('Letter ID')
plt.xticks(vocab, size='small')
plt.ylabel('Probabilities')
plt.title('Final Emission Probabilities For Alphabet')
plt.legend()
plt.show()
plt.savefig('../figs/final_emission_probs.png')


# Plots of the emission probabilities of letters 'a' and 'n' as a function of # iterations
plt.plot(emission_probs_a_s0, color='red', label='Given State 0')
plt.plot(emission_probs_a_s1, color='blue', label='Given State 1')
#plt.plot(emission_probs_a_s2, color='green', label='Given State 2')
#plt.plot(emission_probs_a_s3, color='purple', label='Given State 3')
plt.xlabel('Iterations')
plt.ylabel('Probabilities')
plt.title('Emission Probabilities of "a" Over 600 Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/a_emission_probs.png')

plt.plot(emission_probs_n_s0, color='red', label='Given State 0')
plt.plot(emission_probs_n_s1, color='blue', label='Given State 1')
#plt.plot(emission_probs_n_s2, color='green', label='Given State 2')
#plt.plot(emission_probs_n_s3, color='purple', label='Given State 3')
plt.xlabel('Iterations')
plt.ylabel('Probabilities')
plt.title('Emission Probabilities of "n" Over 600 Iterations')
plt.legend()
plt.show()
plt.savefig('../figs/n_emission_probs.png')