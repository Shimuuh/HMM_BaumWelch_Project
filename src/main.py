from hmm import HiddenMarkovModel
from baum_welch import baum_welch
import numpy as np

# Example HMM (2 states, 2 observation symbols)
A = [[0.7, 0.3],
     [0.4, 0.6]]

B = [[0.5, 0.5],
     [0.1, 0.9]]

pi = [0.6, 0.4]

observations = [0, 1, 0, 1, 1]

model = HiddenMarkovModel(A, B, pi)

trained_model = baum_welch(model, observations)

print("Updated Transition Matrix:")
print(trained_model.A)

print("\nUpdated Emission Matrix:")
print(trained_model.B)