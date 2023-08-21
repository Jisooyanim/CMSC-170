from hmm import HiddenMarkovModel


states = ["rainy", "sunny"]
observables = ["walk", "shop", "clean"]
initProb = [0.6, 0.4]
transProb = [[0.7, 0.3], [0.6, 0.4]]
emissionProb = [[0.1, 0.4, 0.5], [0.6, 0.3,0.1]]

HMM = HiddenMarkovModel(states, observables, transProb, emissionProb, initProb)

observation_sequence = ["walk", "shop", "clean", "shop", "shop", "walk", "shop"]



# likelihood = HMM.likelihood(observation_sequence)
# print("* Observation sequence: {}".format(observation_sequence))
# print("* Likelihood: {:.5f}".format(float(likelihood)))

# path, probability = HMM.decode(observation_sequence)
# print("* Observation sequence: {}".format(observation_sequence))
# print("* Most likely hidden state path: {}".format(path))
# print("* Likelihood for observation sequence along path: {:.5f}".format(float(probability)))


HMM.learn(observation_sequence, iterations=10)
print("PROBABILITIES AFTER LEARNING")
print("----------------------------")
print("* Initial:")
print(HMM.pi)
print("\n* Transition:")
print(HMM.tp)
print("\n* Emission:")
print(HMM.ep)