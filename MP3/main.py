from HMM import hiddenMarkovModel

states = ["rainy", "sunny"]
observables = ["walk", "shop", "clean"]
initProb = [0.6, 0.4]
transProb = [[0.7, 0.3], [0.6, 0.4]]
emissionProb = [[0.1, 0.4, 0.5], [0.6, 0.3,0.1]]

HMM = hiddenMarkovModel(states, observables, transProb, emissionProb, initProb)

observation_sequence = ["shop","shop", "walk"]
likelihood = HMM.likelihood(observation_sequence)
print("* Observation sequence: {}".format(observation_sequence))
print("* Likelihood: {:.2f}".format(float(likelihood)))