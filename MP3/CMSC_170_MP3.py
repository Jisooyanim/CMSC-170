import numpy

class hiddenMarkovModel:
    def __init__(self, states, observables, transProb, emissionProb, initProb):
        self.states                 = numpy.array(states)
        self.observables            = numpy.array(observables)
        self.transProb              = numpy.array(transProb)
        self.emissionProb           = numpy.array(emissionProb)
        self.initProb               = numpy.array(initProb)
        self.numberOfStates         = self.states.shape[0]
        self.numberOfObservables    = self.observables.shape[0]
    
    def likelihood(self, observables):
        probability, _ = self.likelihoodForward(observables)
        return probability
    
    def likelihoodForward(self, observables):
        alpha = numpy.zeros((self.numberOfStates, len(observables)))
        alpha[:, 0] = self.initProb * self.emissionProb[:, self.observablesIndex(observables[0])]

        for i in range(1, len(observables)):
            obsLike = self.observablesIndex(observables[i])
            alpha[:, i] = alpha[:, i - 1].dot(self.transProb) * self.emissionProb[:, obsLike]
        
        probability = alpha[:, len(observables) - 1].sum()
        return probability, alpha
    
    def decoding(self, observables):
        delta = numpy.zeros((self.numberOfStates, len(observables)))
        delta[:, 0] = self.initProb * self.emissionProb[:, self.observablesIndex(observables[0])]

        for i in range(1, len(observables)):
            obsLike = self.observablesIndex(observables[i])
            prevDelta = delta[:, i - 1].reshape(-1, 1)
            delta[:, i] = (prevDelta * self.transProb).max(axis = 0) * self.emissionProb[:, obsLike]

        path = self.states[delta.argmax(axis = 0)]
        probability = delta[:, len(observables) - 1].max()

        return path, probability
        
    def observablesIndex(self, observables):
        return numpy.argwhere(self.observables == observables).flatten().item()


def main():
    states = ["rainy", "sunny"]
    observables = ["walk", "shop", "clean"]
    initProb = [0.6, 0.4]
    transProb = [[0.7, 0.3], [0.6, 0.4]]
    emissionProb = [[0.1, 0.4, 0.5], [0.6, 0.3,0.1]]

    HMM = hiddenMarkovModel(states, observables, transProb, emissionProb, initProb)

    #Holds the observation sequence
    observationSequence = ["clean", "clean", "clean"]


    # Problem 1
    likelihood = HMM.likelihood(observationSequence)
    print("Observation sequence: {}".format(observationSequence))
    print("Likelihood: {:.5f}".format(float(likelihood)))

    # Problem 2
    path, probability = HMM.decoding(observationSequence)
    print("Observation sequence: {}".format(observationSequence))
    print("Most likely hidden state path: {}".format(path))
    print("Likelihood for observation sequence along path: {:.5f}".format(float(probability)))


if __name__ == '__main__':
    main()