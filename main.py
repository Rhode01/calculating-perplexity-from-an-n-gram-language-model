import pickle
BOS = '<BOS>'
EOS = '<EOS>'
OOV = '<OOV>'
class NGramLM:
    def __init__(self, path, smoothing=0.001, verbose=False):
        with open(path, 'rb') as fin:
            data = pickle.load(fin)
        self.n = data['n']
        self.V = set(data['V'])
        self.model = data['model']
        self.smoothing = smoothing
        self.verbose = verbose

    def get_prob(self, context, token):
        context = tuple(context[-self.n+1:])
        # Add <BOS> tokens if the context is too short, i.e., it's at the start of the sequence
        while len(context) < (self.n-1):
            context = (BOS,) + context
        # Handle words that were not encountered during the training by replacing them with a special <OOV> token
        context = tuple((c if c in self.V else OOV) for c in context)
        if token not in self.V:
            token = OOV
        if context in self.model:
            count = self.model[context].get(token, 0)
            prob = (count + self.smoothing) / (sum(self.model[context].values()) + self.smoothing * len(self.V))
        else:
            prob = 1 / len(self.V)
        if self.verbose:
            print(f'{prob:.4n}', *context, '->', token)
        return prob