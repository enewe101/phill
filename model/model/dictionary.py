import pdb

class Dictionary:

    def __init__(self, path=None, vocab_limit=None):
        self.vocab_limit = vocab_limit
        self.tokens_2_ids = {}
        self.ids_2_tokens = []
        self.counts = []
        if path is not None:
            self.read(path)


    def __len__(self):
        length = len(self.ids_2_tokens)
        if self.vocab_limit is not None and self.vocab_limit < length:
            length = self.vocab_limit
        return length


    def get_id(self, token, safe=False):
        if safe:
            idx = self.tokens_2_ids.get(token, "<UNK>")
        else:
            idx = self.tokens_2_ids[token]
        if idx >= self.vocab_limit:
            idx = self.tokens_2_ids["<UNK>"]
        return idx


    def get_ids(self, tokens, safe=False):
        return [self.get_id(token, safe) for token in tokens]


    def get_token(self, idx, safe=False):
        if self.vocab_limit is not None and idx >= self.vocab_limit:
            return "<UNK>"
        try:
            return self.ids_2_tokens[idx]
        except:
            if not safe: 
                raise
            else:
                return "<UNK>"


    def get_tokens(self, idxs, safe=False):
        return [self.get_token(idx, safe) for idx in idxs]


    def read(self, path):

        self.ids_2_tokens = []
        self.counts = []

        with open(path) as f:
            for line in f:
                if line.strip() == "":
                    continue
                try:
                    token, count = line.split('\t')
                except:
                    pdb.set_trace()
                idx = len(self.ids_2_tokens)
                self.ids_2_tokens.append(token)
                self.counts.append(int(count))

        self.tokens_2_ids = {
            token : idx
            for idx, token 
            in enumerate(self.ids_2_tokens)
        }


        
