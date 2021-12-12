

class Dictionary:

    def __init__(self, path=None):
        self.tokens_2_ids = {}
        self.ids_2_tokens = []
        if path is not None:
            self.read(path)


    def __len__(self):
        return len(self.ids_2_tokens)


    def get_id(self, token, safe=False):
        if safe:
            return self.tokens_2_ids.get(token, "<UNK>")
        else:
            return self.tokens_2_ids[token]


    def get_ids(self, tokens, safe=False):
        return [self.get_id(token, safe) for token in tokens]


    def get_token(self, idx, safe=False):
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
        with open(path) as f:
            self.ids_2_tokens = [
                line.strip() 
                for line in f 
                if line.strip() != ""
            ]
        self.tokens_2_ids = {
            token : idx
            for idx, token 
            in enumerate(self.ids_2_tokens)
        }


        
