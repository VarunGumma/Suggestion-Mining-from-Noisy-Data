import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, MWETokenizer


def post_pad_sequences(sequences, maxlen=None, start="<START>", end="<END>", pad="<PAD>", return_masks=True):
    L = max([len(x) for x in sequences])
    if maxlen is None or maxlen-2 > L:
        sequences = [([start] + x + [end] + [pad]*(L-len(x))) for x in sequences]
        masks = [[(0 if x == pad else 1) for x in s] for s in sequences] if return_masks else None
    else:
        sequences = [([start] + (x[:maxlen-2] if len(x) >= maxlen-2 else x) + [end] + [pad]*(max(maxlen-len(x)-2, 0))) for x in sequences]
        masks = [[(0 if x == pad else 1) for x in s] for s in sequences] if return_masks else None
    return {"seq": sequences, "mask": masks}
    
        
def get_encoded_input(fname, tag2idx=None, maxlen=256, vocab=None, tokenizer_name="tree_bank"):
    tokenizer = {
        "tree_bank": TreebankWordTokenizer(),
        "tweet": TweetTokenizer(),
        "mwe": MWETokenizer(),
    }[tokenizer_name]

    inv_vocab = {v:k for (k, v) in enumerate(vocab)}

    data = pd.read_csv(fname,
                       sep=' ',
                       header=None,
                       names=['a', 'b', 'c'],
                       encoding="utf-8",
                       converters={'a': pd.eval, 
                                   'b': pd.eval})

    text = [tokenizer.tokenize(' '.join(words)) for words in data['a']]
    text = post_pad_sequences(text, maxlen=maxlen, return_masks=True)
    encoded_input = [[inv_vocab[w] for w in sent] for sent in text["seq"]]

    labels = [[l.split('-')[0] for l in labels] for labels in data['b']]
    labels = post_pad_sequences(labels, maxlen=maxlen, start='<', end='>', pad='$', return_masks=False)["seq"]
    extended_labels = [[tag2idx[l] for l in lbls] for lbls in labels]

    return encoded_input, text["mask"], extended_labels