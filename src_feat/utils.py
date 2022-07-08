from ast import literal_eval
from pandas import read_csv
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, MWETokenizer


def post_pad_sequences(sequences, maxlen=None, start="<START>", end="<END>", pad="<PAD>", return_masks=True):
    L = max([len(x) for x in sequences])
    if maxlen is None or maxlen-2 > L:
        seq = [([start] + x + [end] + [pad]*(L-len(x))) for x in sequences]
        masks = [([1]*(len(x)+2) + [0]*(L-len(x))) for x in sequences] if return_masks else None
    else:
        seq = [([start] + (x[:maxlen-2] if len(x) >= maxlen-2 else x) + [end] + [pad]*(max(maxlen-len(x)-2, 0))) for x in sequences]
        masks = [([1]*(min(maxlen, len(x)+2)) + [0]*(max(maxlen-len(x)-2, 0))) for x in sequences] if return_masks else None
    return {"seq": seq, "mask": masks}
    
        
def get_encoded_input(fname, tag2idx=None, maxlen=256, vocab=None, tokenizer_name="tree_bank"):
    tokenizer = {
        "tree_bank": TreebankWordTokenizer(),
        "tweet": TweetTokenizer(),
        "mwe": MWETokenizer(),
    }[tokenizer_name]

    data = read_csv(fname, sep=" ", header=None, encoding="utf-8").values.tolist()

    inv_vocab = {v:k for (k, v) in enumerate(vocab)}

    text = [tokenizer.tokenize(' '.join(literal_eval(words))) for (words, _, _) in data]
    text = post_pad_sequences(text, maxlen=maxlen, return_masks=True)
    encoded_input = [[inv_vocab[w] for w in sent] for sent in text["seq"]]
    attention_masks = text["mask"]

    labels = [[lbl.split('-')[0] for lbl in literal_eval(labels)] for (_, labels, _) in data]
    labels = post_pad_sequences(labels, maxlen=maxlen, start='<', end='>', pad='$', return_masks=False)
    extended_labels = [[tag2idx[l] for l in lbls] for lbls in labels["seq"]]

    return encoded_input, attention_masks, extended_labels