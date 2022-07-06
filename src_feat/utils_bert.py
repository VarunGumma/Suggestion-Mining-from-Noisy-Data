from ast import literal_eval
import pandas as pd
from tensorflow.keras.utils import pad_sequences
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, MWETokenizer


def get_encoded_input(fname, tag2idx=None, max_len=256, vocab=None, visualize=False, tokenizer_name="tree_bank"):
    B, I, O, E, S, X = range(6)

    tokenizer = {
        "tree_bank": TreebankWordTokenizer(),
        "tweet": TweetTokenizer(),
        "mwe": MWETokenizer(),
    }[tokenizer_name]

    data = pd.read_csv(fname, 
                       sep=" ", 
                       header=None, 
                       encoding="utf-8").values.tolist()

    inv_vocab = {v:k for (k, v) in enumerate(vocab)}
    text = [' '.join(literal_eval(words)) for (words, _, _) in data]
    text = [['<S>'] + tokenizer.tokenize(s) + ['</S>'] for s in text]

    if visualize:
        df = pd.DataFrame([len(s) for s in text], columns=["seq_len"])
        print(df["seq_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]))

    text_padded = pad_sequences([[inv_vocab[w] for w in sent] for sent in text], 
                                maxlen=max_len,
                                padding="post", 
                                truncating="post",
                                value=inv_vocab["<PAD>"])

    labels = [[tag2idx[l.split('-')[0]] for l in literal_eval(labels)] for (_, labels, _) in data]
    attention_masks = [([1]*len(t) + [0]*(text_padded.shape[1] - len(t)))[:max_len] for t in text]
    labels = [([X] + l + [X]*(text_padded.shape[1] - len(l) - 1))[:max_len] for l in labels]
    return text_padded, attention_masks, labels