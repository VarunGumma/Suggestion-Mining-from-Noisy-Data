import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, MWETokenizer

def post_pad_sequences(sequences, 
                       maxlen=None,
                       start="<START>",
                       end="<END>",
                       pad="<PAD>"):

    L = max([len(x) for x in sequences])
    if maxlen is None or maxlen-2 > L:
        sequences = [([start] + x + [end] + [pad]*(L-len(x))) for x in sequences]
    else:
        sequences = [([start] + (x[:maxlen-2] if len(x) >= maxlen-2 else x) + [end] + [pad]*(max(maxlen-len(x)-2, 0))) for x in sequences]
    return sequences
    
        
def get_encoded_input(fname,
                      maxlen=256, 
                      vocab2idx=None,
                      tokenizer_name="tree_bank"):
    tokenizer = {
        "tree_bank": TreebankWordTokenizer(),
        "tweet": TweetTokenizer(),
        "mwe": MWETokenizer(),
    }[tokenizer_name]

    df = pd.read_csv(fname, names=["sno", "id", "text", "lbl"])
    text = [x.lower().strip() for x in df["text"]]
    text = [tokenizer.tokenize(s) for s in text]
    text = post_pad_sequences(text, maxlen=maxlen)
    encoded_input = [[vocab2idx[w] for w in sent] for sent in text]
    return encoded_input, df["lbl"].to_numpy()