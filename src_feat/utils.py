from ast import literal_eval
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, MWETokenizer

def post_pad_sequences(sequences, max_len=None, pad_value=0):
    longest = max([len(x) for x in sequences])
    if max_len is not None and max_len > longest:
        print("Unnecessary extra padding detected!\nSequences will be only be padded to the longest length")
    sequences = [(x + [pad_value]*(longest-len(x))) for x in sequences]
    return sequences if max_len is None or max_len >= longest else [x[:max_len] for x in sequences]
    
        
def get_encoded_input(fname, tag2idx=None, max_len=256, vocab=None, tokenizer_name="tree_bank"):
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
    text = [tokenizer.tokenize(' '.join(literal_eval(words))) for (words, _, _) in data]
    labels = [[tag2idx[l.split('-')[0]] for l in literal_eval(labels)] for (_, labels, _) in data]

    text_padded = post_pad_sequences([[inv_vocab[w] for w in sent] for sent in text],
                                     max_len=max_len,
                                     pad_value=inv_vocab["<PAD>"])

    L = len(text_padded[0])

    attention_masks = [([1]*len(t) + [0]*(L - len(t)))[:max_len] for t in text]
    labels = [(l + [X]*(L - len(l)))[:max_len] for l in labels]
    return text_padded, attention_masks, labels