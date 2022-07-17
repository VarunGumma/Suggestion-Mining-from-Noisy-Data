import spacy
import pandas as pd
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer, MWETokenizer

def post_pad_sequences(sequences, 
                       maxlen=None,
                       start="<START>",
                       end="<END>",
                       pad="<PAD>",
                       return_masks=True):

    L = max([len(x) for x in sequences])
    if maxlen is None or maxlen-2 > L:
        sequences = [([start] + x + [end] + [pad]*(L-len(x))) for x in sequences]
        masks = [[(0 if x == pad else 1) for x in s] for s in sequences] if return_masks else None
    else:
        sequences = [([start] + (x[:maxlen-2] if len(x) >= maxlen-2 else x) + [end] + [pad]*(max(maxlen-len(x)-2, 0))) for x in sequences]
        masks = [[(0 if x == pad else 1) for x in s] for s in sequences] if return_masks else None
    return {"seq": sequences, "mask": masks}
    
        
def get_encoded_input(fname, 
                      tag2idx=None, 
                      maxlen=256, 
                      vocab2idx=None, 
                      pos_tags2idx=None, 
                      tokenizer_name="tree_bank", 
                      return_pos_tags=False):
    tokenizer = {
        "tree_bank": TreebankWordTokenizer(),
        "tweet": TweetTokenizer(),
        "mwe": MWETokenizer(),
    }[tokenizer_name]

    data = pd.read_csv(fname,
                       sep=' ',
                       header=None,
                       names=['a', 'b', 'c'],
                       encoding="utf-8",
                       converters={'a': pd.eval, 
                                   'b': pd.eval})

    text = [tokenizer.tokenize(' '.join(words)) for words in data['a']]
    text = post_pad_sequences(text, maxlen=maxlen, return_masks=True)
    encoded_input = [[vocab2idx[w] for w in sent] for sent in text["seq"]]
    
    labels = [[l.split('-')[0] for l in labels] for labels in data['b']]
    labels = post_pad_sequences(labels,
                                start='<', 
                                end='>', 
                                pad='$', 
                                maxlen=maxlen, 
                                return_masks=False)["seq"]
    extended_labels = [[tag2idx[l] for l in lbls] for lbls in labels]

    
    if return_pos_tags and pos_tags2idx is not None:
        nlp = spacy.load("en_core_web_sm")
        pos_tags = [[pos_tags2idx[token.pos_] for token in nlp(' '.join(s))] for s in data['a']]
        pos_tags = post_pad_sequences(pos_tags, 
                                      maxlen=maxlen, 
                                      start=pos_tags2idx['X'], 
                                      end=pos_tags2idx['X'], 
                                      pad=pos_tags2idx['X'], 
                                      return_masks=False)["seq"]
        return encoded_input, pos_tags, text["mask"], extended_labels
    else:
        return encoded_input, text["mask"], extended_labels