from transformers import RobertaTokenizerFast, BertTokenizerFast
from ast import literal_eval
from pandas import read_csv

def _extend_labels(L, text, labels, tokenizer):
    # X := 5
    extended_labels = [5]
    for (t, l) in zip(text, labels):
        n = len(tokenizer.tokenize(t))
        extended_labels.extend([l] * n)
    rem = L - len(extended_labels)
    extended_labels.extend([5] * rem)
    return extended_labels


def get_encoded_input(fname, tag2idx, tokenizer_name="roberta-base"):
    data = read_csv(fname, sep=' ', header=None).values.tolist()
    tokenizer = {
        "roberta-base": RobertaTokenizerFast.from_pretrained("roberta-base", do_lower_case=True, add_prefix_space=True),
        "bert-base-uncased": BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True),
    }[tokenizer_name]

    text = [literal_eval(words) for (words, _, _) in data]
    labels = [[tag2idx[l.split('-')[0]] for l in literal_eval(labels)] for (_, labels, _) in data]
    ext_text = tokenizer(text, 
                         padding="longest",
                         return_attention_mask=True,
                         is_split_into_words=True,
                         return_length=True)

    max_len = ext_text["length"][0]
    ext_labels = [[l for l in _extend_labels(max_len, txt, lbl, tokenizer)] for (txt, lbl) in zip(text, labels)]
    return ext_text, ext_labels
