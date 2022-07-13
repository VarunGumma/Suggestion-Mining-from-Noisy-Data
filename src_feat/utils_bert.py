from transformers import RobertaTokenizerFast, BertTokenizerFast
import pandas as pd

# def _extend_labels(L, text, labels, tokenizer):
#     X = 5
#     extended_labels = [X]
#     for (t, l) in zip(text, labels):
#         n = len(tokenizer.tokenize(t))
#         extended_labels.extend([l] * n)
#     rem = L - len(extended_labels)
#     extended_labels.extend([X] * rem)
#     return extended_labels


# def _extend_labels(L, text, labels, tokenizer, tag2idx):
#     extended_labels = [tag2idx['<']]
#     for (t, l) in zip(text, labels):
#         n = len(tokenizer.tokenize(t))
#         extended_labels.extend([l] * n)
#     extended_labels.append(tag2idx['>'])
#     rem = L - len(extended_labels)
#     extended_labels.extend([tag2idx['$']] * rem)
#     return extended_labels

# '<' - START, '>' - END, '$' - PAD
def get_encoded_input(fname, tag2idx, tokenizer_name="roberta-base"):

    def _extend_labels(txt, lbls):
        extended_labels = [tag2idx['<']]
        for (t, l) in zip(txt, lbls):
            n = len(tokenizer.tokenize(t))
            extended_labels.extend([l] * n)
        extended_labels.append(tag2idx['>'])
        rem = max_len - len(extended_labels)
        extended_labels.extend([tag2idx['$']] * rem)
        return extended_labels

    tokenizer = {
        "roberta-base": RobertaTokenizerFast.from_pretrained("roberta-base", do_lower_case=True, add_prefix_space=True),
        "bert-base-uncased": BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True),
    }[tokenizer_name]

    data = pd.read_csv(fname,
                       sep=' ',
                       header=None,
                       names=['a', 'b', 'c'],
                       encoding="utf-8",
                       converters={'a': pd.eval, 
                                   'b': pd.eval})

    txt = data['a'].tolist()
    labels = [[tag2idx[l.split('-')[0]] for l in labels] for labels in data['b']]

    ext_txt = tokenizer(txt, 
                         padding="longest",
                         return_attention_mask=True,
                         is_split_into_words=True,
                         return_length=True)

    max_len = ext_text["length"][0]
    ext_labels = [_extend_labels(txt, lbl) for (txt, lbl) in zip(text, labels)]
    return ext_text, ext_labels
