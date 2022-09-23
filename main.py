# An implementation based on the ConLSTM-CRF model described in:
# Olafsson, S., Wallace, B. C., & Bickmore, T. W. (2020, May). Towards a Computational Framework for Automating Substance Use Counseling with Virtual Agents. In AAMAS (pp. 966-974).

import pandas as pd
import numpy as np
import data_prep as prep
import con_lstm_crf as m
import torch
import os

train_data = pd.read_csv("training_data.csv", encoding="latin1")
train_data = train_data.fillna("")
print(len(train_data))

test_data = pd.read_csv("test_data.csv", encoding="latin1")
test_data = test_data.fillna("")
print(len(test_data))

START_TAG = "<START>"
STOP_TAG = "<STOP>"

tag_to_ix, word_to_ix = prep.make_dicts(
    "training_data.csv", 
    "test_data.csv",
    START_TAG, STOP_TAG)

print(tag_to_ix)
ix2tag = {v:k for k,v in tag_to_ix.items()}

cfg = {
    "emb": 50,
    "hid": 64,
    "lay": 1,
    "bat": 1,
    "fol": 10,
    "ler": 0.001,
    "dec": 0,
    "grp": "Section",
    "pat": 2
}

wv_model = prep.train_word2vec(train_data, test_data, cfg)
print(len(word_to_ix))
print(wv_model)

# Make pretrained word embeddings:
embeddings = np.zeros((len(word_to_ix), cfg['emb']))
for k, v in word_to_ix.items():
    embeddings[v] = wv_model.wv[k]
pre_emb = torch.from_numpy(embeddings).float()
print("Pre emb word embs:")
print(pre_emb.size())

agg_func = lambda s: [(tag, txt, role) for tag, txt, role in zip(
    s["Tag"].values.tolist(),
    s["Text"].values.tolist(),
    s["Speaker"].values.tolist())]

train_grouped = train_data.groupby(cfg["grp"]).apply(agg_func)
train_seqs = [s for s in train_grouped]

test_grouped = test_data.groupby(cfg["grp"]).apply(agg_func)
test_seqs = [s for s in test_grouped]

outfile = "con_lstm_crf"

tagger_model = m.train_conlstm_crf(
    train_seqs, 
    test_seqs, 
    cfg, 
    word_to_ix, 
    tag_to_ix,
    pre_emb, 
    START_TAG, 
    STOP_TAG,
    outfile)

# Save the final version of the model.
# This will be a different version than the one stored based on the best macro F1,
# which is stored during runtime.
runs_path = os.path.join(os.getcwd(), "runs") 
runs_path = os.path.join(runs_path, outfile)
torch.save(tagger_model.state_dict(), os.path.join(runs_path, outfile + "_final"))

# Load a model and evaluate:
torch.manual_seed(1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Evaluating on " + str(DEVICE))
model = m.ConLSTM_CRF(len(word_to_ix), tag_to_ix, cfg["emb"], cfg["hid"], cfg["lay"], DEVICE, pre_emb, START_TAG, STOP_TAG)
model.to(DEVICE)
model.load_state_dict(torch.load("path/to/state/dict"))
true_ys, pred_ys = m.evaluate(model, test_seqs, word_to_ix, tag_to_ix, DEVICE)
