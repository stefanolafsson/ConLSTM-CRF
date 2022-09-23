# An implementation based on the ConLSTM-CRF model described in:
# Olafsson, S., Wallace, B. C., & Bickmore, T. W. (2020, May). Towards a Computational Framework for Automating Substance Use Counseling with Virtual Agents. In AAMAS (pp. 966-974).

import numpy as np
import time
import datetime
import math
import random
import os
import json

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn_crfsuite import metrics as crf_metrics

import pandas as pd

from gensim.models import Word2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim

import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
	
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def prepare_seq(seq, word_to_ix, device):
	X = []
	for data in seq:
		text = data[1].strip()
		int_idxs = [word_to_ix[w] for w in text.split(" ") if w != ""]
		int_i = torch.tensor(int_idxs, dtype=torch.long).to(device)
		X.append((int_i, data[2]))
	return X

def prepare_sent_seq(data, to_ix, device):
	text = data[0].strip()
	idxs = [to_ix[w] for w in text.split(" ") if w != ""]
	return torch.tensor(idxs, dtype=torch.long).to(device)

def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return idx.item()

def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + \
		torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class ConLSTM_CRF(nn.Module):

	def __init__(
		self, 
		vocab_size, 
		tag_to_ix, 
		embedding_dim, 
		hidden_dim, 
		layers, 
		device, 
		pre_emb,
		start_tag, stop_tag):
		
		super(ConLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.tag_to_ix = tag_to_ix
		self.tagset_size = len(tag_to_ix)
		self.device = device
		self.start_tag = start_tag
		self.stop_tag = stop_tag
		
		# Word Embeddings and Word LSTM:
		self.speakerA_embeds = nn.Embedding(pre_emb.size(0), pre_emb.size(1))
		self.speakerA_embeds.weight = nn.Parameter(pre_emb)
		self.speakerB_embeds = nn.Embedding(pre_emb.size(0), pre_emb.size(1))
		self.speakerB_embeds.weight = nn.Parameter(pre_emb)
		self.speakerA_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers).to(device)
		self.speakerA_hidden = self.init_hidden()
		self.speakerB_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=layers).to(device)
		self.speakerB_hidden = self.init_hidden()
		
		# Semantic Context LSTM:
		self.context_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=layers).to(device)
		self.context_hidden = self.init_hidden()
		
		# Maps the output of the Semantic Context LSTM into tag space:
		self.context_hidden2tag = nn.Linear(hidden_dim, self.tagset_size).to(device)
		
		# Matrix of transition parameters.  Entry i,j is the score of
		# transitioning *to* i *from* j.
		self.transitions = nn.Parameter(
			torch.randn(self.tagset_size, self.tagset_size))

		# These two statements enforce the constraint that we never transfer
		# to the start tag and we never transfer from the stop tag
		self.transitions.data[tag_to_ix[self.start_tag], :] = -10000
		self.transitions.data[:, tag_to_ix[self.stop_tag]] = -10000		

	def init_hidden(self):
		return (torch.randn(1, 1, self.hidden_dim).to(self.device),
				torch.randn(1, 1, self.hidden_dim).to(self.device))

	def _forward_alg(self, feats):
		# Do the forward algorithm to compute the partition function
		init_alphas = torch.full((1, self.tagset_size), -10000.)
		# START_TAG has all of the score.
		init_alphas[0][self.tag_to_ix[self.start_tag]] = 0.

		# Wrap in a variable so that we will get automatic backprop
		forward_var = init_alphas
		# Iterate through the session
		for feat in feats:
			alphas_t = []  # The forward tensors at this timestep
			for next_tag in range(self.tagset_size):
				# broadcast the emission score: it is the same regardless of
				# the previous tag
				emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
				# the ith entry of trans_score is the score of transitioning to
				# next_tag from i
				trans_score = self.transitions[next_tag].view(1, -1)
				# The ith entry of next_tag_var is the value for the
				# edge (i -> next_tag) before we do log-sum-exp
				next_tag_var = forward_var.to(self.device) + trans_score + emit_score
				# The forward variable for this tag is log-sum-exp of all the
				# scores.
				alphas_t.append(log_sum_exp(next_tag_var).view(1))
			forward_var = torch.cat(alphas_t).view(1, -1)
			last_feat = feat
		terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
		alpha = log_sum_exp(terminal_var).to(self.device)
		return alpha
	
	def _score_session(self, feats, tags):
		# Gives the score of a provided tag sequence
		score = torch.zeros(1).to(self.device)
		tags = torch.cat([torch.tensor([self.tag_to_ix[self.start_tag]], dtype=torch.long).to(self.device), tags])
		for i, feat in enumerate(feats):
			score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
		score = score + self.transitions[self.tag_to_ix[self.stop_tag], tags[-1]]
		return score

	def _viterbi_decode(self, feats):
		backpointers = []

		# Initialize the viterbi variables in log space
		init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
		init_vvars[0][self.tag_to_ix[self.start_tag]] = 0

		# forward_var at step i holds the viterbi variables for step i-1
		forward_var = init_vvars
		for feat in feats:
			bptrs_t = []  # holds the backpointers for this step
			viterbivars_t = []  # holds the viterbi variables for this step

			for next_tag in range(self.tagset_size):
				# next_tag_var[i] holds the viterbi variable for tag i at the
				# previous step, plus the score of transitioning
				# from tag i to next_tag.
				# We don't include the emission scores here because the max
				# does not depend on them (we add them in below)
				next_tag_var = forward_var + self.transitions[next_tag]
				best_tag_id = argmax(next_tag_var)
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			# Now add in the emission scores, and assign forward_var to the set
			# of viterbi variables we just computed
			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)

		# Transition to STOP_TAG
		terminal_var = forward_var + self.transitions[self.tag_to_ix[self.stop_tag]]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]
		
		# Follow the back pointers to decode the best path.
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		# Pop off the start tag (we dont want to return that to the caller)
		start = best_path.pop()
		assert start == self.tag_to_ix[self.start_tag]  # Sanity check
		best_path.reverse()
		return path_score, best_path, best_tag_id
	
	def _get_lstm_word_features(self, sentence, speaker):
		hidden_list = []
		if speaker == "A":
			self.speakerA_hidden = self.init_hidden()
			speakerA_emb = self.speakerA_embeds(sentence).view(len(sentence), 1, -1)
			speakerA_lstm_out, self.speakerA_hidden = self.speakerA_lstm(speakerA_emb, self.speakerA_hidden)
			hidden_list.append(self.speakerA_hidden[0][0].view(1,1,-1))
		elif speaker == "B":
			self.speakerB_hidden = self.init_hidden()
			speakerB_emb = self.speakerB_embeds(sentence).view(len(sentence), 1, -1)
			speakerB_lstm_out, self.speakerB_hidden = self.speakerB_lstm(speakerB_emb, self.speakerB_hidden)
			hidden_list.append(self.speakerB_hidden[0][0].view(1,1,-1))
		return hidden_list
	
	def _get_lstm_seq_feats(self, seq):
		feat_list = []
		self.context_hidden = self.init_hidden()
		for data in seq:
			# Word/semantic features:
			word_hidden_list = self._get_lstm_word_features(data[0], data[1])
			for w_hid in word_hidden_list:
				context_lstm_out, self.context_hidden = self.context_lstm(w_hid, self.context_hidden)
			
			lstm_feats = self.context_hidden2tag(context_lstm_out.view(-1))
			
			feat_list.append(lstm_feats)
		return feat_list
	
	def neg_log_likelihood(self, seq, tags):
		feats = self._get_lstm_seq_feats(seq)
		# Calculate score, given the features:
		forward_score = self._forward_alg(feats)
		gold_score = self._score_session(feats, tags)
		return forward_score - gold_score
	
	def forward_sess(self, seq):
		feats = self._get_lstm_seq_feats(seq)
		# Decode sequence, given the features, using Viterbi:
		score, tag_seq, best_tag = self._viterbi_decode(feats)
		return tag_seq
	
	def forward(self, data):  # dont confuse this with _forward_alg above.
		# Get the emission scores from the LSTMs:
		w_hid = self._get_lstm_word_features(data[0], data[1]).view(1,1,-1)
		context_lstm_out, self.context_hidden = self.context_lstm(w_hid, self.context_hidden)
		lstm_feats = self.context_hidden2tag(context_lstm_out.view(-1))
		#print(lstm_feats)
		# Find the best path, given the features:
		score, tag_seq, best_tag = self._viterbi_decode(lstm_feats)
		return tag_seq


def train_conlstm_crf(train_seqs, test_seqs, cfg, word_to_ix, tag_to_ix, pre_emb, start_tag, stop_tag, outfile):
	
	torch.manual_seed(1)
	DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	print("Training on " + str(DEVICE))
	
	kf = KFold(n_splits=cfg["fol"], shuffle=True, random_state=321)
	train_test_idxs = kf.split(train_seqs)
	
	fold_count = 0
	start_time = time.time()
	
	best_macro_f1 = 0
	macrof1_dec_count = 0
	val_loss_inc_count = 0
	patience = cfg["pat"]
	cfgs_tested = []
	out_of_patience = False
	
	model = ConLSTM_CRF(len(word_to_ix), tag_to_ix, cfg["emb"], cfg["hid"], cfg["lay"], DEVICE, pre_emb, start_tag, stop_tag)
	model.to(DEVICE)
	optimizer = optim.Adam(model.parameters(), lr=cfg["ler"], weight_decay=cfg["dec"])
	
	ave_train_loss_list = []
	ave_val_loss_list = []
	train_acc_list = []
	ave_train_acc_list = []
	val_acc_list = []
	ave_train_acc_list = []
	
	while not out_of_patience:
		fold_count += 1
		print("\n--------\nStarting Epoch " + str(fold_count) + " with config:")
		print(cfg)
		
		fold_time = time.time()
			
		epoch_counter = 0
		
		random.shuffle(train_seqs)
		random.shuffle(test_seqs)
		train_val_index = math.floor(len(train_seqs)*0.9)
		
		# Stop this loop when best macro f1 decreases over a pre-determined number of epochs (patience).
		# or validation loss has increased the same amount as the patience.
		
		epoch_counter += 1
		
		if val_loss_inc_count == patience:
			out_of_patience = True
			val_loss_inc_count = 0
			print("Stopping criterion met. Breaking out of epoch loop.")
			break
		train_loss_list = []
		model.train()
		train_true_ys = []
		train_pred_ys = []
		train_count = 0
		for train_idx, train_d in enumerate(train_seqs[:train_val_index]):

			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			# Step 2. Get our inputs ready for the network, that is,
			# turn them into Tensors of word indices.
			train_seqs_in = prepare_seq(train_seqs[train_idx], word_to_ix, DEVICE)
			targets = torch.tensor([tag_to_ix[data[0]] for data in train_seqs[train_idx]], dtype=torch.long).to(DEVICE)

			# Step 3. Run our forward pass.
			loss = model.neg_log_likelihood(train_seqs_in, targets)
			train_loss_list.append(loss.item())

			# Step 4. Compute the loss, gradients, and update the parameters by
			# calling optimizer.step()
			loss.backward()
			optimizer.step()

			train_true_y = [tag_to_ix[data[0]] for data in train_seqs[train_idx]]
			train_pred_y = model.forward_sess(train_seqs_in)
			train_true_ys.append(train_true_y)
			train_pred_ys.append(train_pred_y)

			train_count += 1
			print("Sequence: %d" %(train_count) + " of %d" %(len(train_seqs[:train_val_index])) 
				  + "  Elapsed Fold Time: %.3f" %((time.time()-fold_time)/60), end="\r")

		train_acc = crf_metrics.flat_accuracy_score(train_true_ys, train_pred_ys)
		print("\nTraining accuracy: %.2f" %(train_acc))
		train_acc_list.append(train_acc)

		ave_train_loss_list.append(sum(train_loss_list)/len(train_loss_list))

		print("\nValidation ...")
		val_loss_list = []
		model.eval()
		with torch.no_grad():
			val_true_ys = []
			val_pred_ys = []
			val_count = 0
			for val_idx, val_d in enumerate(train_seqs[train_val_index:]):
				val_seqs_in = prepare_seq(train_seqs[val_idx], word_to_ix, DEVICE)
				val_true_y = [tag_to_ix[data[0]] for data in train_seqs[val_idx]]
				val_targets = torch.tensor(val_true_y, dtype=torch.long).to(DEVICE)
				val_loss = model.neg_log_likelihood(val_seqs_in, val_targets)			
				val_loss_list.append(val_loss.item())
				val_pred_y = model.forward_sess(val_seqs_in)
				val_true_ys.append(val_true_y)
				val_pred_ys.append(val_pred_y)
				val_count += 1
				print("Sequence: %d" %(val_count) + " of %d" %(len(train_seqs[train_val_index:])) + "  Elapsed Fold Time: %.3f" %((time.time()-fold_time)/60), end="\r")

		ave_val_loss_list.append(sum(val_loss_list)/len(val_loss_list))

		val_acc = crf_metrics.flat_accuracy_score(val_true_ys, val_pred_ys)
		print("\nValidation accuracy: %.2f" %(val_acc))
		val_acc_list.append(val_acc)
		val_class_report = crf_metrics.flat_classification_report(val_true_ys, val_pred_ys)
		print(val_class_report)

		if len(ave_train_loss_list) > 1 and len(ave_val_loss_list) > 1:
			print("Last ave train loss: " + str(ave_train_loss_list[-1]))
			plt.plot(ave_train_loss_list, color="#2471a3")
			plt.plot(ave_val_loss_list, color="#dc7633")
			plt.figure()
			plt.show()

		if len(train_acc_list) > 1 and len(val_acc_list) > 1:
			plt.plot(train_acc_list, color="#2471a3")
			plt.plot(val_acc_list, color="#dc7633")
			plt.figure()
			plt.show()

		# Check if the current validation loss average is higher than the previous one.
		# If so, add one to the counter. If not, reset the counter.		
		if len(ave_val_loss_list) > 1:
			print("Previous average val loss: " + str(ave_val_loss_list[-2]))
			print("Current average val loss: " + str(ave_val_loss_list[-1]))
			if ave_val_loss_list[-1] > ave_val_loss_list[-2]:
				val_loss_inc_count += 1
			else:
				val_loss_inc_count = 0

		# For each fold ...
		# Evaluate the model on the holdout set:
		print("Evaluating on holdout set ...")
		model.eval()
		test_true_ys = []
		test_pred_ys = []
		with torch.no_grad():
			for test_idx, test_sess in enumerate(test_seqs):
				test_seqs_in = prepare_seq(test_seqs[test_idx], word_to_ix, DEVICE)
				test_true_y = [tag_to_ix[data[0]] for data in test_sess]
				test_pred_y = model.forward_sess(test_seqs_in)
				test_true_ys.append(test_true_y)
				test_pred_ys.append(test_pred_y)
				print("Sequence: %d" %(test_idx+1) + " of %d" %(len(test_seqs)) 
					  + "  Elapsed Time: %.3f" %((time.time()-start_time)/60), end="\r")
		
		test_accuracy = crf_metrics.flat_accuracy_score(test_true_ys, test_pred_ys)
		print("\nTest (holdout) accuracy: " + str(test_accuracy))
		print(tag_to_ix)
		test_macro_f1 = crf_metrics.flat_f1_score(test_true_ys, test_pred_ys, average="macro")
		test_class_report = crf_metrics.flat_classification_report(test_true_ys, test_pred_ys, output_dict=True)
		normal_test_class_report = crf_metrics.flat_classification_report(test_true_ys, test_pred_ys)
		print(normal_test_class_report)
		
		ix2tag = {v:k for k,v in tag_to_ix.items()}
		
		all_true_ys = []
		for tys in test_true_ys:
			for t in tys:
				all_true_ys.append(ix2tag[t])
		
		all_pred_ys = []
		for pys in test_pred_ys:
			for p in pys:
				all_pred_ys.append(ix2tag[p])
		
		assert(len(all_true_ys) == len(all_pred_ys))
		
		tags = list(tag_to_ix.keys())[:5]
		
		cnf_matrix = confusion_matrix(all_true_ys, all_pred_ys)
		np.set_printoptions(precision=2)
		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=tags, normalize=True, title='Normalized confusion matrix')
		plt.show()
		print(cnf_matrix)
		
		# store the model, config, and eval results at every epoch:
		runs_path = os.path.join(os.getcwd(), "runs") 
		runs_path = os.path.join(runs_path, outfile)
		print("Best holdout macro-f1: " + str(best_macro_f1))
		print("Current holdout macro-f1: " + str(test_macro_f1))
		
		if test_macro_f1 > best_macro_f1:
			macrof1_dec_count = 0
			best_macro_f1 = test_macro_f1
		else:
			macrof1_dec_count += 1
		
		if not os.path.exists(runs_path):
			os.makedirs(runs_path)
		torch.save(model.state_dict(), os.path.join(runs_path, outfile + "_dict_E" + str(fold_count) + ".pth"))
		torch.save(model, os.path.join(runs_path, outfile + "_E" + str(fold_count) +".pth"))
		outfile_string = "Test (holdout) accuracy: " + str(test_accuracy)
		outfile_string = outfile_string + "\n\n" + json.dumps(tag_to_ix)
		outfile_string = outfile_string + "\n\n" + json.dumps(test_class_report, sort_keys=True, indent=4, separators=(',', ': '))
		outfile_string = outfile_string + "\n\n" + json.dumps(cfg, sort_keys=True, indent=4, separators=(',', ': '))
		with open(os.path.join(runs_path, outfile + "_report_E" + str(fold_count) + ".txt"), 'w') as file:
			file.write(outfile_string)
		
	return model

def evaluate(model, test_seqs, word_to_ix, tag_to_ix, device):
	print("Evaluating on holdout set ...")
	start_time = time.time()
	model.eval()
	test_true_ys = []
	test_pred_ys = []
	with torch.no_grad():
		for test_idx, test_sess in enumerate(test_seqs):
			test_seqs_in = prepare_seq(test_seqs[test_idx], word_to_ix, device)
			test_true_y = [tag_to_ix[data[0]] for data in test_sess]
			test_pred_y = model.forward_sess(test_seqs_in)
			test_true_ys.append(test_true_y)
			test_pred_ys.append(test_pred_y)
			print("Sequence: %d" %(test_idx+1) + " of %d" %(len(test_seqs)) 
				  + "  Elapsed Time: %.3f" %((time.time()-start_time)/60), end="\r")
	
	test_accuracy = crf_metrics.flat_accuracy_score(test_true_ys, test_pred_ys)
	print("\nTest (holdout) accuracy: " + str(test_accuracy))
	print(tag_to_ix)
	test_macro_f1 = crf_metrics.flat_f1_score(test_true_ys, test_pred_ys, average="macro")
	test_class_report = crf_metrics.flat_classification_report(test_true_ys, test_pred_ys, output_dict=True)
	normal_test_class_report = crf_metrics.flat_classification_report(test_true_ys, test_pred_ys)
	print(normal_test_class_report)
	return test_true_ys, test_pred_ys