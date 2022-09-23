# An implementation based on the ConLSTM-CRF model described in:
# Olafsson, S., Wallace, B. C., & Bickmore, T. W. (2020, May). Towards a Computational Framework for Automating Substance Use Counseling with Virtual Agents. In AAMAS (pp. 966-974).

import pandas as pd
from gensim.models import Word2Vec

def make_dicts(train_path, test_path, start_tag, stop_tag):
	train_data = pd.read_csv(train_path, encoding="latin1")
	train_data = train_data.fillna(method="ffill")
	print("Train data length: " + str(len(train_data)))

	test_data = pd.read_csv(test_path, encoding="latin1")
	test_data = test_data.fillna(method="ffill")
	print("Test data length: " + str(len(test_data)))

	tags = list(set(train_data["Tag"].values)) + list(set(test_data["Tag"].values))

	tag_to_ix = {}
	for t in tags:
		if t not in tag_to_ix:
			tag_to_ix[t] = len(tag_to_ix)
	
	tag_to_ix[start_tag] = len(tag_to_ix)
	tag_to_ix[stop_tag] = len(tag_to_ix)

	words = []
	word_to_ix = {}
	max_sent_len = 0
	all_words = [word for word in list(train_data["Text"].values) + list(test_data["Text"].values) if word != ""]
	print("All words list: " + str(len(all_words)))
	for text in all_words:
		try:
			words = text.split(" ")
		except:
			print(text)
		cur_word_len = len(words)
		if cur_word_len > max_sent_len:
			max_sent_len = cur_word_len
		for word in words:
			if word not in word_to_ix:
				word_to_ix[word] = len(word_to_ix)
				words.append(word)
	
	return tag_to_ix, word_to_ix


def train_word2vec(train, test, cfg):
	texts = [txt.strip().split(" ") for txt in train.Text.values.tolist()] + [txt.strip().split(" ") for txt in test.Text.values.tolist()]
	return Word2Vec(texts, size=cfg['emb'], min_count=1, window=5)
