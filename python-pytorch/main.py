from solver import Solver
from config import get_config
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import pickle
import os
import models
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

data_path = r'C:\text-sentiment-classification-using-BERT\data\sentiment_analysis.csv'
word2id = defaultdict(lambda: len(word2id))
PAD = word2id['']
UNK = word2id['']

def return_unk():
    return UNK

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

class Tweet:
    def __init__(self, config):
        self.train, self.dev, self.test = [], [], []
        data = []
        self.word2id = word2id

        if not os.path.exists('train.pkl'):
            _words, y = [], []
            for line in open(config.data_file_path, encoding='utf-8'):
                line = line.strip()
                splits = line.split(',')
                if splits[0] == 'id':
                    continue
                label = np.asarray([float(splits[1])])
                sentence = splits[2]
                _words.append(sentence)
                y.append(label)

            for i in range(len(_words)):
                words, actual_words = [], []
                for word in _words[i].split(' '):
                    word = word.replace("#", "")
                    actual_words.append(word)
                    words.append(word2id[word])

                words = np.asarray(words)
                data.append(((words, actual_words), y[i]))

            self.train = data[:4000]
            self.dev = data[4000:6000]
            self.test = data[6000:]
            word2id.default_factory = return_unk

            to_pickle(self.train, './train.pkl')
            to_pickle(self.dev, './dev.pkl')
            to_pickle(self.test, './test.pkl')

        else:
            self.train = load_pickle('train.pkl')
            self.dev = load_pickle('dev.pkl')
            self.test = load_pickle('test.pkl')

    def get_data(self, mode):
        if mode == 'train':
            return self.train, self.word2id
        elif mode == 'dev':
            return self.dev, self.word2id
        elif mode == 'test':
            return self.test, self.word2id
        else:
            print('Mode is not set properly (train/dev/test)')
            exit()


class SADataset(Dataset):
    def __init__(self, config):
        dataset = Tweet(config)
        self.data, self.word2id = dataset.get_data(config.mode)
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


def get_loader(config, shuffle=True):
    dataset = SADataset(config)

    def collate_fn(batch):
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)

        SENT_LEN = sentences.size(0)

        bert_details = []
        for sample in batch:
            text = " ".join(sample[0][1])

            encoded_bert_sent = tokenizer.encode_plus(
                text, max_length=SENT_LEN + 2, add_special_tokens=True, pad_to_max_length=True)
            bert_details.append(encoded_bert_sent)

        bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
        bert_sentences_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
        bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, labels, lengths, bert_sentences, bert_sentences_types, bert_sentence_att_mask

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=64,
        shuffle=shuffle,
        collate_fn=collate_fn
    )
    return data_loader

train_config = get_config(mode='train')
dev_config = get_config(mode='dev')
test_config = get_config(mode='test')

train_data_loader = get_loader(train_config, shuffle=True)
dev_data_loader = get_loader(dev_config, shuffle=False)
test_data_loader = get_loader(test_config, shuffle=False)

solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)

solver.build()
solver.train()
solver.eval(mode='test', to_print=True)


#
#
# from itertools import product
#
#
# best_accuracy = 0
# best_params = {}
#
# # Try 128 combinations of num_heads and num_layers
# num_heads_options = [1,2,4, 8, 16,32,64,128]
# num_layers_options = [1,2, 4, 8,16,32,64,128]
#
# for num_heads, num_layers in product(num_heads_options, num_layers_options):
#     train_config = get_config(mode='train')
#     dev_config = get_config(mode='dev')
#     test_config = get_config(mode='test')
#
#     train_data_loader = get_loader(train_config, shuffle=True)
#     dev_data_loader = get_loader(dev_config, shuffle=False)
#     test_data_loader = get_loader(test_config, shuffle=False)
#
#     print(f'Training with num_heads={num_heads}, num_layers={num_layers}')
#     solver = Solver(train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True)
#     solver.build()
#     solver.train()
#     _, accuracy = solver.eval(mode='test', to_print=True)
#
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_params = {'num_heads': num_heads, 'num_layers': num_layers}
#
# print(f'Best Accuracy: {best_accuracy} with Parameters: {best_params}')