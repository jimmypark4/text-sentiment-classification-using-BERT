from transformers import BertModel, BertConfig
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

import torch
import numpy as np
import pickle
import os

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')



bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
     

bertmodel = BertModel.from_pretrained('bert-base-uncased', config=bertconfig)
     


word2id = defaultdict(lambda: len(word2id))
PAD=word2id['']
UNK=word2id['']

def return_unk():
    return UNK

def to_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
data_path='./sentiment_analysis.csv'

class Tweet:
    def __init__(self):
        self.train = train =[]
        self.word2id = word2id
        if not os.path.exists('train.pkl'):
           actual_words=[]
           _words=[]
           y=[] 
           #read data
           for line in open(data_path):
               line=line.strip()
               splits=line.split(',')
               id=splits[0]
               if id=='id':
                   continue
               label=np.asarray([float(splits[1])])
               sentence=splits[2]
               _words.append(sentence)
               y.append(label)
                
           for i in range(len(_words)):
               words=[]
               for word in _words[i].split(' '):
                   actual_words.append(word)
                   words.append(word2id[word])

               words=np.asarray(words)
               label=y[i]
               train.append(((words,actual_words),label))

           word2id.default_factory=return_unk

           to_pickle(train, './train.pkl')

        else:
           self.train=load_pickle('train.pkl')

    def get_data(self):

        return self.train, self.word2id


class SADataset(Dataset):
    #todo:
    def __init__(self):
        dataset=Tweet()
        self.data, self.word2id = dataset.get_data()
        self.len = len(self.data)
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

def get_loader(shuffle=True):
    dataset=SADataset()

    def collate_fn(batch):
        batch=sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        sentences=pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)

        SENT_LEN = sentences.size(0)

        bert_details = []
        for sample in batch:
            text=" ".join(sample[0][1])
            encoded_bert_sent=bert_tokenizer.encode_plus(
                    text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)
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
            collate_fn=collate_fn)
    return data_loader



train_data_loader = get_loader(shuffle=True)

for sent, label, lengths, bert_sent, bert_sent_type, bert_sent_mask in train_data_loader:
    bert_output=bertmodel(input_ids=bert_sent,
                            attention_mask=bert_sent_mask,
                            token_type_ids=bert_sent_type)
    bert_output=bert_output[0]

    masked_output=torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
    mask_len=torch.sum(bert_sent_mask, dim=1, keepdim=True)
    bert_output=torch.sum(masked_output, dim=1, keepdim=False) / mask_len

    utterance_text=bert_output

    print('u:', utterance_text.shape)













