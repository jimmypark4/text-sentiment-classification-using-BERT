from utils import to_gpu

import models
import torch
import torch.nn as nn
import numpy as np

class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader,is_train=True, model=None):
        self.train_config = train_config
        self.train_data_loader=train_data_loader
        self.dev_data_loader=dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model

    def build(self, cuda=True):
        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)

        for name, param in self.model.named_parameters():
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            
        if torch.cuda.is_available() and cuda:
            self.model.cuda()


        if self.is_train:
            self.optimizer=self.train_config.optimizer(
                    filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.train_config.learning_rate)

    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials=1
        self.criterion = criterion = nn.CrossEntropyLoss(reduction='mean')
        
        best_valid_loss=float('inf')
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        train_losses = []
        valid_losses = []

        for e in range(self.train_config.n_epoch):
            print(f'epoch {e+1}')
            train_loss=[]
            self.model.train()
            for batch in self.train_data_loader:
                self.model.zero_grad()
                sent, label, lengths, bert_sent, bert_sent_type, bert_sent_mask = batch

                bert_sent=to_gpu(bert_sent)
                bert_sent_type=to_gpu(bert_sent_type)
                bert_sent_mask=to_gpu(bert_sent_mask)
                label=to_gpu(label.type(torch.LongTensor))

                pred = self.model(bert_sent, bert_sent_type, bert_sent_mask)

                loss = criterion(pred, label)

                loss.backward()

                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)

                self.optimizer.step()
                train_loss.append(loss.item())

            train_losses.append(train_loss)
            print(f"Training loss: {round(np.mean(train_loss), 4)}")

















