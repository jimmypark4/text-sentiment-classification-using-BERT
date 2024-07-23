from sklearn.metrics import classification_report, accuracy_score, f1_score
from utils import to_gpu

import models
import torch
import torch.nn as nn
import numpy as np
import os

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

            valid_loss, valid_acc = self.eval(mode="dev")

            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            if valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                print("Found new best model on dev set!")
                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        self.eval(mode="test", to_print=True)


    def eval(self, mode=None, to_print=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss = []

        if mode=="dev":
            dataloader=self.dev_data_loader
        elif mode=="test":
            dataloader=self.test_data_loader

            if to_print:
                self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))


        with torch.no_grad():
            for batch in dataloader:
                self.model.zero_grad()
                sent, label, lengths, bert_sent, bert_sent_type, bert_sent_mask = batch

                sent=to_gpu(sent)
                bert_sent=to_gpu(bert_sent)
                bert_sent_type=to_gpu(bert_sent_type)
                bert_sent_mask=to_gpu(bert_sent_mask)
                label=to_gpu(label.type(torch.LongTensor))

                pred=self.model(bert_sent, bert_sent_type, bert_sent_mask)
                loss=self.criterion(pred, label)

                eval_loss.append(loss.item())
                y_pred.append(pred.detach().cpu().numpy())
                y_true.append(label.detach().cpu().numpy())

        eval_loss=np.mean(eval_loss)
        y_true=np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        y_pred = np.argmax(y_pred, axis=1)

        corrects=(y_true==y_pred)
        
        accuracy=100*(sum(corrects)/len(corrects))
        
        if to_print:
            print('accuracy: ', accuracy)
        return eval_loss, accuracy





















