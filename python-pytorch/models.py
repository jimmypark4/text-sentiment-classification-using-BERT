from transformers import MambaModel, MambaConfig, BertModel, BertConfig

import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        rnn = nn.LSTM
        lstm_hidden_size = 200

        self.rnn1 = rnn(768, int(lstm_hidden_size), bidirectional=True)
        self.rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional=True)

        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        self.fc = nn.Linear(800, 2)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)

        bert_output = bert_output[0]

        batch_size = bert_output.shape[0]

        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        utterance_text = torch.unsqueeze(bert_output, axis=0)

        _, (h1, _) = self.rnn1(utterance_text)
        _, (h2, _) = self.rnn2(h1)

        o = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        o = self.fc(o)

        return o
class LSTMTransformer(nn.Module):
    def __init__(self, config, num_heads=8, num_layers=2):
        super(LSTMTransformer, self).__init__()
        rnn = nn.LSTM
        lstm_hidden_size = 200

        self.rnn1 = rnn(768, int(lstm_hidden_size), bidirectional=True)
        self.rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional=True)

        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=800, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        self.fc = nn.Linear(800, 2)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)

        bert_output = bert_output[0]

        batch_size = bert_output.shape[0]

        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        utterance_text = torch.unsqueeze(bert_output, axis=0)

        _, (h1, _) = self.rnn1(utterance_text)
        _, (h2, _) = self.rnn2(h1)

        o = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        o = o.unsqueeze(0).permute(1, 0, 2)  # (batch_size, seq_len=1, feature_size)
        transformer_output = self.transformer_encoder(o)

        transformer_output = transformer_output.squeeze(1)  # (batch_size, feature_size)
        logits = self.fc(transformer_output)

        return logits

class LSTMMamba(nn.Module):
    def __init__(self, config, num_heads=8, num_layers=2):
        super(LSTMMamba, self).__init__()
        rnn = nn.LSTM
        lstm_hidden_size = 200

        self.rnn1 = rnn(768, int(lstm_hidden_size), bidirectional=True)
        self.rnn2 = rnn(int(lstm_hidden_size), int(lstm_hidden_size), bidirectional=True)

        self.bertconfig = BertConfig.from_pretrained('bert-base-uncased', output_hidden_states=True)
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased', config=self.bertconfig)

        self.mamba_model = MambaModel.from_pretrained(r'C:\text-sentiment-classification-using-BERT\python-pytorch\mamba-130m', ignore_mismatched_sizes=True)

        self.fc = nn.Linear(800, 2)

    def forward(self, bert_sent, bert_sent_type, bert_sent_mask):
        bert_output = self.bertmodel(input_ids=bert_sent,
                                     attention_mask=bert_sent_mask,
                                     token_type_ids=bert_sent_type)

        bert_output = bert_output[0]

        batch_size = bert_output.shape[0]

        masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
        mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        utterance_text = torch.unsqueeze(bert_output, axis=0)

        _, (h1, _) = self.rnn1(utterance_text)
        _, (h2, _) = self.rnn2(h1)

        o = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        mamba_output = self.mamba_model(inputs_embeds=o)

        mamba_output = mamba_output.last_hidden_state.squeeze(1)
        logits = self.fc(mamba_output)

        return logits
