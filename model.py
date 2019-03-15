import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import BertModel, BertPreTrainedModel
import numpy as np


class BertForNQ(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForNQ, self).__init__(config)
        self.bert = BertModel(config)

        self.type_rnn = torch.nn.LSTM(
            config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.type_output = nn.Linear(2*config.hidden_size, 5)
        self.start_rnn = torch.nn.LSTM(
            3*config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.start_output = nn.Linear(2*config.hidden_size, 1)
        self.end_rnn = torch.nn.LSTM(
            5*config.hidden_size, config.hidden_size, batch_first=True, bidirectional=True)
        self.end_output = nn.Linear(2*config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None, ans_types=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)

        type_rnn_out, (hn, cn) = self.type_rnn(sequence_output)
        hn = hn.permute(1,0,2).contiguous().view(hn.size(1), -1)
        type_logits = self.type_output(hn)

        start_sequence = torch.cat([sequence_output, type_rnn_out], dim=-1)
        start_rnn_out, _ = self.start_rnn(start_sequence)
        start_logits = self.start_output(start_rnn_out).squeeze(-1)

        end_sequence = torch.cat([start_sequence, start_rnn_out], dim=-1)
        end_rnn_out, _ = self.end_rnn(end_sequence)
        end_logits = self.end_output(end_rnn_out).squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(ans_types.size()) > 1:
                ans_types = ans_types.squeeze(-1)
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            type_loss_fct = CrossEntropyLoss()
            type_loss = type_loss_fct(type_logits, ans_types)
            total_loss = type_loss + start_loss + end_loss
            return total_loss
        else:
            return F.softmax(start_logits, dim=0), F.softmax(end_logits, dim=0), F.softmax(type_logits, dim=0)
