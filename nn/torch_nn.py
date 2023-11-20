import sys
sys.path.append('.')

import os
from math import sqrt, log

import torch
from torch import nn
import sentencepiece as spm

class TransformerNLI(nn.Module):
    def __init__(self, specs=None, device='cpu'):
        super(TransformerNLI, self).__init__()

        self.device = device
        
        # vocab
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(os.path.join('vocab', 'spm.model'))
        self.vocab_size = self.sp.get_piece_size()
        self.pad_id = self.sp.PieceToId('<pad>')
        
        if specs is not None:
            self.d_model = specs['d_model']
            self.num_layers = specs['num_layers']
            self.nhead = specs['nhead']
            self.dim_feedforward = specs['dim_feedforward']
            self.embed_dim = specs['embed_dim']
            self.dropout = specs['dropout']
            self.activation = specs['activation']
            self.max_length = specs['max_length']
            self.specs = specs
        else:
            self.d_model = 256
            self.num_layers = 6
            self.nhead = 6
            self.dim_feedforward = 1024
            self.embed_dim = 256
            self.dropout = 0.1
            self.activation = 'relu'
            self.max_length = 3000
            self.specs = {
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'nhead': self.nhead,
                'dim_feedforward': self.dim_feedforward,
                'embed_dim': self.embed_dim,
                'dropout': self.dropout,
                'activation': self.activation,
                'max_length': self.max_length
            }
            
        # pe
        self.pe = self.create_positional_encoding(self.max_length).to(self.device)
        
        # define a name to log output to
        self.name = f'TransformerNLI_d_{self.d_model}_l_{self.num_layers}_h_{self.nhead}'
        self.name += f'_ff_{self.dim_feedforward}_e_{self.embed_dim}_do_{self.dropout}_a_{self.activation}'
        self.specs['name'] = self.name
        
        self.scale = sqrt(self.d_model)
            
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            device=None,
            dtype=None
        )
        self.encoder = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=self.num_layers,
            norm=None,
            enable_nested_tensor=True,
            mask_check=True
        )
                
        self.linear = nn.Linear(self.max_length * self.d_model, 1)
        
        self.n_params = 0
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                self.n_params += p.numel()
        print(self.n_params)
        exit()
        # dropout
        self.Dropout = nn.Dropout(self.dropout)

    def forward(self, indices):
        
        # dimensions of input
        batch_size = indices.size(0)
        max_seq_length = indices.size(1)
        
        # mask
        attn_mask = self.create_attn_mask(max_seq_length)
        padding_mask = self.create_padding_mask(indices)
        
        # embedding with dropout and pe
        embed = self.embed(indices)
        embed = self.Dropout(embed)
        embed = embed*self.scale + self.pe[:max_seq_length, :]
        
        # encoder
        enc_out = self.encoder(embed, mask=attn_mask, src_key_padding_mask=padding_mask)
        enc_out = self.Dropout(enc_out)
        enc_out = enc_out.view(batch_size, -1)
        
        # return single logit for classification
        return self.linear(enc_out)
    
    # mask for padding
    def create_padding_mask(self, indices):
        return (indices != self.pad_id).float().to(self.device)
    
    # mask for self-attention
    def create_attn_mask(self, max_seq_length):
        attn_mask = (torch.triu(torch.ones(max_seq_length, max_seq_length)) == 1).transpose(0, 1)
        attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf'))
        attn_mask = attn_mask.masked_fill(attn_mask == 1, float(0.0))
        return attn_mask.to(self.device)
    
    # positional encoding
    def create_positional_encoding(self, max_len):
        positional_encoding = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-log(10000.0) / self.d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        return positional_encoding
