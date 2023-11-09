import os
import torch
from torch import nn
import sentencepiece as spm
from math import sqrt, log

class TransformerNLI(nn.Module):
    def __init__(self, specs=None, mode='encoder_decoder'):
        super(TransformerNLI, self).__init__()
        
        sp = spm.SentencePieceProcessor()
        sp.Load(os.path.join('vocab', 'spm.model'))
            
        self.vocab_size = sp.get_piece_size()
        
        if specs is not None:
            self.d_model = specs['d_model']
            self.num_layers = specs['num_layers']
            self.nhead = specs['nhead']
            self.dim_feedforward = specs['dim_feedforward']
            self.embed_dim = specs['embed_dim']
            self.dropout = specs['dropout']
            self.activation = specs['activation']
            self.specs = specs
        else:
            self.d_model = 256
            self.num_layers = 6
            self.nhead = 6
            self.dim_feedforward = 1024
            self.embed_dim = 256
            self.dropout = 0.1
            self.activation = 'relu'
            self.specs = {
                'd_model': self.d_model,
                'num_layers': self.num_layers,
                'nhead': self.nhead,
                'dim_feedforward': self.dim_feedforward,
                'embed_dim': self.embed_dim,
                'dropout': self.dropout,
                'activation': self.activation
            }
        
        # define a name to log output to
        self.name = f'TransformerNLI_{mode}_d{self.d_model}_l{self.num_layers}_h{self.nhead}'
        self.name += f'_ff{self.dim_feedforward}_e{self.embed_dim}_do{self.dropout}_a{self.activation}'
        self.specs['name'] = self.name
        
        self.scale = sqrt(self.d_model)
            
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        
        if mode == 'encoder_decoder' or mode == 'decoder':
            self.decoder_layer = nn.TransformerDecoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                layer_norm_eps=1e-5,
                batch_first=False,
                norm_first=False,
                bias=True,
                device=None,
                dtype=None
            )
            self.decoder = nn.TransformerDecoder(
                self.decoder_layer,
                num_layers=self.num_layers,
                norm=None,
                enable_nested_tensor=True,
                mask_check=True
            )
            if mode == 'decoder':
                self.encoder = None
        if mode == 'encoder_decoder' or mode == 'encoder':
            self.encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                activation=self.activation,
                layer_norm_eps=1e-5,
                batch_first=False,
                norm_first=False,
                bias=True,
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
            if mode == 'encoder':
                self.decoder = None
        
        self.n_params = 0
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                self.n_params += p.numel()
    
    # TODO: add mask, pe, and encoder/decoder
    def forward(self, indices, mask=None, pe=None):
        num_batch = len(indices)
        num_tokens = len(indices[0])
        
        prev = torch.cat(indices).view(num_batch, num_tokens)
        prev = self.embed(prev)
        prev = self.dropout(prev)
        
        if pe is not None: prev = prev * self.scale + pe[:num_tokens, :]
        if mask is None:
            prev = self.layers(prev) # TODO: encoder/decoder? could do T5 thing
        else:
            prev = self.layers(prev, mask=mask)
        
        prev = self.dropout(prev)
        
        out = torch.matmul(prev, torch.t(self.embed.weight))
        
        return out
    
    def create_self_attention_mask(self, max_size):
        self_attn_mask = (torch.triu(torch.ones(max_size, max_size)) == 1).transpose(0, 1)
        self_attn_mask = self_attn_mask.float().masked_fill(self_attn_mask == 0, float('-inf')).masked_fill(self_attn_mask == 1, float(0.0))
        return self_attn_mask
    
    def create_positional_encoding(self, max_size):
        positional_encoding = torch.zeros(max_size, self.d_model)
        position = torch.arange(0, max_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-log(10000.0) / self.d_model))
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        # positional_encoding = positional_encoding.unsqueeze(0).transpose(0, 1)
        return positional_encoding