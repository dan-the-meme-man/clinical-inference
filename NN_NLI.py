import torch
from torch import nn
import sentencepiece as spm
from math import sqrt, log

class TransformerNLI(nn.Module):
    def __init__(self, specs=None, sp=None, decoder=True):
        super(TransformerNLI, self).__init__()
        
        if sp is not None:
            self.sp = sp
        else:
            self.sp = None # FIXME: add spm.SentencePieceProcessor()
            
        self.vocab_size = sp.something # FIXME
        
        if specs is not None:
            self.dropout = specs['dropout']
            self.embed_dim = specs['embed_dim']
            self.nhead = specs['nhead']
            self.d_model = specs['d_model']
            self.dim_feedforward = specs['dim_feedforward']
            self.num_layers = specs['num_layers']
            self.conv_dim = specs['conv_dim']
            self.kernel_size = specs['kernel_size']
            self.activation = specs['activation']
            self.specs = specs
        else:
            self.dropout = 0.1
            self.embed_dim = 300
            self.nhead = 6
            self.d_model = 300
            self.dim_feedforward = 1024
            self.num_layers = 6
            self.conv_dim = 300
            self.kernel_size = 3
            self.activation = 'relu'
            self.specs = {
                'embed_dim': self.embed_dim,
                'nhead': self.nhead,
                'd_model': self.d_model,
                'dim_feedforward': self.dim_feedforward,
                'dropout': self.dropout,
                'num_layers': self.num_layers,
                'conv_dim': self.conv_dim,
                'kernel_size': self.kernel_size
            }
            
        self.scale = sqrt(self.d_model)
        self.Dropout = nn.Dropout(self.dropout)
            
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
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
            decoder_layer,
            num_layers=self.num_layers,
            norm=None,
            enable_nested_tensor=True,
            mask_check=True
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation=self.activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        self.n_params = 0
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
                self.n_params += p.numel()

        # self.act = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        # self.linear = nn.Linear(lin_dim, lin_dim)

        # self.layers = nn.ModuleList([nn.Embedding(vocab_size, embed_dim)])
        
        # for _ in range(n_layers):
        #     self.layers.append(nn.TransformerDecoderLayer(d_model=tf_dim,
        #                                                   nhead=tf_heads))
        #     self.layers.append(nn.TransformerDecoderLayer(d_model=tf_dim,
        #                                                   nhead=tf_heads))
        #     self.layers.append(nn.Conv1d(conv_dim,
        #                                  conv_dim,
        #                                  kernel_size=kernel_size,
        #                                  padding=kernel_size//2))
        #     self.layers.append(nn.MaxPool1d(kernel_size=kernel_size,
        #                                     padding=kernel_size//2))
        #     self.layers.append(nn.Dropout(dropout))
        
    def forward(self, indices, mask=None, pe=None):
        num_batch = len(indices)
        num_tokens = len(indices[0])
        
        prev = torch.cat(indices).view(num_batch, num_tokens)
        prev = self.embed(prev)
        prev = self.dropout(prev)
        
        if pe is not None: prev = prev * self.scale + pe[:num_tokens, :]
        if mask is None:
            prev = self.layers(prev)
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