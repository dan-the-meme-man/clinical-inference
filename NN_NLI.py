from torch import nn

class TransformerNLI(nn.Module):
    def __init__(self, specs=None):
        super(TransformerNLI, self).__init__()
        
        if specs is not None:
            vocab_size = specs['vocab_size']
            embed_dim = specs['embed_dim']
            tf_heads = specs['tf_heads']
            tf_dim = specs['tf_dim']
            lin_dim = specs['lin_dim']
            conv_dim = specs['conv_dim']
            kernel_size = specs['kernel_size']
            dropout = specs['dropout']
            n_layers = specs['n_layers']
        else:
            vocab_size = 10000
            embed_dim = 300
            tf_heads = 6
            tf_dim = 300
            lin_dim = 300
            conv_dim = 300
            kernel_size = 3
            dropout = 0.1
            n_layers = 6

        #self.act = nn.ReLU()
        #self.softmax = nn.Softmax(dim=1)
        #self.linear = nn.Linear(lin_dim, lin_dim)

        self.layers = nn.ModuleList([nn.Embedding(vocab_size, embed_dim)])
        
        for _ in range(n_layers):
            self.layers.append(nn.TransformerDecoderLayer(d_model=tf_dim,
                                                          nhead=tf_heads))
            self.layers.append(nn.TransformerDecoderLayer(d_model=tf_dim,
                                                          nhead=tf_heads))
            self.layers.append(nn.Conv1d(conv_dim,
                                         conv_dim,
                                         kernel_size=kernel_size,
                                         padding=kernel_size//2))
            self.layers.append(nn.MaxPool1d(kernel_size=kernel_size,
                                            padding=kernel_size//2))
            self.layers.append(nn.Dropout(dropout))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x