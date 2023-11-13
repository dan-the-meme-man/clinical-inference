import sys
sys.path.append('.')

import os

# pulls dataset splits
from data.retrieve_data import get_data

# pull train set
train_data = get_data('train', use_control=False, flatten=False)

s = '' # string to write to file

# append all training sentences to string
for item in train_data:
    s += item[0]

# write string to file
with open(os.path.join('vocab', 'vocab.txt'), 'w+', encoding='utf-8') as f:
    f.write(s)