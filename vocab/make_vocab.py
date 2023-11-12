import os
import sys
sys.path.append('.')

from data.retrieve_data import get_data

train_data = get_data('train', use_control=False, flatten=False)

s = ''

for item in train_data:
    s += item[0]
    
with open(os.path.join('vocab', 'vocab.txt'), 'w+', encoding='utf-8') as f:
    f.write(s)