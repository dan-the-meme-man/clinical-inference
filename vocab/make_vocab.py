from retrieve_data import get_data

train_data = get_data('train', use_control=False)

for item in train_data:
    print(item[0])
    break