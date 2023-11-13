import json
import os
train = 'training_data/train.json'

with open(train, 'r') as train_json:
    train_data = json.load(train_json)


def get_input_target_pair(split_data):
    data_keys = list(split_data.keys())
    data_pair = []
    for i in range(len(data_keys)):
        data_point = {}
        section_id = split_data[data_keys[i]]['Section_id']
        statement = split_data[data_keys[i]]['Statement']
        label = split_data[data_keys[i]]['Label']
        primary_ctr_path = os.path.join("training_data/CT json", split_data[data_keys[i]]["Primary_id"] + ".json")
        with open(primary_ctr_path) as json_file:
            primary_text = json.load(json_file)[section_id]
            data_point['primary_text'] = primary_text
        if split_data[data_keys[i]]['Type'] == 'Comparison':
            secondary_ctr_path = os.path.join("training_data/CT json",
                                              split_data[data_keys[i]]["Secondary_id"] + ".json")
            with open(secondary_ctr_path) as secondary_file:
                secondary_text = json.load(secondary_file)[section_id]
                data_point['secondary_text'] = secondary_text
        else:
            data_point['secondary_text'] = None
        data_point['statement'] = statement
        data_point['label'] = label

        data_pair.append(data_point)

    return data_pair

train_data_pair = get_input_target_pair(train_data)
with open('training_data/train_data.json', 'w') as train:
    json.dump(train_data_pair, train)