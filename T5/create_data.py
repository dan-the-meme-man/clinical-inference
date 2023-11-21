import json

def update_json(infile_path, outfile_path):
    j = ''
    with open(infile_path, 'r') as read_content:
        j = json.load(read_content)

    with open(outfile_path, 'w') as outfile:
        #json.dump(data, outfile)
        outfile.write(
        '[' +
        ',\n'.join(json.dumps(i) for i in j) +
        ']\n')


if __name__ == "__main__":
    path = 'Data/'
    in_train = path + 'train_data.json'
    in_dev = path + 'dev_data.json'
    out_train = path + 'train.json'
    out_dev = path + 'dev.json'
    update_json(in_train, out_train)
    update_json(in_dev, out_dev)

    # data = ''
    # with open('Data/train_data.json') as r:
    #     data = json.load(r)
    # #data = json.load('Data/train_data.json')
    # print(data)
    # with open('test.json', "w") as tf:
    #     tf.write(
    #     ',\n'.join(json.dumps(i) for i in data) +
    #     '\n')


