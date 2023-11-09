import os
import json

def main():
    
    # path to clinical trials jsons
    CT_dir = os.path.join('..', 'Task-2-SemEval-2024', 'training_data', 'training_data', 'CT json')
    
    if not os.path.exists(CT_dir):
        raise Exception('Cloned repo not found. Please run fetch_task.py first.')

    # get all json filenames
    CT_json_names = [os.path.join(CT_dir, x) for x in os.listdir(CT_dir) if x.endswith('.json')]

    # load all jsons
    CT_jsons = [json.load(open(x, 'r', encoding='utf-8')) for x in CT_json_names]

    # create dict of jsons with Clinical Trial ID as key
    keys = []
    for i in range(len(CT_json_names)):
        assert CT_json_names[i].endswith('.json')
        key = CT_json_names[i].split(os.path.sep)[-1][:-5]
        assert '.json' not in key
        keys.append(key)
    CT_dict = dict(zip(keys, CT_jsons))
    
    # dump dict to json file
    CT_dict_file = 'CT_dict.json'
    json.dump(CT_dict, open(CT_dict_file, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
    
    # load dumped jsons and ensure they have been serialized properly
    jsons_dumped = json.load(open(CT_dict_file, 'r', encoding='utf-8'))
    print(f'{len(jsons_dumped)} jsons dumped to {CT_dict_file}.')
    for json_dumped in jsons_dumped:
        assert json_dumped == jsons_dumped[json_dumped]['Clinical Trial ID']

if __name__ == '__main__':
    main()
