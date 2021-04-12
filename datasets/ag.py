import wget
import pandas as pd
import numpy as np
import os
from clean_text import clean_text
import string

def dump_data(phase, name, text, path='./ag_od/'):
    full_path = os.path.join(path, phase)

    if not os.path.exists(full_path):
        os.makedirs(full_path)

    with open(f'{full_path}/{name}.txt', 'w') as fp:
        fp.write(text)
        print('Succesfully written', f'{full_path}/{name}.txt')

def export_ds(subsets, phase):
    raw_text = {
        "world" : '',
        "sports" : '',
        "business": '',
        "sci": ''
    }

    for data_label, subset in zip(subsets, raw_text):
        for text in subsets[data_label]:
            text = clean_text(text)
            raw_text[subset] += f'\n\n{text}'

    for subset in raw_text:
        dump_data(phase, subset, raw_text[subset])

wget.download("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv")
wget.download("https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv")

ag_train = pd.read_csv('./train.csv', header=None)
ag_test = pd.read_csv('./test.csv', header=None)
ag_test.columns = ['label', 'title', 'description']
ag_train.columns = ['label', 'title', 'description']

subsets_test = {
    "1": [],
    "2": [],
    "3": [],
    "4": []
}

for idx, el in enumerate(np.array(ag_test)):
    label = el[0]
    text = el[1] + ' ' + el[2]
    subsets_test[f'{label}'].append(text)

print('Total samples (test)', idx+1)

export_ds(subsets_test, 'test')

subsets_train = {
    "1": [],
    "2": [],
    "3": [],
    "4": []
}

for idx, el in enumerate(np.array(ag_train)):
    label = el[0]
    text = el[1] + ' ' + el[2]
    subsets_train[f'{label}'].append(text)

print("Total samples (train):", idx+1)

export_ds(subsets_train, 'train')
