import os
from os import listdir
from os.path import isfile, join
import random

def contaminate_ds(root, clean_file, contamination, write=True):
    print('Clean file:', clean_file)
    all_files = [f for f in listdir(root) if isfile(join(root, f))]
    all_files.remove(clean_file)

    clean_fp = open(join(root, clean_file), 'r')
    clean_file_lines = clean_fp.read().split('\n')
    clean_file_lines = [line for line in clean_file_lines if line not in ['', ' ', '\n']]

    print(f'Total clean entries: {len(clean_file_lines)}.')

    anomalies = []
    for anom_name in all_files:
        print(anom_name)
        anom_fp = open(join(root, anom_name), 'r')
        anomalies.extend(anom_fp.read().split('\n'))

    anomalies = [line for line in anomalies if line not in ['', ' ', '\n']]

    print(f'Total anomaly entries: {len(anomalies)}.')
    random.shuffle(anomalies)
    print('Shuffled anomalies.')
    print(f'Contamination: {contamination}%')

    no_entries_anomaly = int( (100 * len(clean_file_lines))/(100-contamination) ) - len(clean_file_lines)

    print(f'Entries to be added: {no_entries_anomaly}')
    clean_file_lines.extend(anomalies[:no_entries_anomaly])
    random.shuffle(clean_file_lines)
    print(f'Total len of contaminated ds:', len(clean_file_lines))
    print('Shuffled clean entries.')

    if write:
        contaminated_ds = ''
        for item in clean_file_lines:
            if item!='\n':
                contaminated_ds += f'{item}\n'
            else:
                print('newline ignored')

        if not os.path.exists(root + f'/{clean_file[:-4]}-contaminated/'):
            os.makedirs(root + f'/{clean_file[:-4]}-contaminated/')
        with open(join(root + f'/{clean_file[:-4]}-contaminated/',f'{clean_file[:-4]}_c{contamination}.txt'), 'w') as f:
            f.write(contaminated_ds)

    print('\n\n')

# AG News
phases = ['train', 'test']
ag_subsets = ['business', 'sci', 'sports', 'world']
contamination = [5, 10, 15]

for phase in phases:
    for subset in ag_subsets:
        for cont in contamination:
            contaminate_ds(f'./ag_od/{phase}', f'{subset}.txt', cont)

# 20Newsgroups
ag_subsets = ['comp', 'misc', 'pol', 'rec', 'rel', 'sci']
contamination = [5, 10, 15]

for phase in phases:
    for subset in ag_subsets:
        for cont in contamination:
            contaminate_ds(f'./20ng_od/{phase}', f'{subset}.txt', cont)