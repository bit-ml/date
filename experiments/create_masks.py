import pickle as pkl

sample_len = 128
mask_list = []
no_of_masks = 100
generated_masks = 0
perc_of_els = 0.5

def mask_random_els(mask, perc_of_els=0.5):
    from random import randint
    total_els = len(mask)
    masked_els = int(total_els*perc_of_els)
    already_masked = 0
    while(already_masked != masked_els):
        mask_idx = randint(0, total_els-1)
        if(mask[mask_idx] == True):
            continue
        else:
            mask[mask_idx] = True
            already_masked += 1
    return mask

def mask_exists(mask, mask_list):
    no_of_els = len(mask)
    identical_els = 0
    for mask2 in mask_list:
        for el1, el2 in zip(mask, mask2):
            if(el1 == el2):
                identical_els += 1
        if(identical_els == no_of_els):
            return True
    return False


while(generated_masks != no_of_masks):
    mask = [False for _ in range(sample_len)]
    mask = mask_random_els(mask, perc_of_els)

    if(not mask_exists(mask, mask_list)):
        mask_list.append(mask)
        generated_masks += 1
    else:
        print('Found a dupe')
print('done')

def common_elements(list1, list2):
    common = 0

    for el1, el2 in zip(list1, list2):
        if el1==el2:
            common += 1
    return common

def check_for_duplicates(list_of_lists):
    max_common = len(list_of_lists[0])
    no_of_commons = 0
    print('MAX COMMON:', max_common)
    for idx1 in range(len(list_of_lists)):
        for idx2 in range(idx1+1, len(list_of_lists)):
            total_common = common_elements(list_of_lists[idx1], list_of_lists[idx2])
            if  total_common == max_common:
                print('COMMON: ', idx1, idx2)
                no_of_commons += 1

    print('TOTAL IDENTICAL:', no_of_commons)

check_for_duplicates(mask_list)

pseudo_labels = []
for label, mask in enumerate(mask_list):
    pseudo_labels.append({'mask': mask, 'label': label})

pkl.dump(pseudo_labels, open(f'./mask.pkl', 'wb'))

