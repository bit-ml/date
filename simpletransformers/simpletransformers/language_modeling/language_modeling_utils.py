import copy
import io
import itertools
import logging
import os
import pickle
from multiprocessing import Pool
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer
from sklearn.metrics import auc, roc_curve, precision_recall_curve

logger = logging.getLogger(__name__)


def encode(data):
    tokenizer, line = data
    return tokenizer.encode(line)


def encode_sliding_window(data):
    tokenizer, line, max_seq_length, special_tokens_count, stride, no_padding = data

    tokens = tokenizer.tokenize(line)
    stride = int(max_seq_length * stride)
    token_sets = []
    if len(tokens) > max_seq_length - special_tokens_count:
        token_sets = [tokens[i : i + max_seq_length - special_tokens_count] for i in range(0, len(tokens), stride)]
    else:
        token_sets.append(tokens)

    features = []
    if not no_padding:
        sep_token = tokenizer.sep_token_id
        cls_token = tokenizer.cls_token_id
        pad_token = tokenizer.pad_token_id

        for tokens in token_sets:
            tokens = [cls_token] + tokens + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)

            assert len(input_ids) == max_seq_length

            features.append(input_ids)
    else:
        for tokens in token_sets:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(input_ids)

    return features



def encode_sliding_window_custom(data, tokenizer):
    line, max_seq_length, special_tokens_count, stride, no_padding = data

    tokens = tokenizer.tokenize(line)
    stride = int(max_seq_length * stride)
    token_sets = []
    if len(tokens) > max_seq_length - special_tokens_count:
        token_sets = [tokens[i : i + max_seq_length - special_tokens_count] for i in range(0, len(tokens), stride)]
    else:
        token_sets.append(tokens)
    import sys

    features = []
    if not no_padding:
        sep_token = tokenizer.sep_token
        cls_token = tokenizer.cls_token
        pad_token = tokenizer.pad_token_id

        for tokens in token_sets:
            tokens = [cls_token] + tokens + [sep_token]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            padding_length = max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_length)

            assert len(input_ids) == max_seq_length

            features.append(input_ids)
    else:
        logger.info("I NEVER GET HERE")
        for tokens in token_sets:
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            features.append(input_ids)

    return features

class SimpleDataset2(Dataset):
    def __init__(self, tokenizer, args, file_path, mode, block_size=514, special_tokens_count=2, sliding_window=False):
        super().__init__()
        self.tokenizer

        assert os.path.isfile(file_path)
        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)

        no_padding = False

        with open(file_path, encoding="utf-8") as f:
            lines = [
                (tokenizer, line, args.max_seq_length, special_tokens_count, args.stride, no_padding)
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

# CUSTOM:

class SimpleDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=514, special_tokens_count=2, sliding_window=False):
        logger.info(f" SimpleDataset - {file_path}")
        assert os.path.isfile(file_path)
        self.tokenizer = tokenizer

        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)

        no_padding = False

        with open(file_path, encoding="utf-8") as f:
            logger.info(f" Opened {file_path}")
            self.examples = [
                encode_sliding_window_custom((line, args.max_seq_length, special_tokens_count, args.stride, no_padding), tokenizer)
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]
            logger.info(f" Made example list. Length: {len(self.examples)}. Expanding...")

        self.examples = [example for example_set in self.examples for example in example_set]

        logger.info(" Expanded example list.")
        logger.info(f" Examples: {len(self.examples)}")
        logger.info(80 * '-')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


class DocumentDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, block_size=514, special_tokens_count=2, sliding_window=False):
        logger.info(f" DocumentDataset - {file_path}")
        assert os.path.isfile(file_path)
        self.tokenizer = tokenizer

        block_size = block_size - special_tokens_count
        directory, filename = os.path.split(file_path)

        logger.info(" Creating document dataset from %s", file_path)

        no_padding = False

        with open(file_path, encoding="utf-8") as f:
            logger.info(f" Opened {file_path}")

            self.examples = [
                [encode_sliding_window_custom((line, args.max_seq_length, special_tokens_count, args.stride, no_padding), tokenizer)]
                for line in f.read().splitlines()
                if (len(line) > 0 and not line.isspace())
            ]

            logger.info(f" Made example list. Total samples: {len(self.examples)}. Expanding...")

        self.examples = [example for example_set in self.examples for example in example_set]

        idx_bot = [len(ex) for ex in self.examples]

        idx_l = []
        for idx in idx_bot:
            idx_l.append(idx) if len(idx_l) == 0 else idx_l.append(idx_l[-1] + idx)
        self.idx_l = idx_l[:-1]

        self.examples = [exp for ex_list in self.examples for exp in ex_list]

        logger.info(" Expanded example list.")
        logger.info(f" Examples: {len(self.examples)}")
        logger.info(80 * '-')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def neg_entropy(score: np.array):
    """ https://github.com/demonzyj56/E3Outlier
    """
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return (score+1)@np.log2(score+1)

def pl_score(score: np.array):
    return score

def mp_score(score: np.array):
    score = np.max(score, axis=1)
    return score

def calculate_acc(predict: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        logits = F.softmax(predict)
        predictions = logits.max(dim=1)[1]

        predictions = predictions.cpu().numpy()
        gt = gt.cpu().numpy()
        correct = (gt == predictions).sum()
        total = len(gt)
        acc = correct/total

        return acc, predictions, gt

def plot_confusion_matrix(cm, class_names):
    import seaborn as sn
    sn.set(font_scale=1.4)
    hm = sn.heatmap(cm, annot=True, annot_kws={"size": 16})
    figure = hm.get_figure()
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    from: https://towardsdatascience.com/exploring-confusion-matrix-evolution-on-tensorboard-e66b39f4ac12
    """

    buf = io.BytesIO()

    plt.savefig(buf, format='png')

    plt.close(figure)
    buf.seek(0)

    from PIL import Image
    import torchvision.transforms.functional as TF

    img = Image.open(buf)
    img_t = TF.to_tensor(img)
    return img_t


def mask_tokens_vanilla(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling."
            "Set 'mlm' to False in args if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]

    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    if args.model_type == "electra":
        # For ELECTRA, we replace all masked input tokens with tokenizer.mask_token
        inputs[masked_indices] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    else:
        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    PAD_VALUE = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    CLS_VALUE = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
    MSK_VALUE = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


def mask_tokens(batch: torch.Tensor, tokenizer: PreTrainedTokenizer, masks_, args, custom_mask=None, train=None, no_mask=False) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    import random

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling."
            "Set 'mlm' to False in args if you want to use this tokenizer."
        )

    def get_first_occurence(array, value):
        """ Not used
        """
        where = np.argwhere(array == value)
        indices = np.full((array.shape[0]), -1)
        for el in where:
            if indices[el[0]] != -1:
                continue
            indices[el[0]] = el[1]
        return indices, where

    def get_first_occurence_array(array, value):
        for idx, el in enumerate(array):
            if el == value:
                break
        return idx

    PAD_VALUE = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    MSK_VALUE = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    inputs = batch.clone()
    labels = inputs.clone()

    masks = copy.deepcopy(masks_)

    for msk in masks:
        msk['mask'] = [False] + msk['mask'] + [False]
        #always mask first element
        if no_mask==False:
            msk['mask'][1] = True

    # We sample a few tokens in each sequence for masked-LM training
    # (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    batch_size = labels.shape[0]
    # batch_element = 0

    def find_viable_masks(masklist, first_pad_idx):
        viable_masks = []
        for msk in masklist:
            for idx, el in enumerate(msk['mask']):
                if idx >= first_pad_idx:
                    break
                if el == True:
                    viable_masks.append(msk)
                    break
        return viable_masks

    if custom_mask is None:
        viable_masks = masks

        max_labels = len(viable_masks)
        choice = random.randint(0, max_labels-1)

        picked_mask = viable_masks[choice]['mask']
        picked_label = [viable_masks[choice]['label']]

        first_mask = picked_mask

        masked_indices = torch.Tensor(first_mask).unsqueeze(0).bool()

        clf_labels = torch.Tensor(picked_label).unsqueeze(0)

        while(masked_indices.shape[0] < batch_size):
            choice = random.randint(0, max_labels-1)

            picked_mask = viable_masks[choice]['mask']
            picked_label = [viable_masks[choice]['label']]

            new_mask = picked_mask

            new_mask = torch.Tensor(new_mask).unsqueeze(0).bool()
            new_label = torch.Tensor(picked_label).unsqueeze(0)

            masked_indices = torch.cat((masked_indices, new_mask))
            clf_labels = torch.cat((clf_labels, new_label))

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

    else:
        picked_mask = [False] + custom_mask['mask'] + [False]
        #always mask first element
        if no_mask == False:
            picked_mask[1] = True

        picked_label = [custom_mask['label']]
        first_mask = picked_mask

        masked_indices = torch.Tensor(first_mask).unsqueeze(0).bool()
        clf_labels = torch.Tensor(picked_label).unsqueeze(0)

        while(masked_indices.shape[0] < batch_size):
            new_mask = picked_mask

            new_mask = torch.Tensor(new_mask).unsqueeze(0).bool()
            new_label = torch.Tensor(picked_label).unsqueeze(0)

            masked_indices = torch.cat((masked_indices, new_mask))
            clf_labels = torch.cat((clf_labels, new_label))

        labels[~masked_indices] = -100  # We only compute loss on masked tokens

    indices_replaced = torch.bernoulli(torch.full(labels.shape, 1.)).bool() & masked_indices

    pad_idx = np.nonzero(inputs.numpy() == PAD_VALUE)
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    inputs[pad_idx] = PAD_VALUE
    labels[pad_idx] = -100

    return inputs, labels, clf_labels


def mask_tokens_multiple(batch: torch.Tensor, tokenizer: PreTrainedTokenizer, masks_, args, custom_mask=None, train=None, mask_ablation=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    masks = []
    for maskset in masks_:
        inputs, labels, labels_clf = mask_tokens(batch, tokenizer, maskset, args, train=True, mask_ablation=None)
        masks.append((inputs, labels, labels_clf))
    return masks

def merge_batches(dataloader, tqdm_name, b_size=32, seq_len=514): # Deprecated // change magic number
    docs = 0
    batch = torch.rand((0, seq_len))
    idx_l = []
    space_left = b_size
    print('LEN OF DATALOADER', len(dataloader))

    for idx, doc in enumerate(tqdm(dataloader, desc=tqdm_name)):
        idx_l.append(doc.shape[0]) if len(idx_l) == 0 else idx_l.append(idx_l[-1] + doc.shape[0])
        while(doc.shape[0] != 0):
            batch = torch.cat((batch, doc[:space_left]), 0)
            added = doc[:space_left].shape[0]

            doc = doc[space_left:]
            space_left -= added

            if(space_left == 0):
                yield (batch, idx_l[:-1])
                batch = torch.rand((0, seq_len))
                space_left = b_size

            if(idx == len(dataloader)-1 and doc.shape[0] <= b_size and batch.shape[0] != 0): # last el in batch
                yield (batch, idx_l[:-1])

def get_metrics(gt, preds):
    fpr, tpr, roc_thresholds = roc_curve(gt, preds)
    cutoff = np.argmax(tpr-fpr)

    roc_auc = auc(fpr, tpr)

    precision_norm, recall_norm, pr_thresholds_norm = precision_recall_curve(gt, preds)
    pr_auc_norm = auc(recall_norm, precision_norm)

    precision_anom, recall_anom, pr_thresholds_anom = precision_recall_curve(gt, -preds, pos_label=0)
    pr_auc_anom = auc(recall_anom, precision_anom)

    return roc_auc, pr_auc_norm, pr_auc_anom, roc_thresholds[cutoff]
