import random
import torch
import numpy as np
import pandas as pd

PAD = 0
MASK = 1


def map_column(df: pd.DataFrame, col_name: str):
    """
    Maps column values to integers
    :param df:
    :param col_name:
    :return:
    """
    values = sorted(list(df[col_name].unique()))
    mapping = {k: i + 2 for i, k in enumerate(values)}
    inverse_mapping = {v: k for k, v in mapping.items()}

    df[col_name + "_mapped"] = df[col_name].map(mapping)

    return df, mapping, inverse_mapping


def get_context(df: pd.DataFrame, split: str, context_size: int = 120, val_context_size: int = 5):
    """
    Create a training / validation samples
    Validation samples are the last horizon_size rows
    :param df:
    :param split:
    :param context_size:
    :param val_context_size:
    :return:
    """
    if split == "train":
        end_index = random.randint(10, df.shape[0] - val_context_size)
    elif split in ["val", "test"]:
        end_index = df.shape[0]
    else:
        raise ValueError

    start_index = max(0, end_index - context_size)

    context = df[start_index:end_index]

    return context


def pad_arr(arr: np.ndarray, expected_size: int = 30):
    """
    Pad top of array when there is not enough history
    :param arr:
    :param expected_size:
    :return:
    """
    arr = np.pad(arr, [(expected_size - arr.shape[0], 0), (0, 0)], mode="edge")
    return arr


def pad_list(list_integers, history_size: int, pad_val: int = PAD, mode="left"):
    """

    :param list_integers:
    :param history_size:
    :param pad_val:
    :param mode:
    :return:
    """

    if len(list_integers) < history_size:
        if mode == "left":
            list_integers = [pad_val] * (history_size - len(list_integers)) + list_integers
        else:
            list_integers = list_integers + [pad_val] * (history_size - len(list_integers))

    return list_integers


def df_to_np(df, expected_size=30):
    arr = np.array(df)
    arr = pad_arr(arr, expected_size=expected_size)
    return arr


def genome_mapping(genome):
    genome.sort_values(by=["movieId", "tagId"], inplace=True)
    movie_genome = genome.groupby("movieId")["relevance"].agg(list).reset_index()

    movie_genome = {a: b for a, b in zip(movie_genome['movieId'], movie_genome['relevance'])}

    return movie_genome

def mask_list(l1, p=0.8):

    l1 = [a if random.random() < p else MASK for a in l1]

    return l1


def mask_last_elements_list(l1, val_context_size: int = 5):

    l1 = l1[:-val_context_size] + mask_list(l1[-val_context_size:], p=0.5)

    return l1


class Dataset(torch.utils.data.Dataset):
    def __init__(self, groups, grp_by, split, history_size=120):
        self.groups = groups
        self.grp_by = grp_by
        self.split = split
        self.history_size = history_size

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        group = self.groups[idx]

        df = self.grp_by.get_group(group)

        context = get_context(df, split=self.split, context_size=self.history_size)

        trg_items = context["movieId_mapped"].tolist()

        if self.split == "train":
            src_items = mask_list(trg_items)
        else:
            src_items = mask_last_elements_list(trg_items)

        pad_mode = "left" if random.random() < 0.5 else "right"
        trg_items = pad_list(trg_items, history_size=self.history_size, mode=pad_mode)
        src_items = pad_list(src_items, history_size=self.history_size, mode=pad_mode)

        src_items = torch.tensor(src_items, dtype=torch.long)

        trg_items = torch.tensor(trg_items, dtype=torch.long)

        return src_items, trg_items