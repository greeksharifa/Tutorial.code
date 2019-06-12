import glob
import random
import string
import torch


def read_files():
    """
    make training_set from files

    # gender: 'M', 'F'
    :return: training_set, which is a list of (name, gender)
    """
    training_set = []
    name_set = set()

    for filename in glob.iglob('names/yob*.txt'):
        with open(filename, 'r', encoding='utf-8') as fp:
            for line in fp.readlines():
                name, gender, _ = line.split(',')
                data = (name, gender)
                if data not in name_set:
                    training_set.append(data)
                    name_set.add(data)

        print('\rFile: {} | len(train_set): {:8d}'.format(filename, len(training_set)), end='')
    print()

    # random.shuffle(training_set)

    return training_set


def get_all_letters():
    """
    :return: 'abcdefg...xyzABCDEFG...XYZ'
    """
    # print(help(string))
    all_letters = string.ascii_letters
    return len(all_letters), all_letters


def prepare_sequence(seq, to_ix):
    idx_list = [to_ix[w] for w in seq]
    return torch.tensor(idx_list, dtype=torch.long)

