import pickle

import numpy as np
from tqdm import tqdm

from process_image import get_image, process_image


def get_test_data(train_set_size, test_set_size):
    with open("data/female_names.txt") as file:
        females = np.array(file.read().splitlines())
        np.random.shuffle(females)
    with open("data/male_names.txt") as file:
        males = np.array(file.read().splitlines())
        np.random.shuffle(males)

    full_size = int((train_set_size + test_set_size) / 2)
    selected_females = np.c_[np.full((full_size, 1), 0, dtype=np.int), females[:full_size].reshape(-1, 1)]
    selected_males = np.c_[np.full((full_size, 1), 1, dtype=np.int), males[:full_size].reshape(-1, 1)]

    full_set = np.r_[selected_females, selected_males]
    np.random.shuffle(full_set)

    labels = full_set[:, 0].astype(int)
    file_names = full_set[:, 1]
    hogs = np.array([process_image(get_image("lfw_funneled_normalized/" + file_name)) for file_name in tqdm(file_names)])

    train_set = labels[:train_set_size], hogs[:train_set_size]
    test_set = labels[train_set_size:], hogs[train_set_size:]
    return train_set, test_set


def save_to_pickle(my_object, file_name):
    with open(file_name, "wb") as output:
        pickle.dump(my_object, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    save_to_pickle(get_test_data(1000, 200), "data/test_data/t_1000_200.pkl")
    save_to_pickle(get_test_data(2000, 400), "data/test_data/t_2000_400.pkl")
    save_to_pickle(get_test_data(1500, 300), "data/test_data/t_1500_300.pkl")
