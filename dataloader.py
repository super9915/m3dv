import random
import os
import numpy as np
import pandas as pd
from collections.abc import Sequence
from mylib.utils.misc import rotation, reflection, crop, random_center, _triple

path = "./dataset/"
TRAIN = pd.read_csv(os.path.join(path, 'train_val.csv'))
TEST = pd.read_csv(os.path.join(path, 'test.csv'))

class ClfDataset(Sequence):
    def __init__(self, subset=[0, 1, 2, 3]):
        index = []
        for sset in subset:
            index += list(TRAIN[TRAIN['subset'] == sset].index)
        self.index = tuple(sorted(index))

        self.label = np.array(TRAIN.loc[self.index, 'label'])


    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']

        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            voxel = npz['voxel']

        label = self.label[item]
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


class ClfSegDataset(ClfDataset):
    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']
        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            voxel, seg = npz['voxel'], npz['seg']

        label = self.label[item]
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}


class ClfvalDataset(Sequence):
    def __init__(self, crop_size=32, move=3, subset=[0, 1, 2, 3]):
        index = []
        for sset in subset:
            index += list(TRAIN[TRAIN['subset'] == sset].index)
        self.index = tuple(sorted(index))

        self.label = np.array(TRAIN.loc[self.index, 'label'])
        self.transform = Transform(crop_size, move)

    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']

        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])

        label = self.label[item]
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)


class ClfvalSegDataset(ClfvalDataset):
    def __getitem__(self, item):
        name = TRAIN.loc[self.index[item], 'name']
        with np.load(os.path.join("./dataset/train_val", '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])

        label = self.label[item]
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}


class ClfTestDataset(Sequence):
    def __init__(self, crop_size=32, move=3):
        self.index = tuple(TEST.index)
        self.label = np.array(TEST.loc[self.index, 'label'])  # label : 0/1
        self.transform = Transform(crop_size, move)

    def __getitem__(self, item):
        name = TEST.loc[self.index[item], 'name']

        with np.load(os.path.join("./dataset/test", '%s.npz' % name)) as npz:
            voxel = self.transform(npz['voxel'])
        label = self.label[item]
        return voxel, label

    def __len__(self):
        return len(self.index)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        for x, y in data:
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

class ClfSegTestDataset(ClfTestDataset):
    def __getitem__(self, item):
        name = TEST.loc[self.index[item], 'name']
        with np.load(os.path.join("./dataset/test", '%s.npz' % name)) as npz:
            voxel, seg = self.transform(npz['voxel'], npz['seg'])
        label = self.label[item]
        return voxel, (label, seg)

    @staticmethod
    def _collate_fn(data):
        xs = []
        ys = []
        segs = []
        for x, y in data:
            xs.append(x)
            ys.append(y[0])
            segs.append(y[1])
        return np.array(xs), {"clf": np.array(ys), "seg": np.array(segs)}


def get_loader_inorder(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = order_iterator(range(total_size))
    while True:
        data = []
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def get_loader(dataset, batch_size):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    while True:
        data = []
        # 0,1,2...31
        for _ in range(batch_size):
            idx = next(index_generator)
            data.append(dataset[idx])
        yield dataset._collate_fn(data)


def get_mixup_loader(dataset, batch_size, alpha=1.0):
    total_size = len(dataset)
    print('Size', total_size)
    index_generator = shuffle_iterator(range(total_size))
    transform = Transform([32, 32, 32], 3)

    while True:
        data = []
        lam = np.random.beta(alpha, alpha, batch_size)
        # print(lam)
        for i in range(batch_size):
            X_l = lam[i]
            y_l = lam[i]

            idx = next(index_generator)

            idx1 = next(index_generator)

            data0 = dataset[idx]
            data1 = dataset[idx1]

            newdata = data0[0] * X_l + data1[0] * (1 - X_l)

            lb = y_l * data0[1][0] + (1 - y_l) * data1[1][0]

            seg = X_l * data0[1][1] + (1 - X_l) * data1[1][1]

            newdata = transform(newdata)
            seg = transform(seg)

            datanew = (newdata, (lb, seg))
            data.append(datanew)
        yield dataset._collate_fn(data)


class Transform:

    def __init__(self, size, move):
        self.size = _triple(size)
        self.move = move

    def __call__(self, arr, aux=None):
        shape = arr.shape
        if self.move is not None:
            center = random_center(shape, self.move)
            arr_ret = crop(arr, center, self.size)
            angle = np.random.randint(4, size=3)
            arr_ret = rotation(arr_ret, angle=angle)
            axis = np.random.randint(4) - 1
            arr_ret = reflection(arr_ret, axis=axis)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = rotation(aux_ret, angle=angle)
                aux_ret = reflection(aux_ret, axis=axis)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret
        else:
            center = np.array(shape) // 2
            arr_ret = crop(arr, center, self.size)
            arr_ret = np.expand_dims(arr_ret, axis=-1)
            if aux is not None:
                aux_ret = crop(aux, center, self.size)
                aux_ret = np.expand_dims(aux_ret, axis=-1)
                return arr_ret, aux_ret
            return arr_ret


def shuffle_iterator(iterator):
    # iterator should have limited size
    index = list(iterator)  # 0~464
    total_size = len(index)
    i = 0
    random.shuffle(index)

    while True:
        yield index[i]
        i += 1
        if i >= total_size:
            i = 0
            random.shuffle(index)


def order_iterator(iterator):
    index = list(iterator)
    total_size = len(index)
    i = 0


    while True:
        yield index[i]
        # print(i)
        i += 1
        if i >= total_size:
            i = 0
