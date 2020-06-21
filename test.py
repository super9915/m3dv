import os
import pandas as pd
import numpy as np

from dataloader import *
from keras.optimizers import Adam, SGD
from mylib.models.misc import set_gpu_usage

set_gpu_usage()  # use gpu to run the code

from mylib.models import densesharp, metrics, losses
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, \
    LearningRateScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'

path = "./dataset/"
TEST = pd.read_csv(os.path.join(path, 'test.csv'))

crop_size=[32, 32, 32]

test_dataset = ClfSegTestDataset(crop_size=crop_size, move=None)
test_loader = get_loader_inorder(test_dataset, batch_size=1)

model = densesharp.get_compiled()
model.load_weights('./tmp/test/best12.h5')

pred = model.predict_generator(generator=test_loader, steps=len(test_dataset), verbose=1)

index = tuple(TEST.index)
name = TEST.loc[index, 'name']
name.tolist()

df = pd.DataFrame(columns=['name'], data=name)
df['predicted'] = pred[0]
df.to_csv('submission.csv',index=False)