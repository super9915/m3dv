import os
import pandas as pd
import numpy as np

from dataloader import *
from keras.optimizers import Adam, SGD
from mylib.models.misc import set_gpu_usage

set_gpu_usage()

from mylib.models import densesharp, metrics, losses
from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard, EarlyStopping, ReduceLROnPlateau, \
    LearningRateScheduler

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'


def main(batch_size, crop_size, learning_rate, segmentation_task_ratio, weight_decay, save_folder, epochs,
         alpha):

    print(learning_rate)
    print(alpha)
    print(weight_decay)

    train_dataset = ClfSegDataset(subset=[0, 1])
    train_loader = get_mixup_loader(train_dataset, batch_size=batch_size, alpha=alpha)

    val_dataset = ClfvalSegDataset(crop_size=crop_size, move=None, subset=[2])
    val_loader = get_loader(val_dataset, batch_size=batch_size)

    model = densesharp.get_compiled(output_size=1,
                                    optimizer=Adam(lr=learning_rate),
                                    loss={"clf": 'binary_crossentropy',
                                          "seg": losses.DiceLoss()},
                                    metrics={'clf': ['accuracy', metrics.precision, metrics.recall, metrics.fmeasure,
                                                     metrics.auc],
                                             'seg': [metrics.precision, metrics.recall, metrics.fmeasure]},
                                    loss_weights={"clf": 1., "seg": segmentation_task_ratio},
                                    weight_decay=weight_decay, weights='tmp/test/weights42_222639.h5')

    checkpointer = ModelCheckpoint(filepath='tmp/%s/weights.{epoch:02d}.h5' % save_folder, verbose=1,
                                   period=1, save_weights_only=True)
    csv_logger = CSVLogger('tmp/%s/training.csv' % save_folder)
    tensorboard = TensorBoard(log_dir='tmp/%s/logs/' % save_folder)

    best_keeper = ModelCheckpoint(filepath='tmp/%s/best.h5' % save_folder, verbose=1, save_weights_only=True,
                                  monitor='val_clf_acc', save_best_only=True, period=1, mode='max')

    early_stopping = EarlyStopping(monitor='val_clf_acc', min_delta=0, mode='max',
                                   patience=20, verbose=1)

    lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.334, patience=10,
                                   verbose=1, mode='min', epsilon=1.e-5, cooldown=2, min_lr=0)

    model.fit_generator(generator=train_loader, steps_per_epoch=50, max_queue_size=10, workers=1,
                        validation_data=val_loader, epochs=epochs, validation_steps=50,
                        callbacks=[checkpointer, csv_logger, best_keeper, early_stopping, lr_reducer, tensorboard])


if __name__ == '__main__':
    main(batch_size=32,
         crop_size=[32, 32, 32],
         learning_rate=1.e-5,
         segmentation_task_ratio=0.2,
         weight_decay=0.0,
         save_folder='test',
         epochs=10,
         alpha=1.0)