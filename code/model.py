# Training parameters
num_epochs = 200
batch_size = 4

from resnet3d import Resnet3DBuilder
import keras as K
from keras.optimizers import SGD
# ResNet-18, 48^3 input, 2 node softmax output, l2 regularization
model = Resnet3DBuilder.build_resnet_18((64, 64, 64, 13), 2, reg_factor=0.01)

model.compile(optimizer=SGD(lr=0.01, momentum=0.99, nesterov=True),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# load data
from load_data import load_data
startct = 0

# save model checkpoints
from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('../models/model.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', save_best_only=True, mode='max')

# use cyclic LR for faster learning
from clr_callback import CyclicLR
lr_cycler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=2000, mode='triangular2')
epoch = 0
ligandct = 0
while epoch < num_epochs:
    while startct < 10000:
        while ligandct < 200:
            # fit model
            x_train, x_test, y_train, y_test = load_data(startct, ligandct)
            # don't train if there's no decoys
            if len(x_train) < 50:
                break
            print(ligandct)
            model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test))
            ligandct += 10
        startct += 1
        ligandct = 0
    epoch += 1
    model.save('../models/' + str(epoch) + '-model.h5')
    startct = 0
    