# Training parameters
num_epochs = 200
batch_size = 32

from resnet3d import Resnet3DBuilder
model = Resnet3DBuilder.build_resnet_18((48, 48, 48, 1), 2, reg_factor=0.01)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from load_data import load_data
x_train, x_test, y_train, y_test = load_data()

from keras.callbacks import ModelCheckpoint
model_checkpoint = ModelCheckpoint('./models/model.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', save_best_only=True, mode='max')
from clr_callback import CyclicLR
lr_cycler = CyclicLR(base_lr=0.001, max_lr=0.01, step_size=2000, mode='triangular2')
model.fit(x_train, y_train, batch_size=batch_size, callbacks=[model_checkpoint, lr_cycler], epochs=num_epochs, validation_data=(x_test, y_test))