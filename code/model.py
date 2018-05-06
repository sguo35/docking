# Training parameters
num_epochs = 200
batch_size = 4

from resnet import residual_network
import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
# ResNet-18, 48^3 input, 2 node softmax output, l2 regularization
inputLayer = Input(shape=(48, 48, 48, 13,))
model = residual_network(inputLayer)
model = Dense(2, activation='softmax')(model)
model = Model(inputs=[inputLayer], outputs=[model])

def precision(y_true, y_pred):
	"""Precision metric.

	Only computes a batch-wise average of precision.

	Computes the precision, a metric for multi-label classification of
	how many selected items are relevant.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

def recall(y_true, y_pred):
	"""Recall metric.

	Only computes a batchwise average of recall.

	Computes the recall, a metric for multilabel classification of
	how many relevant items are selected.
	"""
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall

def fbeta_score(y_true, y_pred, beta=1):
	"""Computes the F score.

	The F score is the weighted harmonic mean of precision and recall.
	Here it is only computed as a batchwise average, not globally.

	This is useful for multilabel classification, where input samples can be
	classified as sets of labels. By only using accuracy (precision) a model
	would achieve a perfect score by simply assigning every class to every
	input. In order to avoid this, a metric should penalize incorrect class
	assignments as well (recall). The Fbeta score (ranged from 0.0 to 1.0)
	computes this, as a weighted mean of the proportion of correct class
	assignments vs. the proportion of incorrect class assignments.

	With beta = 1, this is equivalent to a Fmeasure. With beta < 1, assigning
	correct classes becomes more important, and with beta > 1 the metric is
	instead weighted towards penalizing incorrect class assignments.
	"""
	if beta < 0:
		raise ValueError('The lowest choosable beta is zero (only precision).')

	# If there are no true positives, fix the F score at 0 like sklearn.
	if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
		return 0

	p = precision(y_true, y_pred)
	r = recall(y_true, y_pred)
	bb = beta ** 2
	fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
	return fbeta_score


def fmeasure(y_true, y_pred):
	"""Computes the fmeasure, the harmonic mean of precision and recall.

	Here it is only computed as a batchwise average, not globally.
	"""
	return fbeta_score(y_true, y_pred, beta=1)

model.compile(optimizer='adam',
			  loss='categorical_crossentropy',
			  metrics=['accuracy', precision, recall, fmeasure])


# load data
from data_indices import Docking

epoch = 0
while epoch < num_epochs:
	dockingGenerator = Docking(batch_size=batch_size)
	model.fit_generator(dockingGenerator, epochs=1, verbose=1, workers=4, use_multiprocessing=True, shuffle=True, initial_epoch=0)
	model.save('../models/' + str(epoch) + '-model.h5')
	epoch += 1