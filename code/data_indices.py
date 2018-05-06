from pybel import *
data_directory = '../data'

import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from augment_data import augment_example_rotate

def get_raw(data_directory):
	# Get the list of folders in the data directory
	import os
	files = os.listdir(data_directory + '/dude')
	dataset = []
	ct = 0
	import random
	# shuffle files
	random.shuffle(files)
	for filename in files[:5]:
		# parse the receptor coordinates
		receptors = readfile('pdb', data_directory + '/dude/' + filename + '/receptor.pdb')
		receptor = None
		for r in receptors:
			receptor = r
		# parse through active compounds
		ligands = readfile('sdf', data_directory + '/dude/' + filename + '/actives_final.sdf.gz')
		for ligand in ligands:
			dataset.append(DataExample(receptor, ligand, True))
		# parse through inactive compounds
		ligands = readfile('sdf', data_directory + '/dude/' + filename + '/decoys_final.sdf.gz')
		for ligand in ligands:
			dataset.append(DataExample(receptor, ligand, False))
		ct += 1
		print(ct)
	return dataset


class DataExample():
	def __init__(self, x_receptor, x_ligand, y):
		self.x_receptor = x_receptor
		self.x_ligand = x_ligand
		self.y = y


import numpy as np

# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.
import keras as K
class Docking(K.utils.Sequence):

	def __init__(self, batch_size=16):
		self.x = get_raw('../data')
		self.batch_size = batch_size
		np.random.shuffle(self.x)

	def __len__(self):
		return int(np.ceil(len(self.x) / float(self.batch_size)))

	def on_epoch_end(self):
		# shuffle in place after each epoch
		np.random.shuffle(self.x)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

		x_data = []
		y_data = []

		for i in range(len(batch_x)):
			newExample = np.zeros(shape=(48, 48, 48, 13))
			# add the receptor first
			newExample = preprocess(newExample, batch_x[i].x_receptor)
			# add the ligand
			newExample = preprocess(newExample, batch_x[i].x_ligand)
			# randomly rotate it
			newExample = np.rot90(newExample, k=np.random.randint(1, 4))
			# append the example
			x_data.append(newExample)
			# onehot the y
			if batch_x[i].y == True:
				y_data.append([1., 0.])
			else:
				y_data.append([0., 1.])
		return np.array(x_data), np.array(y_data)


atom_types = {
	6: 0, # carbon
	35: 1, # bromine
	20: 2, # calcium
	17: 12, # chlorine
	9: 3, # fluorine
	53: 4, # iodine
	26: 5, # iron
	12: 6, # mg
	7: 7, # N
	8: 8, # O
	15: 9, # P
	16: 10, # sulphur
	30: 11, # zinc
}

def preprocess(arr, sdfObj):
	for atom in sdfObj:
		# get atomic coordinates
		# convert atomic number to atom channel one hot encoding
		# rescale and round coords
		x, y, z = atom.coords
		x *= 48. / 75
		y *= 48. / 75
		z *= 48. / 75
		x = int(x)
		y = int(y)
		z = int(z)
		if x > 47:
			x = 47
		if y > 47:
			y = 47
		if z > 47:
			z = 47
		if x < 0:
			x = 0
		if y < 0:
			y = 0
		if z < 0:
			z = 0
		if atom.atomicnum in atom_types:
			ch = atom_types[atom.atomicnum]
			arr[x][y][z][ch] = 1.
	# avoid python pass by ref issues
	return arr