from pybel import *
data_directory = '../data'

import gzip
import numpy as np
from sklearn.model_selection import train_test_split
from augment_data import augment_example_rotate

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

def load_data(startct, ligandct):
	# Get the list of folders in the data directory
	import os
	x_data = []
	y_data = []
	files = os.listdir(data_directory + '/dude')
	count = 0
	for filename in files[startct:startct+1]:
		# parse the receptor coordinates
		receptors = readfile('pdb', data_directory + 
			'/dude/' + filename + '/receptor.pdb')
		receptor = None
		for r in receptors:
			receptor = r
			print(receptor.title)
		# parse through active compounds
		ligands = readfile('sdf', data_directory + '/dude/' + filename +
		'/actives_final.sdf.gz')
		lict = 0
		for ligand in ligands:
			#print(ligand.title)
			if lict > ligandct + 10:
				break
			if lict >= ligandct:
				# develop a 48^3 array for each example
				# x, y, z, c format, 13 channels one for each atom type
				newExample = np.zeros(shape=(64, 64, 64, 13))
				for atom in receptor:
					# get atomic coordinates
					x, y, z = atom.coords
					# convert atomic number to atom channel one hot encoding
					#print(atom.atomicnum)
					# rescale and round coords
					x *= 64. / 150
					y *= 64. / 150
					z *= 64. / 150
					x = int(x)
					y = int(y)
					z = int(z)
					if x > 63:
						x = 63
					if y > 63:
						y = 63
					if z > 63:
						z = 63
					if x < 0:
						x = 0
					if y < 0:
						y = 0
					if z < 0:
						z = 0
					if atom.atomicnum in atom_types:
						ch = atom_types[atom.atomicnum]
						newExample[x][y][z][ch] = 1.
				for atom in ligand:
					# get atomic coordinates
					x, y, z = atom.coords
					# convert atomic number to atom channel one hot encoding
					# print(atom.atomicnum)
					# rescale and round coords
					x *= 64. / 150
					y *= 64. / 150
					z *= 64. / 150
					x = int(x)
					y = int(y)
					z = int(z)
					if atom.atomicnum in atom_types:
						ch = atom_types[atom.atomicnum]
						newExample[x][y][z][ch] = 1.
				# add all augmented examples
				if count == 0:
					x_data = augment_example_rotate(newExample)
				else:
					x_data = np.concatenate((x_data, augment_example_rotate(newExample)))
					#print(len(x_data))
				# augment adds 4 examples total
				if count == 0:
					y_data = [[1., 0.], [1., 0.], [1., 0.], [1., 0.]]
				else:
					y_data = np.concatenate((y_data, [[1., 0.], [1., 0.], [1., 0.], [1., 0.]]))
				count += 1
				#print("COUNT: " + str(count), end="\r")
			lict += 1



		try:
			# parse through inactive compounds
			ligands = readfile('sdf', data_directory + '/dude/' + filename +
			'/decoys_final.sdf.gz')
			lict = 0
			for ligand in ligands:
				#print(ligand.title)
				if lict > ligandct + 10:
					break
				if lict >= ligandct:
					# develop a 48^3 array for each example
					# x, y, z, c format, 13 channels one for each atom type
					newExample = np.zeros(shape=(64, 64, 64, 13))
					for atom in receptor:
						# get atomic coordinates
						x, y, z = atom.coords
						# convert atomic number to atom channel one hot encoding
						#print(atom.atomicnum)
						# rescale and round coords
						x *= 64. / 150
						y *= 64. / 150
						z *= 64. / 150
						x = int(x)
						y = int(y)
						z = int(z)
						if x > 63:
							x = 63
						if y > 63:
							y = 63
						if z > 63:
							z = 63
						if x < 0:
							x = 0
						if y < 0:
							y = 0
						if z < 0:
							z = 0
						if atom.atomicnum in atom_types:
							ch = atom_types[atom.atomicnum]
							newExample[x][y][z][ch] = 1.
					for atom in ligand:
						# get atomic coordinates
						x, y, z = atom.coords
						# convert atomic number to atom channel one hot encoding
						# print(atom.atomicnum)
						# rescale and round coords
						x *= 64. / 150
						y *= 64. / 150
						z *= 64. / 150
						x = int(x)
						y = int(y)
						z = int(z)
						if atom.atomicnum in atom_types:
							ch = atom_types[atom.atomicnum]
							newExample[x][y][z][ch] = 1.
					# add all augmented examples
					if count == 0:
						x_data = augment_example_rotate(newExample)
					else:
						x_data = np.concatenate((x_data, augment_example_rotate(newExample)))
						#print(len(x_data))
					# augment adds 4 examples total
					if count == 0:
						y_data = [[0., 1.], [0., 1.], [0., 1.], [0., 1.]]
					else:
						y_data = np.concatenate((y_data, [[0., 1.], [0., 1.], [0., 1.], [0., 1.]]))
					count += 1
					#print("COUNT: " + str(count), end="\r")
				lict += 1
		except:
			continue
	x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
	return x_train, x_test, y_train, y_test

								
