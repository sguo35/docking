import numpy as np

""" Augments an example by rotating it. """
def augment_example_rotate(example):
  augmented_data = []
  augmented_data.append(example)
  for i in range(start=1, stop=4):
    # Rotate the given 4d array 4 times
      augmented_data.append(np.rot90(example, k=i))
  return augmented_data