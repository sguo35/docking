# Protein-Ligand interaction docking program
- Discretize protein-ligand atomic 3d coordinates into a 64^3 matrix
- Use ResNet3D-18 to identify patterns
- End w/binary softmax

# Dependencies
- `pip install git+https://github.com/JihongJu/keras-resnet3d.git`
- `sudo apt-get install openbabel`
- `pip install openbabel` 

# Todo
- GPU accelerated and/or asynchronous data prep for faster training and inference
- Use a larger and more modern model e.g. NASNet or SENet
- Only use binding pocket atoms?
