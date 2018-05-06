# Protein-Ligand interaction docking program
- Discretize protein-ligand atomic 3d coordinates into a 48^3 matrix
- Use ResNet3D-18 to identify patterns
- End w/binary softmax

# Dependencies
- `conda install -c openbabel`

# Todo
- GPU accelerated and/or ~~asynchronous data prep~~ for faster training and inference
- Use a larger and more modern model e.g. NASNet or SENet or ResNeXT
- Only use binding pocket atoms?
