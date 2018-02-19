# Protein-Ligand interaction docking program
- Discretize atomic 3d coordinates into a 48x48x48 matrix
- Use ResNet3D-18 to identify patterns
- End w/binary softmax

# Dependencies
- `pip install git+https://github.com/JihongJu/keras-resnet3d.git`
- `sudo apt-get install openbabel`
- `pip install openbabel` 