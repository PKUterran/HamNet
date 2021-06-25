# HamNet
Conformation-Guided Molecular Representation with Hamiltonian Neural Networks. In ICLR 2021

Note that QM9 dataset is not uploaded to this project due to 100M capacity limit, but it's available on [MoleculeNet Dataset](http://moleculenet.ai/datasets-1). After downloading QM9(structure), just unzip it and copy `gdb9.sdf` to `data/gdb9/` and everything is ok.

You can run the `xxx.slurm` to reproduce the results claimed in paper.
- Run `fit.slurm` first to train a HamEngine (run `fit2.slurm` if you don't need adj3_loss)
- Run molecule property prediction jobs with the trained HamEngine (molecule conformation generator)
