# HamNet
Conformation-Guided Molecular Representation with Hamiltonian Neural Networks. In ICLR 2021

All the used datasets are available on [MoleculeNet Dataset](http://moleculenet.ai/datasets-1).

You can run the `xxx.slurm` to reproduce the results claimed in paper.
- Run `fit.slurm` first to train a HamEngine (run `fit2.slurm` if you don't need adj3_loss)
- Run molecule property prediction jobs with the trained HamEngine (molecule conformation generator)
