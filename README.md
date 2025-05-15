Codes used to train and test models for CCS prediction in immunopeptides.

Objectives:
1) To evaluate the suitability of an existing language model (PPLN) for predicting CCS in immunopeptides.
2) To propose improvements to the existing language model for CCS prediction in immunopeptides.
3) To use the improvements to build dedicated language models for immunopeptides, where possible.

This repository contains/will contain:
1) Data: Immunopeptide data derived from initially 2 public datasets: PXD038782 (TOF-IMS/Orbitrap data) and JPST002044 (timsTOF).
2) Models: PPLN pretrained model and the ESM-1b and 2 models (to be included later).
3) Microsoft BitNet as a model resource optimizer.
4) Links to Colab, where most codes will be run.

The order of functioning is as follows:
1) Prepare the datasets to contain at the minimum peptide sequence, charge, mass and collision cross section values (1/k0 or invk0) in addition to other optional parameters such as length.
2) Input the prepared data to the "preprocessing.py" script of PPLN and run preprocessing.
3) Perform training and testing with the "main" script.
4) Output will be the predicted CCS values per immunopeptide, which must be evaluated using either or all of RMSE, MAE and Pearson's correlation.
5) Output will also be evaluated on how accurately the CCS values can be predicted for immune peptides and whether it can distinguish between MHC-1 and MHC-2.
