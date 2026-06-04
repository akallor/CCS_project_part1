This repository details the development of a collision-cross section (CCS) predictor dedicated to immunopeptides, both HLA-I and HLA-II peptides.
Separate predictors have been developed for each class, which can be found in the "CCS_unified_model" directory.

The predictor works as follows:

<img width="10050" height="5465" alt="CCS_model_explained" src="https://github.com/user-attachments/assets/21d47d66-b0a6-4e8e-a29a-e5757df36036" />


1) A feature extractor consisting of the ESM-2 protein language model concatenated with charge tokens and peptide features (hydrophobicity, polarity, basicity) to form a combined feature vector.
2) A bidirectional, 2-layered, LSTM recurrent neural network (RNN) to predict the collison cross section of immunopeptides based on the extracted features.

The results are evaluated through a) R-squared measurements, b) Residual density, c) Learning rates and d) SHAP value measurements. 

This model was inspired by the PPLN model developed by Nakai et al.


###Previous version:
####This repository also contains:
1) Data: Immunopeptide data derived from initially 2 public datasets: PXD038782 (TOF-IMS/Orbitrap data) and JPST002044 (timsTOF).
2) Models: PPLN pretrained model and the ESM-1b and 2 models (to be included later).
3) Microsoft BitNet as a model resource optimizer.
4) Links to Colab, where most codes will be run.


