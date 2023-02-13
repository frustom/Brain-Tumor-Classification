# Brain-Tumor-Classification
Datasets and MATLAB code for Brain Tumor Classification CNNs, using camouflage animal detection transfer learning

These files contain the code for neural network training and data analysis techniques:
  - Camo_classifier8-11 involve training the clear and camo seed networks, as well as the transfer learned ExpClearNet and ExpCamoNet
  - GliomaClassifier1&2 involve training network either w/ or w/ the oligoastrocytoma label
  - TL GliomaClassifier1&2 correspond to transfer learning from camouflage animal detection to brain tumor classification networks
  
  - Glioma PCA is the code that was used to generate the feature spaces of the trained networks
  - ROC ExpT2 and Meningioma contains the code for the ROC curves 
  - Occlusion Sensitivity MRI was used for creating image sensitivity maps of the data
