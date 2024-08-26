# PLImageReconstruction

## LabNotebook
.tex files containing results and plots

## Utils
Common functions used in different notebooks
- **ami_analysis_constants.py**: Constants and paths used in Adjusted Mutual Information analysis
- **configurations.py**: Configuration templates for model training
- **constants.py**: Constants and paths used in SLM Reconstruction
- **data_utils.py**: Functions to process and generate data
- **lantern_fiber_utils.py**: Functions that model the photonic lantern.
- **minidataset_constants.py**: Constants and paths used in the Clustering Algorithms analysis
- **modeling_utils.py**: Functions to instantiate models from configurations.py, save models, load models and plot training histories.
- **nmi_analysis_constants**: Constants and paths used in Normalized Mutual Information analysis (ami_analysis_constants.py is the one to use, NMI did not yield good results).
- **plot_utils.py**: Functions to plot distributions, clusters, PSFs...
- **psf_constants.py**: Constants and paths used in PSF Reconstruction and Euclidean Distances analysis.
- **zernike_psf_utils.py**: Utils related to zernike and psf (data_utils.py is the main module used)

## AmpPhaseReconstruction
A set of notebooks to train neural networks that reconstruct SLM data from PL outputs
- **AmplitudePhaseReconstructionConvolutional**: Notebook to train convolutional neural networks
- **AmplitudePhaseReconstructionFullyConnected**: Notebook to train fully connected networks
- **AutoEncoderTraining**: Notebook to train a flux autoencoder
- **DataAnalysis**: Plots of data distributions and MSE evolution from the models
- **DataComparison**: Notebook to compare original and processed data
- **DataProcessing**: Normalization, reshaping, padding of data
- **EncoderConvolutionalTraining**: Notebook to train encoder+convolutional neural networks
- **Evaluation notebook**: Notebook to  evaluate models and plot output examples
