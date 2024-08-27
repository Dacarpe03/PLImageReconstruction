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

## PSFReconstruction
Folder containing notebooks for multiple projects (PSF Reconstruction, Clustering, Euclidean distances, AMI analysis)

### DataNotebooks
- **Clustering.ipynb**: Cluster of zernike generated PSFs, zernike coefficients, lp coefficients and PL outputs
- **CorrelationTest.ipynb**: Analysis of intensity sumatory of psf vs LP coefficients L2SUM
- **CreateArbitraryMatrix.ipynb**: Notebook to create an arbitrary transfer matrix with a high condition number.
- **DimensionalityReduction**: Notebook to create UMAPS and PCA of different datasets
- **ExampleCalcCouplingArbinputCube.ipynb**: An example to test the PL lantern module.
- **ExampleUseMat.ipynb**: An example on how to use a transfer matrix to compute PL outputs.
- **HcipyTutorial.ipynb**: A tutorial on how to create PSFs with Hcipy
- **LPModeCoefficientAnalysis**: A data analysis on the LPCoefficients
- **LastDanceAMIAnalysisOverNClusters.ipynb**: Adjusted Mutual Information analysis on datasets of different sizes generated with 9 zernike modes.
- **LowOrderZernikePLInformationDetermination.ipynb**: Euclidean distances analysis for zernike generated datasets.
- **LowOrderZernikePSFGeneration.ipynb**: Notebook to generate train, validation, and test data using zernike modes.
- **MinidatasetnModesClustering.ipynb**: Clustering analysis for minidatasets
- **MinidatasetDimensionalityReduction.ipynb**: Notebook to create UMAPS and PCAs for minidatasets used in Clustering analysis.
- **MinidatasetProcessing.ipynb**: PSF intensity computing and flattening of data
- **MinidatasetZernikePSFGeneration.ipynb**: Notebook to generate minidatasets for Clustering analysis.
- **NMIAnalysisDatasetClustering.ipynb**: Notebook to cluster the datasets related to NMI analysis.
- **NMIAnalysisDatasetDimensionalityReduction**: Notebook to create UMAP and PCAs for the NMI analysis related datasets.
- **NMIAnalysisDatasetZernikePSFGeneration-Copy1.ipynb**: Notebook used to generate datasets for NMI analysis.
- **NMIAnalysisOverNClusters.ipynb**: Notebook to plot cluster densities and NMI evolution.
- **NMIAnalysisdatasetProcessing.ipynb**: Notebook to preprocess NMI analysis related data
- **NormalizedMutualInformation.ipynb**: First test of NMI using clusters from DBSCAN algorithm
- **PLInformationDetermination.ipynb**: Notebook to analyze Euclidean distances for atmospheric aberrated PSFs.
- **PSFGeneration.ipynb**: Notebook to generate data related to the PSF Reconstruction models
- **PSFProcessing.ipynb**: Notebook to preprocess data related to the PSF Reconstruction models
- **PredictionsGeneration.ipybn**: Notebook to generate big datasets of predicted outputs from the models
