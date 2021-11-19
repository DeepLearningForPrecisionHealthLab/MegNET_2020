# MEGnet source code.
### MEGnet: Automatic ICA-based artifact removal for MEG using spatiotemporal convolutional neural networks. 

### Manuscript available: https://pubmed.ncbi.nlm.nih.gov/34274419/
### DOI: https://doi.org/10.1016/j.neuroimage.2021.118402
### PMID: 34274419

### Uses:
Application of MEGnet to new data:
The main use of this source code is the application of MEGnet to new data. To achive this we suggest starting with 'label_ICA_components.py'. 
This is the most convenient way of deploying MEGnet. It is usable from command line as such: <br>"python label_ICA_components.py --input_path example_data/HCP/100307/@rawc_rfDC_8-StoryM_resample_notch_band/ICA202DDisc --output_dir example_data/HCP/100307/@rawc_rfDC_8-StoryM_resample_notch_band/ICA202DDisc --output_type list"
<br>One subject's ICA components are available in the ./MEGnet/example_data folder, such that the code can be tested.

Please see the comments in the code for additional info and a suggestion of how to efficently integrate it into a pipeline.

Note: If computational efficency is a major concearn, you may want to imbed the function label_ICA_components.fPredictICA directly into your pipeline. This will ensure that tensorflow is only loaded once per run, rather than once per subject.

### Trained Model:
If you only want the final trained model, it is available in both .h5 format to be loaded with keras. This is the suggested method for loading the model. However, sometimes keras can be picky when loading models saved with different versions, to aid in the use of the model with different versions the model configuration and weights are saved as json and npy files respectivley. All can be found in MEGnet/model

### Major Dependencies:
* Cuda11
* Tensorflow2.4
* Tensorflow_addons
* scipy (for loading mat files)

We suggest using the included conda_environment.yml file to create a conda environrment with all nessisary MEGnet dependencies.