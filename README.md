# Survival_plus_x: Multitask approaches for outcome modelling of head and neck cancer

This is the source code accompanying our manuscript entitled "Multitask learning with convolutional neural networks and vision transformers can improve
outcome prediction for head and neck cancer patients" (submitted to Cancers).

## Installation

Our code was developed and tested on a linux system, so instructions on setup will assume such an OS.
Create a virtual environment for this project, e.g. using anaconda (python virtualenvs are also fine) via

```
conda create -n survival_plus_x python=3.8.12
```
Once the environment got created, activate it with
```
conda activate survival_plus_x
```
The commandline prompt should now have changed slightly to read
```
(survival_plus_x) yourname@yourmachine:~$
```
indicating that the environment is active.

Install dependencies and the project itself (note that this will install the dependencies into your virtual environment only, not systemwide) via

First we install pytorch and pytorch lightning, as this might require specific versions and commands for getting gpu support and compatibility with our code:

```
# pytorch 1.11.0
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# pytorch lightning 1.6.0
pip install pytorch_lightning==1.6.0
```
```
pip install -r requirements.txt
# installs our package
pip install -e .
```

## Usage

All relevant scripts are located in the `scripts/multitask` directory.

### Model training
To train a model on a single split of training/validation data as done in our experiments during cross-validation, execute the script (remember to have the conda environment activated)

```
./run_training.sh
```

The script is pre-configured to run with the HECKTOR2021 dataset. In order to adapt it to work with your data, you might need to adapt the variables defined in the bash script:

- ID: an integer for specifying the run index of the experiments (in our case of 3 repetitions of 5 fold cross-validation, it is a number between 0 and 14) which can be split into a repetition and fold index.
- DATA_DIR: the directory that contains the outcome file in form of a csv and the image files for all experiments
- IMG_DIR: a directory containing subdirectories for each patient, named according to the patients ID. Within each patient subdirectory, numpy array files in the form of `*.npy` files have to be available for the image and segmentation mask. For all patients, the same naming scheme is assumed.
- IMAGE_NAME: name of the numpy file that contains the image input data (like the CT/PET or PET-CT image in concatenated form). Assumed to be the same name for each patient
- MASK_NAME: name of the numpy file that contains the binary tumor segmentation data. Assumed to be the same name for each patient
- OUTCOME_FILE: path to the csv file that contains columns for the patient id, the event status (1=event, 0=censored) and the event time in months
- OUTCOME_SEP: column separator of the above file
- OUTCOME_ID_COL: column name within the above csv file for the column that contains patient ids
- OUTCOME_EVENT_COL: column name within the above csv file for the column that contains event status of each patient
- OUTCOME_TIME_COL: column name within the above csv file for the column that contains the time to event or censoring for each patient
- OUTPUT_DIR: path to where model checkpoints, patient predictions and metrics will be written
- TRAIN_ID_FILE/VALID_ID_FILE: path to a csv file that contains the patient ids used for training/validating the model. This csv file must NOT have a header and is expected to have a single column containing the patient names in different rows. We provide the script `create_cv_folds.py` which can be used for this purpose, using stratified cross-validation to ensure that each folds contains about the same fraction between events and censoring cases as the full dataset.
- `VIT_*`: parameters adjusting the ViT architecture
- HEADS_TO_USE: specifies which outcome models to optimize during training. Recommended is either `cox`, `gensheimer` or their combination
- VIT_OR_CNN: switch between `cnn` and `vit` to use the UNET or UNETR variant of our experiments
- UNETR_*: parameters adjusting the UNETR architecture
- IMAGE_SIZE: size of the crops around the tumor center of mass used as model input in Z (height), Y and X dimension
- PATCH_SIZE: patch size for the ViT. Ensure that IMAGE_SIZE is divisible by this number for each dimension
- GENSHEIMER_INTERVAL_BREAKS: the bin borders for specifying the intervals of the Gensheimer model for which the survival function is discretized on and for which the model will make predictions. Has to start with 0.
- TIMEPOINTS_CINDEX: The time bins at which to evaluate the C-index metric of the Gensheimer model
- TIMEPOINTS_BRIER: the time bins of the Gensheimer model that will be included for computing the integrated Brier score for this model

The parameter flags `--with_segmentation_loss` and `--with_densenet` of the `train.py` script will enable training with auxiliary segmentation loss and the densenet component, respectively. Feel free to deactivate either of them by uncommenting.
The remaining flags are not recommended to be changed (except for parameters explained above)

For model inference, the following parameters are important:

- N_SAMPLES_INFERENCE: how many random crops around the tumor center will be created for each patient during inference. The final patient prediction is then aggregated over the predictions of each randomly chosen crop.
- INFERENCE_SAMPLE_AGGREGATION: the way predictions for a patient will be aggregated to a final output
- CKPT_FILE: the file to the model checkpoint which should be used for making predictions during inference.
- STRATIFICATION_CUTOFF_COX: If inference is done on data not used for training the model, a cutoff for the Cox model head should be provided for patient stratification (on the training data: leave this parameter out and the median will be used). In case the Cox model was not selected in HEADS_TO_USE, simply set this to 0, as it will not play a role.

### Model inference

After training a model, its performance can be evaluated using the `inference.py` script. For convenience, this is already done for the training and validation split inside the `run_training.sh` script. Have a look there on how to run model inference.

For an evaluation of the segmentation quality of the predictions, you can run the `segmentation_inference.py` script or diretly execute `run_segmentation_inference.sh` after adjusting the data paths properly. This will provide you with dice scores for each patient's predicted tumor segmentation mask.

### Model ensembling
After training a model architecture in cross-validation mode on different splits of the data, ensemble performance can be evaluated as well, where predictions for each patient are made by all models before being averaged.

This can be done via
```
./run_ensemble_evaluation.sh
```

This script allows you to define multiple experiment directories within the `EXPERIMENTS` variable. Within each directory defined therein, we expect subdirectories for each repetition and fold of the cross-validation, so be sure to run the training script for all data splits and all specified experiments first.

### Ensemble confidence intervals

After obtaining ensemble predictions for each patient and corresponding ensemble metrics, confidence intervals for the ensemble C-indices can be obtained by running the provided R script `ensemble_cindex_confidence_intervals.R`. Note that this relies on the `survcomp` library which should be installed first. See [here](https://bioconductor.statistik.tu-dortmund.de/packages/3.16/bioc/html/survcomp.html) for instructions on how to do that. We used version `1.48.0` of the library and R version `4.2.2`. Before running the script, set the `workdir` variable to point to the directory which contains your experimental subdirectories. In case you did not run all combinations for multi-outcome, model architecture, segmentation loss and densenet block as suggested by the four loops in the scripts, this has to be adapted manually.