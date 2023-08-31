RESULT_DIR="$HOME/experiments/hecktor2021/survival_plus_x/petct"

EXPERIMENTS=(
    "multitask+cnn_with-seg_with-densenet"
    "multitask+cnn_with-seg_without-densenet"
    "multitask+cnn_without-seg_with-densenet"
    "multitask+cnn_without-seg_without-densenet"
    "multitask+vit_with-seg_with-densenet"
    "multitask+vit_with-seg_without-densenet"
    "multitask+vit_without-seg_with-densenet"
    "multitask+vit_without-seg_without-densenet"

    # single task cox
    #"st-cox+cnn_with-seg_with-densenet"
    #"st-cox+cnn_with-seg_without-densenet"
    #"st-cox+cnn_without-seg_with-densenet"
    #"st-cox+cnn_without-seg_without-densenet"
    #"st-cox+vit_with-seg_with-densenet"
    #"st-cox+vit_with-seg_without-densenet"
    #"st-cox+vit_without-seg_with-densenet"
    #"st-cox+vit_without-seg_without-densenet"

    # single task gensheimer
    # "st-gh+cnn_with-seg_with-densenet"
    # "st-gh+cnn_with-seg_without-densenet"
    # "st-gh+cnn_without-seg_with-densenet"
    # "st-gh+cnn_without-seg_without-densenet"
    # "st-gh+vit_with-seg_with-densenet"
    # "st-gh+vit_with-seg_without-densenet"
    # "st-gh+vit_without-seg_with-densenet"
    # "st-gh+vit_without-seg_without-densenet"
)

# NOTE: for single task evaluation (st-cox or st-gh), you have to comment out one call to aggregate_cv and one of the ensemble calls below for the model that did not get included


N_SAMPLES_INFERENCE=8
ENSEMBLE_REDUCTION=mean
# names for ensembling
PRED_TIME_COL="event_time"
PRED_EVENT_COL="event"
PRED_ID_COL="patient"
PRED_PRED_COL_COX="prediction"
PRED_PRED_COL_GH="predict"
OUTCOME_FILE=$HOME/data/hecktor2021/hecktor2021_training_endpoint.csv # needs columns 'PatientID', 'event', 'event_time' for all patients of the training data to compute Brier score for the Gensheimer outputs


for EXPERIMENT in ${EXPERIMENTS[*]}; do
    echo $EXPERIMENT
    OUTPUT_DIR_CV=$RESULT_DIR/$EXPERIMENT/imagesize-48_64_64/dim-192/depth-9/heads-6/dimhead-64/mlpdim-768/patchsize-16_16_16

    # summary of the CV performances

    echo "========COX CV======="
    python aggregate_cv_results.py --result_dir $OUTPUT_DIR_CV --metric_filename "cox_metrics.csv" --output_filename "cox_metrics_on-"$N_SAMPLES_INFERENCE"-samples.csv"

    echo "========GENSHEIMER CV======="
    python aggregate_cv_results.py --result_dir $OUTPUT_DIR_CV --metric_filename "gensheimer_metrics.csv" --output_filename "gensheimer_metrics_on-"$N_SAMPLES_INFERENCE"-samples.csv"

    # ensembling

    OUTPUT_DIR_ENSEMBLE=$OUTPUT_DIR_CV/ensemble_$ENSEMBLE_REDUCTION

    OUTPUT_DIR_ENSEMBLE_TRAIN=$OUTPUT_DIR_ENSEMBLE/training
    OUTPUT_DIR_ENSEMBLE_VAL=$OUTPUT_DIR_ENSEMBLE/validation
    OUTPUT_DIR_ENSEMBLE_TEST=$OUTPUT_DIR_ENSEMBLE/test

    # for COX
    # want the rep_<R>/fold_<F>/inference/training/predictions.csv but exclude the rep_<R>/fold_<F>/inference/validation/predictions.csv
    PRED_FILES_TRAIN=$(find $OUTPUT_DIR_CV -type f -name cox_predictions.csv | grep training)
    python ensemble_prediction_cox.py --prediction_files $PRED_FILES_TRAIN\
                                      --prediction_file_id_col $PRED_ID_COL\
                                      --prediction_file_pred_col $PRED_PRED_COL_COX\
                                      --prediction_file_time_col $PRED_TIME_COL\
                                      --prediction_file_event_col $PRED_EVENT_COL\
                                      --ensemble_reduction_fn $ENSEMBLE_REDUCTION\
                                      --output_dir $OUTPUT_DIR_ENSEMBLE_TRAIN\

    STRATIFICATION_CUTOFF_ENSEMBLE_TRAIN=$(tail -1 $OUTPUT_DIR_ENSEMBLE_TRAIN/cox_metrics.csv | cut -d"," -f2)

    PRED_FILES_VAL=$(find $OUTPUT_DIR_CV -type f -name cox_predictions.csv | grep validation)
    python ensemble_prediction_cox.py --prediction_files $PRED_FILES_VAL\
                                      --prediction_file_id_col $PRED_ID_COL\
                                      --prediction_file_pred_col $PRED_PRED_COL_COX\
                                      --prediction_file_time_col $PRED_TIME_COL\
                                      --prediction_file_event_col $PRED_EVENT_COL\
                                      --ensemble_reduction_fn $ENSEMBLE_REDUCTION\
                                      --output_dir $OUTPUT_DIR_ENSEMBLE_VAL\
                                      --stratification_cutoff $STRATIFICATION_CUTOFF_ENSEMBLE_TRAIN\

    # for gensheimer
    PRED_FILES_TRAIN=$(find $OUTPUT_DIR_CV -type f -name gensheimer_predictions.csv | grep training)
    python ensemble_prediction_gensheimer.py --prediction_files $PRED_FILES_TRAIN\
                                             --prediction_file_id_col $PRED_ID_COL\
                                             --prediction_file_pred_col $PRED_PRED_COL_GH\
                                             --prediction_file_time_col $PRED_TIME_COL\
                                             --prediction_file_event_col $PRED_EVENT_COL\
                                             --ensemble_reduction_fn $ENSEMBLE_REDUCTION\
                                             --output_dir $OUTPUT_DIR_ENSEMBLE_TRAIN\
                                             --train_outcome_file $OUTCOME_FILE\

    # the cutoffs for gensheimer could be many, one for each timepoint so we point to a file
    STRATIFICATION_CUTOFF_ENSEMBLE_TRAIN=$OUTPUT_DIR_ENSEMBLE_TRAIN/gensheimer_stratification_cutoffs.csv

    PRED_FILES_VAL=$(find $OUTPUT_DIR_CV -type f -name gensheimer_predictions.csv | grep validation)
    python ensemble_prediction_gensheimer.py --prediction_files $PRED_FILES_VAL\
                                             --prediction_file_id_col $PRED_ID_COL\
                                             --prediction_file_pred_col $PRED_PRED_COL_GH\
                                             --prediction_file_time_col $PRED_TIME_COL\
                                             --prediction_file_event_col $PRED_EVENT_COL\
                                             --ensemble_reduction_fn $ENSEMBLE_REDUCTION\
                                             --output_dir $OUTPUT_DIR_ENSEMBLE_VAL\
                                             --stratification_cutoff $STRATIFICATION_CUTOFF_ENSEMBLE_TRAIN\
                                             --train_outcome_file $OUTCOME_FILE\

done
