#!/bin/bash --login

ID=0  # a number between 0 and 14 inclusive to determine the repetition and the fold indices
FOLD=$(( $(( $ID % 5 )) + 1 ))  # remainder gives 0 - 4 but should be 1 - 5, so we add + 1
REP=$(( $ID / 5 ))  # 0 - 2
echo TASK_ID=$ID, rep=$REP, fold=$FOLD

EPOCHS=400
BATCH_SIZE=16
LEARNING_RATE=1.e-4
NUM_WORKERS=0
GPUS=1

DATA_DIR="$HOME/data/hecktor2021"
IMG_DIR=$DATA_DIR/numpy_images

IMAGE_NAME="ct_pt.npy"  # two channel array (channels first, index=0 -> ct)
#IMAGE_NAME="ct.npy"
MASK_NAME="mask.npy"
OUTCOME_FILE=$DATA_DIR/outcome.csv
OUTCOME_SEP=","
OUTCOME_ID_COL="PatientID"
OUTCOME_EVENT_COL="Progression"
OUTCOME_TIME_COL="PFS_months"  # not present in original dataset, was created


# ARGUMENTS FOR THE VIT ARCHITECTURE
VIT_DIM=192
VIT_DEPTH=9
VIT_HEADS=6
VIT_DIM_HEAD=64
VIT_MLP_DIM=768
VIT_EMB_DROPOUT=0.
VIT_DROPOUT=0.

HEADS_TO_USE=("cox" "gensheimer")
VIT_OR_CNN="cnn"

COX_OUTPUT_ACTIVATION="tanh"
UNETR_FEATURE_SIZE=16
UNETR_NORM_NAME="instance"
UNETR_ATTENTION_LAYER_OUTPUT_IDX=(2 4 6)

IMAGE_SIZE=(48 64 64)
PATCH_SIZE=(16 16 16)
if [[ $IMAGE_NAME == "ct_pt.npy" ]]; then
    CHANNELS=2
    MODALITY="petct"
else
    CHANNELS=1
    if [[ $IMAGE_NAME == "ct.npy" ]]; then
        MODALITY="ct"
    else
        MODALITY="pet"
    fi
fi
#IMAGE_SIZE=(48 81 81)
#PATCH_SIZE=(1 81 81)


GENSHEIMER_INTERVAL_BREAKS=(0 6 12 18 24 30 36 48 60 84 120)
TIMEPOINTS_CINDEX=(12 24 36 60)
TIMEPOINTS_BRIER=(6 12 18 24 30 36 48 60)

N_SAMPLES_INFERENCE=8
INFERENCE_SAMPLE_AGGREGATION="mean"

OUTPUT_DIR="$HOME/experiments/hecktor2021/survival_plus_x/$MODALITY/multitask+"$VIT_OR_CNN"_with-seg_with-densenet/"
OUTPUT_DIR=$OUTPUT_DIR/"imagesize-48_64_64"/"dim-"$VIT_DIM/"depth-"$VIT_DEPTH/"heads-"$VIT_HEADS/"dimhead-"$VIT_DIM_HEAD/"mlpdim-"$VIT_MLP_DIM/"patchsize-16_16_16"/"rep_$REP"/"fold_$FOLD"

# TODO: specify your path to the patient id splits
TRAIN_ID_FILE=$DATA_DIR/three_rep_five_fold_cv/rep_$REP/train_ids_fold_$FOLD.csv
VALID_ID_FILE=$DATA_DIR/three_rep_five_fold_cv/rep_$REP/valid_ids_fold_$FOLD.csv


TRAIN_OUTPUT_DIR=$OUTPUT_DIR/"training"
python train.py --input $IMG_DIR\
                 --img_filename $IMAGE_NAME\
                 --mask_filename $MASK_NAME\
                 --image_size ${IMAGE_SIZE[*]}\
                 --patch_size ${PATCH_SIZE[*]}\
                 --channels $CHANNELS\
                 --outcome $OUTCOME_FILE\
                 --outcome_sep $OUTCOME_SEP\
                 --time_col $OUTCOME_TIME_COL\
                 --event_col $OUTCOME_EVENT_COL\
                 --id_col $OUTCOME_ID_COL\
                 --train_id_file $TRAIN_ID_FILE\
                 --valid_id_file $VALID_ID_FILE\
                 --default_root_dir $TRAIN_OUTPUT_DIR\
                 --batch_size $BATCH_SIZE\
                 --num_workers $NUM_WORKERS\
                 --accelerator "gpu"\
                 --devices $GPUS\
                 --max_epochs $EPOCHS\
                 --vit_or_cnn $VIT_OR_CNN\
                 --heads_to_use ${HEADS_TO_USE[*]}\
                 --dim $VIT_DIM\
                 --depth $VIT_DEPTH\
                 --heads $VIT_HEADS\
                 --dim_head $VIT_DIM_HEAD\
                 --mlp_dim $VIT_MLP_DIM\
                 --emb_dropout $VIT_EMB_DROPOUT\
                 --dropout $VIT_DROPOUT\
                 --output_activation $COX_OUTPUT_ACTIVATION\
                 --log_every_n_steps 1\
                 --learning_rate $LEARNING_RATE\
                 --timepoints_brier ${TIMEPOINTS_BRIER[*]}\
                 --timepoints_cindex ${TIMEPOINTS_CINDEX[*]}\
                 --gensheimer_interval_breaks ${GENSHEIMER_INTERVAL_BREAKS[*]}\
                 --unetr_norm_name $UNETR_NORM_NAME\
                 --unetr_feature_size $UNETR_FEATURE_SIZE\
                 --unetr_attention_layer_output_idx ${UNETR_ATTENTION_LAYER_OUTPUT_IDX[*]}\
                 --num_sanity_val_steps 0\
                 --num_best_checkpoints 3\
                 --checkpoint_every_n_epochs 40\
                 --nll_on_batch_only\
                 --balance_nevents\
                 --with_densenet\
                 --with_segmentation_loss\

# NOTE: this might list multiple files if we didnt use the 'head' command

# with head -n 4 | tail -1 we get the worst validation loss
# with head -n 2 | tail -1 we get (last, best val loss) and select best val loss
# with head -n 3 | tail -1 we get second best validation loss
# with head -n 1 | tail -1 we get 'last.ckpt'
CKPT_FILE=$(find $TRAIN_OUTPUT_DIR -type f -name \*.ckpt | sort -n | head -n 2 | tail -1)
INFERENCE_OUTPUT_DIR=$OUTPUT_DIR/"inference_"$N_SAMPLES_INFERENCE"_samples"

INFERENCE_OUT_TRAIN=$INFERENCE_OUTPUT_DIR/"training"
python inference.py --input $IMG_DIR\
                    --img_filename $IMAGE_NAME\
                    --mask_filename $MASK_NAME\
                    --image_size ${IMAGE_SIZE[*]}\
                    --outcome $OUTCOME_FILE\
                    --outcome_sep $OUTCOME_SEP\
                    --time_col $OUTCOME_TIME_COL\
                    --event_col $OUTCOME_EVENT_COL\
                    --id_col $OUTCOME_ID_COL\
                    --test_id_file $TRAIN_ID_FILE\
                    --output_dir $INFERENCE_OUT_TRAIN\
                    --batch_size $BATCH_SIZE\
                    --num_workers $NUM_WORKERS\
                    --gpus $GPUS\
                    --vit_or_cnn $VIT_OR_CNN\
                    --ckpt_file $CKPT_FILE\
                    --n_samples $N_SAMPLES_INFERENCE\
                    --sample_aggregation $INFERENCE_SAMPLE_AGGREGATION\

# We now run the inference on the validation data using the above
# determined cutoff from the training data: Note that the cutoff is the second column of the metrics files
# STRATIFICATION_CUTOFF_COX=0  # NOTE: needed when gensheimer only training
STRATIFICATION_CUTOFF_COX=$(tail -1 $INFERENCE_OUT_TRAIN/cox_metrics.csv | cut -d"," -f2)

INFERENCE_OUT_VALID=$INFERENCE_OUTPUT_DIR/"validation"
python inference.py --input $IMG_DIR\
                    --img_filename $IMAGE_NAME\
                    --mask_filename $MASK_NAME\
                    --image_size ${IMAGE_SIZE[*]}\
                    --outcome $OUTCOME_FILE\
                    --outcome_sep $OUTCOME_SEP\
                    --time_col $OUTCOME_TIME_COL\
                    --event_col $OUTCOME_EVENT_COL\
                    --id_col $OUTCOME_ID_COL\
                    --test_id_file $VALID_ID_FILE\
                    --output_dir $INFERENCE_OUT_VALID\
                    --batch_size $BATCH_SIZE\
                    --num_workers $NUM_WORKERS\
                    --gpus $GPUS\
                    --vit_or_cnn $VIT_OR_CNN\
                    --ckpt_file $CKPT_FILE\
                    --n_samples $N_SAMPLES_INFERENCE\
                    --sample_aggregation $INFERENCE_SAMPLE_AGGREGATION\
                    --stratification_cutoff_cox $STRATIFICATION_CUTOFF_COX\
