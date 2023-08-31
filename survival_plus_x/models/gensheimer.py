import pytorch_lightning as pl
import torch
import numpy as np
import pandas as pd

from sksurv.metrics import integrated_brier_score, concordance_index_censored
from sksurv.util import Surv

from survival_plus_x.models.vit import vit_output_head
from survival_plus_x.losses.gensheimer import neg_log_likelihood
from survival_plus_x.utils.utils import collect_from_batch_outputs
from survival_plus_x.models.model_head import SurvivalModelHead


def interpolate_predictions(predictions,
                            interval_breaks_prediction,
                            timepoints_wanted):
    # Predicted survival probability from Nnet-survival model
    # taken from https://github.com/MGensheimer/nnet-survival/blob/1d728f8c9c4a5f6b886c1910bedf4cf358171dcb/nnet_survival.py#L71

    # Inputs are Numpy arrays.
    # y_pred: Rectangular array, each individual's conditional probability of surviving each time interval
    # breaks: Break-points for time intervals used for Nnet-survival model, starting with 0
    # fu_time: Follow-up time point at which predictions are needed
    #
    # Returns: predicted survival probability for each individual at specified follow-up time
    y_pred = np.cumprod(predictions, axis=1)  # N x intervals
    pred_surv = []

    for i in range(y_pred.shape[0]):
        pred_interpolated = np.interp(
            timepoints_wanted, interval_breaks_prediction[1:], y_pred[i])
        pred_surv.append(pred_interpolated)

    return np.array(pred_surv)


def metrics_from_step_outputs(step_output_list,
                              timepoints_cindex,
                              timepoints_brier,
                              training_labels,
                              interval_breaks):
    labels = collect_from_batch_outputs(
        step_output_list,
        key="label",
        dtype=np.float32)

    event_times = labels[:, 0].astype(np.float32)
    events = labels[:, 1].astype(np.uint8)

    survival = Surv.from_arrays(
        event=events,
        time=event_times)

    survival_train = Surv.from_arrays(
        event=training_labels[:, 1].astype(np.uint8),
        time=training_labels[:, 0].astype(np.float32))

    patients = collect_from_batch_outputs(
        step_output_list,
        key="patient",
        to_numpy=False)

    # interpolation of predictions to the requested
    # timepoints of brier
    # NOTE: those will then be unconditional survival probabilities, in contrast
    # to the "predictions"
    predictions = collect_from_batch_outputs(
        step_output_list,
        key="prediction",
        dtype=np.float32)

    pred_survival_brier = interpolate_predictions(
        predictions,
        interval_breaks_prediction=interval_breaks,
        timepoints_wanted=timepoints_brier)

    try:
        integrated_brier = integrated_brier_score(
            survival_train=survival_train,
            survival_test=survival,
            estimate=pred_survival_brier,
            times=timepoints_brier)
    except ValueError as e:
        print("[W]: Failed to compute integrated brier score: ", e)
        integrated_brier = np.nan

    metrics = {
        'integrated_brier_score': integrated_brier,
    }

    pred_dict = {
        'patient': patients,
        'event': events,
        'event_time': event_times,
    }
    for i in range(len(timepoints_brier)):
        k = f"predicted_survival_for_brier_time_{timepoints_brier[i]}"
        pred_dict[k] = pred_survival_brier[:, i]

    # also write out the conditional survival predictions for all intervals
    for i in range(predictions.shape[1]):
        k = f"prediction_{i}"
        pred_dict[k] = predictions[:, i]

    if timepoints_cindex is not None:
        for i, timepoint_cindex in enumerate(timepoints_cindex):
            pred_survival_cindex = interpolate_predictions(
                predictions,
                interval_breaks_prediction=interval_breaks,
                timepoints_wanted=timepoint_cindex
            )

            # NOTE: sksurv cindex is above 0.5 for risk scores that are higher for patients with shorter survival
            # and below 0.5 for survival probabilities which should be higher for patients with longer survival
            # which is why we actually should take 1-the computed value

            c_index = concordance_index_censored(
                event_indicator=events.astype(bool),
                event_time=event_times,
                estimate=pred_survival_cindex)[0]
            c_index = 1. - c_index

            metrics[f'c_index_time_{timepoint_cindex}'] = c_index
            k = f"predicted_survival_for_cindex_time_{timepoint_cindex}"
            pred_dict[k] = pred_survival_cindex

    predictions = pd.DataFrame(pred_dict)

    return metrics, predictions


class GensheimerHead(SurvivalModelHead):
    def __init__(self,
                 input_shape,
                 timepoints_cindex,
                 timepoints_brier,
                 training_labels,
                 interval_breaks,
                 ):
        # self.save_hyperparameters()

        self.training_labels = training_labels
        self.timepoints_cindex = timepoints_cindex
        self.timepoints_brier = timepoints_brier
        self.interval_breaks = interval_breaks

        super().__init__(input_shape=input_shape)
        self.output_activation = torch.nn.Sigmoid()

    def _create_network(self, input_shape):
        return vit_output_head(
            dim_input=input_shape,
            n_outputs=len(self.interval_breaks) - 1,
            bias=False)

    def _latent_to_final_output(self, latent):
        return self.output_activation(latent)

    def _predict_and_return_dict(self, batch):
        # override base class function
        assert "img_features" in batch
        assert "label" in batch
        assert "label_gensheimer" in batch
        assert "patient" in batch

        # tensor with shape B input_shape
        batch_img_feats = batch["img_features"]
        batch_labels = batch["label"]  # tensor with shape B 2 (time, event)
        # see GensheimerDatasetInMemory class
        batch_labels_gh = batch["label_gensheimer"]
        batch_patients = batch["patient"]  # list of str

        # the conditional probabiliy of surviving each time interval, given
        # we survived up to that interval.
        # This is needed for loss computation
        batch_predictions_cond_surv = self.forward(
            batch_img_feats)  # B x n_intervals

        # the unconditional survival probability at each timepoint, B x n_intervals
        batch_predictions_surv = batch_predictions_cond_surv.cumprod(dim=1)

        return {
            "patient": batch_patients,
            "label": batch_labels,
            "label_gensheimer": batch_labels_gh,
            "prediction": batch_predictions_cond_surv,
            "prediction_surv": batch_predictions_surv
        }

    def _compute_loss(self, batch_data):
        return neg_log_likelihood(
            labels=batch_data["label_gensheimer"],
            pred=batch_data["prediction"],
            reduction_fn=torch.mean)

    def _compute_metrics(self, step_output_list):
        metrics, _ = metrics_from_step_outputs(
            step_output_list,
            timepoints_cindex=self.timepoints_cindex,
            timepoints_brier=self.timepoints_brier,
            training_labels=self.training_labels,
            interval_breaks=self.interval_breaks)

        return metrics
