import torch
import numpy as np
import pandas as pd

from sksurv.metrics import integrated_brier_score, concordance_index_censored
from sksurv.util import Surv

from survival_plus_x.models.vit import vit_output_head
from survival_plus_x.losses.parametric import neg_log_likelihood
from survival_plus_x.utils.utils import collect_from_batch_outputs
from survival_plus_x.models.model_head import SurvivalModelHead


def metrics_from_step_outputs(step_output_list,
                              timepoints_cindex,
                              timepoints_brier,
                              training_labels):
    """
    For fully parametric models, we can compute metrics
    that are time dependent as well and might give better
    insight into model performance than the c-index that
    requires us to map the predicted distribution again to
    a single number.

    Parameters
    ----------
    timepoint_cindex: float or None
        timepoint from which to get the probability values to compute c-index.
        If None, no c-index will be computed
    timepoints_brier: list of float
        timepoints from which the integrated brier score will be computed.
    labels_train: np.array of shape n_samples x 2
        Where the first column contains event times, the second the event
        indicators (1 for event, 0 for censoring)
    """

    labels = collect_from_batch_outputs(
        step_output_list,
        key="label",
        dtype=np.float32)

    patients = collect_from_batch_outputs(
        step_output_list,
        key="patient",
        to_numpy=False)

    pred_dist_params = collect_lognormal_params_from_batch_outputs(
        step_output_list,
        dtype=np.float32)

    pred_survival_brier = collect_survival_probs_from_batch_outputs(
        step_output_list, timepoints=timepoints_brier,
        dtype=np.float32)

    event_times = labels[:, 0].astype(np.float32)
    events = labels[:, 1].astype(np.uint8)

    survival = Surv.from_arrays(
        event=events,
        time=event_times)

    survival_train = Surv.from_arrays(
        event=training_labels[:, 1].astype(np.uint8),
        time=training_labels[:, 0].astype(np.float32))

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
        'lognormal_loc': pred_dist_params[:, 0],
        'lognormal_scale': pred_dist_params[:, 1],
    }
    for i in range(len(timepoints_brier)):
        k = f"predicted_survival_for_brier_time_{timepoints_brier[i]}"
        pred_dict[k] = pred_survival_brier[:, i]

    if timepoints_cindex is not None:
        for i, timepoint_cindex in enumerate(timepoints_cindex):
            pred_survival_cindex = collect_survival_probs_from_batch_outputs(
                step_output_list, timepoints=[timepoint_cindex],
                dtype=np.float32)[:, 0]

            # NOTE: sksurv cindex is above 0.5 for risk scores that are higher for patients with shorter survival
            # and below 0.5 for survival probabilities which should be higher for patients with longer survival
            # which is why we actually should take 1-the computed value
            c_index = concordance_index_censored(
                event_indicator=events.astype(bool),
                event_time=event_times,
                estimate=pred_survival_cindex)[0]

            metrics[f'c_index_time_{timepoint_cindex}'] = c_index
            k = f"predicted_survival_for_cindex_time_{timepoint_cindex}"
            pred_dict[k] = pred_survival_cindex

    predictions = pd.DataFrame(pred_dict)

    return metrics, predictions


def collect_lognormal_params_from_batch_outputs(step_output_list,
                                                dtype=None):
    values = []
    for step_dict in step_output_list:
        pred_dist = step_dict["prediction"]

        locs = pred_dist.loc.detach().cpu().numpy()
        scales = pred_dist.scale.detach().cpu().numpy()
        params = np.stack([locs, scales], axis=1)

        values.extend(params)

    values = np.stack(values)

    if dtype is not None:
        values = values.astype(dtype)

    # first column are the scale, second the concentrations
    return values


def collect_survival_probs_from_batch_outputs(step_output_list,
                                              timepoints,
                                              dtype=None):
    values = []
    for step_dict in step_output_list:
        pred_dist = step_dict["prediction"]

        # unsqueezing needed for proper broadcasting
        pred_cdf = pred_dist.cdf(
            torch.Tensor(timepoints).unsqueeze(1).to(pred_dist.scale.device))
        # number_of_times x batch_size
        pred_cdf = pred_cdf.detach().cpu().numpy()
        # this is num_timepoints x num_samples and we have to transpose
        # for sksurvival
        pred_s = np.transpose(1. - pred_cdf)

        values.extend(pred_s)

    values = np.stack(values)

    if dtype is not None:
        values = values.astype(dtype)

    return values


class LognormalHead(SurvivalModelHead):
    def __init__(self,
                 input_shape,
                 timepoints_cindex,
                 timepoints_brier,
                 training_labels,
                 ):
        # self.save_hyperparameters()

        self.training_labels = training_labels
        self.timepoints_cindex = timepoints_cindex
        self.timepoints_brier = timepoints_brier

        self.distribution = torch.distributions.LogNormal
        super().__init__(input_shape=input_shape)

    def _create_network(self, input_shape):
        return vit_output_head(
            dim_input=input_shape,
            n_outputs=2,
            bias=False)

    def _latent_to_final_output(self, latent):
        loc = latent[:, 0]  # arbitrary
        scale = torch.nn.functional.softplus(latent[:, 1])  # > 0
        # if torch.any(torch.isnan(scale)):
        #     raise ValueError(
        #         f"Scale: Softplus produced nan ({scale}) with logits {out[:, 1]}.")

        # the output of the model will be distribution objects!
        return self.distribution(loc=loc, scale=scale)

    def _compute_loss(self, batch_data):
        return neg_log_likelihood(
            labels=batch_data["label"],
            predicted_distributions=batch_data["prediction"],
            reduction_fn=torch.mean)

    def _compute_metrics(self, step_output_list):
        metrics, _ = metrics_from_step_outputs(
            step_output_list,
            timepoints_cindex=self.timepoints_cindex,
            timepoints_brier=self.timepoints_brier,
            training_labels=self.training_labels)

        return metrics
