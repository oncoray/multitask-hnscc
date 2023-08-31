import numpy as np
import torch
import pandas as pd

from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test
from lifelines import CoxPHFitter

from sklearn.preprocessing import MinMaxScaler

from survival_plus_x.losses.coxph import neg_cox_log_likelihood,\
    neg_cox_log_likelihood_with_decay
from survival_plus_x.models.vit import vit_output_head
from survival_plus_x.models.model_head import SurvivalModelHead


def predictions_and_labels_from_step_outputs(step_output_list):
    # also convert to numpy

    predictions = []
    event_times = []
    events = []
    patients = []
    for step_dict in step_output_list:
        pred = step_dict["prediction"].cpu().numpy()
        lab = step_dict["label"].cpu().numpy()
        pat = step_dict["patient"]

        predictions.extend(pred)
        event_times.extend(lab[:, 0])
        events.extend(lab[:, 1])
        patients.extend(pat)

    predictions = np.stack(predictions).astype(np.float32)[:, 0]
    event_times = np.stack(event_times).astype(np.float32)
    events = np.stack(events).astype(np.uint8)
    patients = np.array(patients)

    return {
        'prediction': predictions,
        'event_time': event_times,
        'event': events,
        'patient': patients
    }


def compute_cindex(pred_and_label_dict):

    try:
        c_index = concordance_index(
            event_times=pred_and_label_dict["event_time"],
            predicted_scores=pred_and_label_dict["prediction"],
            event_observed=pred_and_label_dict["event"])
    except Exception as e:
        print("[WW]: Concordance index could not be computed!", e)
        c_index = np.nan

    return c_index


def compute_stratification_logrank_pvalue(pred_and_label_dict,
                                          cutoff,
                                          alpha=0.95):
    """
    Returns
    -------
    p_value of difference between risk groups obtained by using given cutoff
        """
    preds = pred_and_label_dict["prediction"].to_numpy()
    times = pred_and_label_dict["event_time"].to_numpy()
    events = pred_and_label_dict["event"].to_numpy()

    for arr in [preds, times, events]:
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1

    low_risk_idx = np.where(preds <= cutoff)[0]
    high_risk_idx = np.where(preds > cutoff)[0]

    print(f"Using cutoff {cutoff}, n_low={len(low_risk_idx)}, "
          f"n_high={len(high_risk_idx)}.")

    try:
        test_res = logrank_test(
            times[low_risk_idx],
            times[high_risk_idx],
            event_observed_A=events[low_risk_idx],
            event_observed_B=events[high_risk_idx],
            alpha=alpha)

        p_val = test_res.p_value
    except Exception as e:
        print("[WW]: Caught exception when computing logrank pvalue!", e)
        p_val = np.nan

    return p_val


def metrics_from_step_outputs(step_output_list):
    # also convert to numpy

    pred_and_label_dict = predictions_and_labels_from_step_outputs(
        step_output_list)

    c_index = compute_cindex(pred_and_label_dict)

    metrics = {'c_index': c_index}
    predictions = pd.DataFrame(pred_and_label_dict)

    return metrics, predictions


def init_memory_bank_from_volume_model(train_dataset,
                                       val_dataset,
                                       normalize_pred_range=(-1, 1),
                                       output_dir=None):
    """
    When initialising the memory bank we fit a univariate Cox model
    on the tumor volume and estimate log hazards from that in order
    to not start from fully random predictions.

    Parameters
    ----------
    train_dataset: CancerDatasetInMemory instance with training patients
    val_dataset: CancerDatasetInMemory instance with validation patients
    normalize_pred_range: the range to normalize the predicted log hazards for in order to match the outputs of a Cox model later on. If None, no normalization
    will be performed.
    """

    print("Fitting univariate Cox model based on volume for initialising memory bank!")

    # 1) get the volume of the training patients
    # 2) get the volume of the validation patients
    train_vol_df = train_dataset.get_patient_info_as_df()
    val_vol_df = val_dataset.get_patient_info_as_df()

    # 3) fit a cox model
    fitter = CoxPHFitter()
    time_col = train_dataset.dataset.outcome_file_time_column
    event_col = train_dataset.dataset.outcome_file_event_column
    fitter.fit(
        df=train_vol_df,
        duration_col=time_col,
        event_col=event_col)

    fitter.print_summary()

    # 4) predict hazard for training patients and store to dict
    # 5) predict hazard for validation patients and store to dict
    pred_col = "log_hazard"
    train_haz = fitter.predict_log_partial_hazard(
        train_vol_df).astype(np.float32).to_frame(pred_col)
    val_haz = fitter.predict_log_partial_hazard(
        val_vol_df).astype(np.float32).to_frame(pred_col)

    if normalize_pred_range is not None:
        assert isinstance(normalize_pred_range, (list, tuple))
        assert len(normalize_pred_range) == 2

        scaler = MinMaxScaler(normalize_pred_range)
        train_haz = pd.DataFrame(scaler.fit_transform(
            train_haz), index=train_haz.index, columns=train_haz.columns)
        val_haz = pd.DataFrame(scaler.transform(
            val_haz), index=val_haz.index, columns=val_haz.columns)

    train_c = compute_cindex({
        "event": train_vol_df[event_col],
        "event_time": train_vol_df[time_col],
        "prediction": train_haz
    })
    val_c = compute_cindex({
        "event": val_vol_df[event_col],
        "event_time": val_vol_df[time_col],
        "prediction": val_haz
    })

    print(
        f"\nVolume Model\nC-index (Training)={train_c}\nC-index (Validation)={val_c}\n")

    memory_bank_init_dict = {}
    for pat, row in train_haz.iterrows():
        memory_bank_init_dict[pat] = row[pred_col]
    for pat, row in val_haz.iterrows():
        memory_bank_init_dict[pat] = row[pred_col]

    if output_dir is not None:
        print(f"Storing volume-model information to {output_dir}")

        train_vol_df.to_csv(output_dir / "volume-model_training_data.csv")
        val_vol_df.to_csv(output_dir / "volume-model_validation_data.csv")

        train_haz.to_csv(output_dir / "volume-model_training_predictions.csv")
        val_haz.to_csv(output_dir / "volume-model_validation_predictions.csv")

        pd.DataFrame({
            "cohort": ["Training", "Validation"],
            "C-index": [train_c, val_c]
        }).to_csv(output_dir / "volume-model_metrics.csv", index=False)

    return memory_bank_init_dict


class CoxHead(SurvivalModelHead):
    def __init__(self,
                 input_shape,
                 output_activation="linear",
                 nll_on_batch_only=False,
                 average_over_events_only=False,
                 training_labels=None,
                 validation_labels=None,
                 memory_bank_init_dict=None,
                 memory_bank_decay_factor=1.,
                 bias=False,
                 ):

        self.bias = bias
        self.memory_bank_decay_factor = memory_bank_decay_factor
        super().__init__(input_shape=input_shape)

        if output_activation == "linear":
            self.output_activation = torch.nn.Identity()
        elif output_activation == "tanh":
            self.output_activation = torch.nn.Tanh()
        else:
            raise ValueError(
                "Output activation needs to be 'linear' or 'tanh'!")

        # TODO: dont do, only for upstream used models
        # self.save_hyperparameters()

        # store predictions for each patient so we can
        # compute the Cox NLL for the full dataset instead
        # of batchwise (reusing predictions from previous epochs
        # for all patients not in the current batch)
        self.nll_on_batch_only = nll_on_batch_only
        self.average_over_events_only = average_over_events_only

        self.training_labels = training_labels
        self.validation_labels = validation_labels
        if not self.nll_on_batch_only:
            assert self.training_labels is not None, "Need training labels for full dataset loss computation!"
            assert isinstance(self.training_labels, dict)
            assert self.validation_labels is not None, "Need validation labels for full dataset loss computation!"
            assert isinstance(self.validation_labels, dict)

            self.training_ids = sorted(self.training_labels.keys())
            self.validation_ids = sorted(self.validation_labels.keys())
            if not set(self.training_ids).isdisjoint(set(self.validation_ids)):
                raise ValueError(
                    "Training and validation ids seem not disjoint!")

            # initialize random predictions for each sample
            if memory_bank_init_dict is None:
                print("Initialising memory banks with random numbers")

                # note that the initialisation should match the outputs of the final activation,
                # i.e. if we have output_activation = "tanh", we need to initialize with values in (-1, 1)
                # whereas if we have output_activation = "linear", the range of values is unlimited and we
                # could go with a normal distribution
                if output_activation == "linear":
                    def rand_init(pat): return torch.randn(
                        1, requires_grad=False)
                    print("\t Memory bank init from standard normal distribution.")
                elif output_activation == "tanh":
                    # rand is in range (0, 1) so 2*rand - 1 is in range (-1, 1)
                    def rand_init(pat): return 2 * \
                        torch.rand(1, requires_grad=False) - 1
                    print("\t Memory bank init from uniform(-1, 1) distribution.")

                self.training_memory = {
                    pat: rand_init(pat) for pat in self.training_ids}
                self.validation_memory = {
                    pat: rand_init(pat) for pat in self.validation_ids}

            else:
                print("Initialising memory banks from given dict!")
                self.training_memory = {
                    pat: torch.tensor(memory_bank_init_dict[pat]).unsqueeze(0) for pat in self.training_ids
                }
                self.validation_memory = {
                    pat: torch.tensor(memory_bank_init_dict[pat]).unsqueeze(0) for pat in self.validation_ids
                }

            # initialize decay factors
            self.decays = {}
            assert 0 < self.memory_bank_decay_factor <= 1
            for pat in self.training_ids:
                self.decays[pat] = torch.tensor(1.)
            for pat in self.validation_ids:
                self.decays[pat] = torch.tensor(1.)

        else:
            self.training_memory = None
            self.training_ids = None

            self.validation_memory = None
            self.validation_ids = None

    def _create_network(self, input_shape):
        return vit_output_head(
            dim_input=input_shape,
            n_outputs=1,
            bias=self.bias)

    def _latent_to_final_output(self, latent):
        return self.output_activation(latent)

    def _shared_step(self, memory, patient_ids, patient_labels, batch, batch_idx):
        batch_data = self._predict_and_return_dict(batch)

        # print(
        #    f"cox: batch {batch_idx}: {batch_data['label'][:, 1].sum()} / {len(batch_data['label'])} patients with event")

        if not self.nll_on_batch_only:
            # Compute the ordering based Cox NLL for all samples in the
            # dataset by using predictions of previous epochs for all
            # samples not in the current batch.
            patients_in_batch = batch_data["patient"]

            # and compute the cox likelihood on the full dataset
            # but enable gradients only for the ones from the current batch
            dataset_labels = []
            dataset_preds = []
            dataset_decays = []
            for pat in patient_ids:
                dataset_labels.append(
                    torch.Tensor(patient_labels[pat]))
                # now take old (detached) tensors from the queue if they
                # dont belong to the current batch or the current value
                # if in the batch
                if pat in patients_in_batch:
                    idx = patients_in_batch.index(pat)
                    pred = batch_data["prediction"][idx]
                    # reset the decay to one
                    self.decays[pat] = torch.tensor(1.)
                else:
                    pred = memory[pat].to(self.device)
                    # downweight the prediction further since it got "older"
                    self.decays[pat] = self.memory_bank_decay_factor * \
                        self.decays[pat]

                dataset_decays.append(self.decays[pat])
                dataset_preds.append(pred)

            # n_samples x 2
            dataset_labels = torch.stack(dataset_labels).to(self.device)
            # print("device for labels", dataset_labels.device)
            # n_samples
            dataset_preds = torch.cat(dataset_preds)

            dataset_decays = torch.stack(dataset_decays).to(self.device)
            # print("device for dataset_preds", dataset_preds.device)

            loss = neg_cox_log_likelihood_with_decay(
                labels=dataset_labels,
                risk=dataset_preds,
                decays=dataset_decays,
                reduction_fn=torch.mean,
                average_over_events_only=self.average_over_events_only)

            # update the predictions of the current batch
            # within the memory, leaving all others untouched
            # but also take care that we do not require gradient
            # here any more
            for i, pat in enumerate(patients_in_batch):
                memory[pat] = batch_data["prediction"][i].detach()
        else:
            loss = neg_cox_log_likelihood(
                labels=batch_data["label"],
                risk=batch_data["prediction"],
                reduction_fn=torch.mean,
                average_over_events_only=self.average_over_events_only)

        batch_data["loss"] = loss

        # we need to detach the predictions
        batch_data["prediction"] = batch_data["prediction"].detach()
        batch_data["label"] = batch_data["label"].detach()

        return batch_data

    def training_step(self, batch, batch_idx):
        # override behaviour of base_class

        batch_data = self._shared_step(
            self.training_memory,
            self.training_ids,
            self.training_labels,
            batch, batch_idx)

        # self.log("train_loss", batch_data["loss"],
        #          on_step=False,
        #          logger=True,
        #          prog_bar=True,
        #          on_epoch=True,
        #          sync_dist=True)

        return batch_data

    def validation_step(self, batch, batch_idx):
        # override behaviour of base_class

        batch_data = self._shared_step(
            self.validation_memory,
            self.validation_ids,
            self.validation_labels,
            batch, batch_idx)

        # self.log("val_loss", batch_data["loss"],
        #          on_step=False,
        #          logger=True,
        #          prog_bar=True,
        #          on_epoch=True,
        #          sync_dist=True)

        return batch_data

    def _compute_metrics(self, step_output_list):
        metrics, _ = metrics_from_step_outputs(step_output_list)
        return metrics

    @staticmethod
    def add_model_specific_args(parent_parser):
        cox_group = parent_parser.add_argument_group("Cox model")
        cox_group.add_argument(
            "--nll_on_batch_only",
            action="store_true",
            default=False
        )
        cox_group.add_argument(
            "--average_over_events_only",
            action="store_true",
            default=False,
            help="In Cox loss divide the NLL over the number of events in batch. "
                 "If not provided, divide the NLL by the number of samples in batch."
        )
        cox_group.add_argument(
            "--output_activation",
            type=str,
            choices=["linear", "tanh"],
            help="Activation applied to the model output."
        )
        cox_group.add_argument(
            "--memory_bank_decay_factor",
            type=float,
            default=1.,
            help="How to decay predictions in memory bank that have not been updated recently."
                 "Has to be between 0. (exclusive) and 1. (inclusive)"
        )
        cox_group.add_argument(
            "--memory_bank_init_from_volume",
            action="store_true",
            default=False,
            help="When given, memory bank is not initialized randomly but from Cox model predictions "
                 "obtained from fitting a univariate model based on tumor volume."
        )

        return parent_parser
