import torch
import numpy as np
import pandas as pd

from tqdm import tqdm

from survival_plus_x.data.util import read_outcome, load_image_as_numpy,\
    print_event_censoring_summary


class CancerDataset(object):
    def __init__(self,
                 image_directories,
                 image_filename,
                 mask_filename,
                 patient_ids,
                 outcome_file,
                 outcome_file_sep,
                 outcome_file_id_column,
                 outcome_file_time_column,
                 outcome_file_event_column,
                 transform=None,
                 print_summary=True
                 ):

        self.outcome_file_time_column = None
        self.outcome_file_event_column = None
        if outcome_file is not None:
            outcome_dict = read_outcome(
                outcome_file=outcome_file,
                id_col=outcome_file_id_column,
                time_col=outcome_file_time_column,
                event_col=outcome_file_event_column,
                dropna=True,
                csv_sep=outcome_file_sep,
                event_time_transformation="identity")

            self.outcome_file_time_column = outcome_file_time_column
            self.outcome_file_event_column = outcome_file_event_column

            # assert we have outcome for all requested patients
            if not set(patient_ids).issubset(set(outcome_dict.keys())):
                raise ValueError(
                    "outcome has to be available for all requested "
                    "patient_ids")

            # limit the outcome to the provided patient ids
            self.outcome_dict = {
                k: v for k, v in outcome_dict.items()
                if k in patient_ids}

        else:
            print()
            print("No outcome for patients available!")
            self.outcome_dict = None

        self.transform = transform

        if not isinstance(image_directories, list):
            image_directories = [image_directories]

        # collect the paths to image and mask for
        # each patient that has outcome
        img_path_lookup = {}
        for image_dir in image_directories:
            # assume one directory per patient named by the patient id
            patient_paths = sorted([
                d for d in image_dir.iterdir() if d.is_dir()])

            for patient_path in patient_paths:
                pat_id = patient_path.name
                # now restrict to the patients within the given ids
                if pat_id not in patient_ids:
                    print(f"Skip {patient_path}. {pat_id} not in patient_ids!")
                    continue

                if self.outcome_dict is not None:
                    if pat_id not in self.outcome_dict:
                        print(f"Skip {patient_path}. {pat_id} has no outcome!")
                        continue

                image_path = patient_path / image_filename
                mask_path = patient_path / mask_filename

                if not image_path.exists():
                    print(
                        f"Skip {patient_path}. Image file {image_path} does not exist.")
                    continue
                if not mask_path.exists():
                    print(
                        f"Skip {patient_path}. Mask file {mask_path} does not exist.")
                    continue

                img_path_lookup[pat_id] = {
                    "image": image_path,
                    "mask": mask_path
                }

        self.image_directories = img_path_lookup
        self.patient_ids = sorted(self.image_directories.keys())

        if self.outcome_dict is not None:
            # only keep the outcomes for patients with images
            self.outcome_dict = {
                k: v for k, v in self.outcome_dict.items()
                if k in self.patient_ids}

            if print_summary:
                print_event_censoring_summary(self.outcome_dict)

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pat_id = self.patient_ids[idx]

        img_file = self.image_directories[pat_id]["image"]
        mask_file = self.image_directories[pat_id]["mask"]

        img_np = load_image_as_numpy(img_file).astype(np.float32)
        mask_np = load_image_as_numpy(mask_file).astype(np.uint8)
        print(pat_id, img_np.shape, img_np.dtype, mask_np.shape, mask_np.dtype)

        data_dict = {
            "patient": pat_id,
            "img": torch.tensor(img_np),
            "mask": torch.tensor(mask_np),
        }

        if self.outcome_dict is not None:
            lab = self.outcome_dict[pat_id]
            data_dict["label"] = torch.Tensor(lab)

        if self.transform is not None:
            data_dict = self.transform(data_dict)

        return data_dict


def load_data(cancer_dataset):
    n = len(cancer_dataset)
    list_of_data_dicts = [None] * n

    for idx in tqdm(range(n)):
        list_of_data_dicts[idx] = cancer_dataset[idx]

    return list_of_data_dicts


class CancerDatasetInMemory(object):
    def __init__(self,
                 image_directories,
                 image_filename,
                 mask_filename,
                 patient_ids,
                 outcome_file,
                 outcome_file_sep,
                 outcome_file_id_column,
                 outcome_file_time_column,
                 outcome_file_event_column,
                 preprocess_transform=None,
                 augmentation_transform=None,
                 ):

        self.dataset = CancerDataset(
            image_directories=image_directories,
            image_filename=image_filename,
            mask_filename=mask_filename,
            patient_ids=patient_ids,
            outcome_file=outcome_file,
            outcome_file_sep=outcome_file_sep,
            outcome_file_id_column=outcome_file_id_column,
            outcome_file_time_column=outcome_file_time_column,
            outcome_file_event_column=outcome_file_event_column,
            transform=preprocess_transform)

        self.augmentation_transform = augmentation_transform

        print("Loading data into memory!")
        self.data = load_data(self.dataset)

    def __getitem__(self, idx):
        data_dict = self.data[idx]

        if self.augmentation_transform is not None:
            data_dict = self.augmentation_transform(data_dict)

        return data_dict

    def __len__(self):
        return len(self.data)

    def get_patient_info_as_df(self):
        data = []
        for entry in self.data:

            mask = entry["mask"]
            # assuming isotropic voxel spacing of 1mm^3
            # we compute volume in cm^3
            vol = np.sum(mask.numpy()) / 1.e3

            if "label" in entry:
                t, e = entry["label"].numpy()
            else:
                t, e = np.nan, np.nan

            data.append({
                "patient": entry["patient"],
                "tumor_volume": vol,
                self.dataset.outcome_file_time_column: t,
                self.dataset.outcome_file_event_column: e,
            })

        df = pd.DataFrame(data).set_index("patient")
        # make sure we only use float32, not float64
        df[df.select_dtypes(np.float64).columns] = df.select_dtypes(
            np.float64).astype(np.float32)
        return df


def make_surv_array_gensheimer(t, f, breaks):
    """
    Copy pasted from https://github.com/MGensheimer/nnet-survival/blob/1d728f8c9c4a5f6b886c1910bedf4cf358171dcb/nnet_survival.py#L48

    Transforms censored survival data into vector format that can be used in Keras.
      Arguments
          t: Array of failure/censoring times.
          f: Censoring indicator. 1 if failed, 0 if censored.
          breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
      Returns
          Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
    """
    assert breaks[0] == 0, "First entry must be zero!"

    n_samples = t.shape[0]
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap
    y_train = np.zeros((n_samples, n_intervals * 2))
    for i in range(n_samples):
        if f[i]:  # if failed (not censored)
            # give credit for surviving each time interval where failure time >= upper limit
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks[1:])
            if t[i] < breaks[-1]:  # if failure time is greater than end of last time interval, no time interval will have failure marked
                # mark failure at first bin where survival time < upper break-point
                y_train[i, n_intervals + np.where(t[i] < breaks[1:])[0][0]] = 1
        else:  # if censored
            # if censored and lived more than half-way through interval, give credit for surviving the interval.
            y_train[i, 0:n_intervals] = 1.0 * (t[i] >= breaks_midpoint)

    return y_train


class GensheimerDatasetInMemory(CancerDatasetInMemory):
    def __init__(self,
                 image_directories,
                 image_filename,
                 mask_filename,
                 patient_ids,
                 outcome_file,
                 outcome_file_sep,
                 outcome_file_id_column,
                 outcome_file_time_column,
                 outcome_file_event_column,
                 interval_breaks,
                 preprocess_transform=None,
                 augmentation_transform=None,
                 ):
        super().__init__(
            image_directories=image_directories,
            image_filename=image_filename,
            mask_filename=mask_filename,
            patient_ids=patient_ids,
            outcome_file=outcome_file,
            outcome_file_sep=outcome_file_sep,
            outcome_file_id_column=outcome_file_id_column,
            outcome_file_time_column=outcome_file_time_column,
            outcome_file_event_column=outcome_file_event_column,
            preprocess_transform=preprocess_transform,
            augmentation_transform=augmentation_transform)

        # the labels have been loaded as part of self.data
        # and now we have to create gensheimer labels from it

        self._create_gensheimer_labels(interval_breaks)

    def _create_gensheimer_labels(self, interval_breaks):
        times = []
        events = []
        for data_dict in self.data:
            assert "label" in data_dict
            label_tensor = data_dict["label"]
            times.append(float(label_tensor[0]))
            events.append(int(label_tensor[1]))

        gens_label = make_surv_array_gensheimer(
            np.array(times),
            np.array(events),
            np.array(interval_breaks))

        # update the data with another key
        for idx, _ in enumerate(self.data):
            self.data[idx]["label_gensheimer"] = torch.Tensor(gens_label[idx])
