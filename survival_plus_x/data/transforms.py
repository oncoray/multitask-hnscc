import numpy as np

from collections import Iterable
from skimage.measure import regionprops
from monai.transforms import Transform, MapTransform, SpatialCrop
from monai.transforms import (
    Compose,
    EnsureTyped,
    ThresholdIntensityd,
    ScaleIntensityd,
    ScaleIntensity,
    Resized,
    RandGaussianNoised,
    RandStdShiftIntensityd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandRotated,
    RandAxisFlipd,
    RandAffined,
    RandSpatialCropd
)


def center_of_mass_largest_component(mask_tensor):

    assert mask_tensor.ndim == 4, f"expected channel-first 3d mask but got shape {mask_tensor.shape}"
    assert mask_tensor.shape[
        0] == 1, f"expected only a single channel but got {mask_tensor.shape[0]}"
    # discard the channel dimension
    mask = mask_tensor.numpy().astype(np.uint8)[0]

    # pick the properties of the first region
    properties = regionprops(mask)
    region_sizes = [p.area for p in properties]

    # np.argsort sorts ascendingly, so the largest index is last
    largest_region_idx = np.argsort(region_sizes)[-1]

    center_of_mass = np.round(
        properties[largest_region_idx].centroid).astype(int)
    return center_of_mass


class SpatialCropAroundMaskCentroidd(MapTransform):
    def __init__(self, keys, roi_size, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.roi_size = roi_size

    def __call__(self, data):
        d = dict(data)
        assert "mask" in d, "Need a mask to compute centroid!"

        mask = d["mask"]
        center_coords = center_of_mass_largest_component(mask)

        cropper = SpatialCrop(roi_center=center_coords, roi_size=self.roi_size)

        for key in self.key_iterator(d):
            d[key] = cropper(d[key])

        return d


class ScaleIntensitySelectedChannels(Transform):
    """
    Allows different intensity scaling for each channel.
    Also supports selecting only a subset of channels, leaving
    the unselected ones untouched.
    """

    def __init__(self,
                 selected_channels,
                 minvs=0.,
                 maxvs=1.):

        assert isinstance(selected_channels, Iterable)
        self.selected_channels = selected_channels
        # if min/max values are not given for each selected channel, use the same for each channel
        if not isinstance(minvs, Iterable):
            minvs = [minvs] * len(self.selected_channels)
        if not isinstance(maxvs, Iterable):
            maxvs = [maxvs] * len(self.selected_channels)

        assert len(minvs) == len(self.selected_channels)
        assert len(maxvs) == len(self.selected_channels)

        self.transforms_per_channel = {}
        for c, cidx in enumerate(self.selected_channels):
            self.transforms_per_channel[cidx] = ScaleIntensity(
                minv=minvs[c], maxv=maxvs[c], channel_wise=False)

    def __call__(self, img):
        """
        Parameters
        ----------
        img: torch.Tensor
            with first dimension being channels,
            followed by spatial dimensions
        """
        res = img.clone().detach()  # creates a copy of the input that we modify
        n_channels = res.shape[0]
        for cidx, channel_transform in self.transforms_per_channel.items():
            if cidx >= n_channels:
                raise ValueError(
                    f"Image has {n_channels} channels, but selected_channels contains index {cidx}.")

            channel_content = img[cidx]
            res[cidx] = channel_transform(channel_content)

        return res


class ScaleIntensitySelectedChannelsd(MapTransform):
    """Version of the above to be applied on dictionaries."""

    def __init__(self,
                 keys,
                 selected_channels,
                 minvs=0.,
                 maxvs=1.,
                 allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

        self.transform = ScaleIntensitySelectedChannels(
            selected_channels, minvs, maxvs)

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.transform(d[key])
        return d


def get_preprocess_transforms(image_size):
    return Compose([
        # make sure we have all tensors
        EnsureTyped(keys=["img", "mask"], data_type="tensor"),

        # rescale the HU of the CT
        # set everything below -200 to -200
        ThresholdIntensityd(keys=["img"], above=True,
                            threshold=-200, cval=-200),
        # set everything above 200 to 200
        ThresholdIntensityd(keys=["img"], above=False,
                            threshold=200, cval=200),

        # rescale CT (first channel) to range 0, 1
        # and leaves a possible second channel (PET) untouched
        # ScaleIntensityd(keys=["img"], minv=0., maxv=1., channel_wise=True),
        ScaleIntensitySelectedChannelsd(
            keys=["img"], selected_channels=[0], minvs=[0], maxvs=[1]),

        # crop the central part of the image
        SpatialCropAroundMaskCentroidd(
            keys=["img", "mask"], roi_size=image_size),

        # resize it to the desired image size
        # in case the cropped region is smaller than the desired
        # size
        Resized(
            keys=["img", "mask"], spatial_size=image_size)

        # restrict to the segmentation mask
        # CropForegroundd(
        #     keys=["img", "mask"],
        #     source_key="mask")
    ])


def get_train_transforms(image_size):
    # NOTE: we assume that during training an inflated
    # crop was chosen in preprocessing and we now use
    # a random crop of the given size as augmentation.

    return Compose([
        RandSpatialCropd(
            keys=["img", "mask"],
            roi_size=image_size,
            random_size=False,
            random_center=True
        ),
        # resize it to the desired image size
        # in case the cropped region is smaller than the desired
        # size
        Resized(
            keys=["img", "mask"], spatial_size=image_size),

        RandAffined(
            keys="img", prob=0.5,
            rotate_range=None,
            shear_range=None,
            translate_range=[25, 25, 25],
            scale_range=None,
            mode="bilinear"),
        RandGaussianNoised(
            keys="img", prob=.5, mean=0, std=0.1),
        RandStdShiftIntensityd(
            keys="img", prob=.5, factors=(-2., 2.)),
        RandAdjustContrastd(
            keys="img", prob=.5, gamma=(.5, 2.5)),
        RandGaussianSmoothd(
            keys="img", prob=0.5,
            sigma_x=(0.25, 0.75),
            sigma_y=(0.25, 0.75),
            sigma_z=(0.25, 0.75)),
        RandRotated(
            keys="img", prob=0.5,
            range_x=np.deg2rad(20),
            # range_y=deg2rad(20),
            # range_z=deg2rad(20),
            mode="bilinear"),
        # not sure if we should flip all axis
        RandAxisFlipd(
            keys="img", prob=0.5),
        # not sure if we should use that at all
        # Rand3DElasticd(
        #     keys="img", prob=0.5,
        #     sigma_range=(5, 7),
        #     magnitude_range=(50, 150),
        #     padding_mode="zeros"),
    ])


def get_train_transforms_segmentation(image_size):
    # NOTE: we assume that during training an inflated
    # crop was chosen in preprocessing and we now use
    # a random crop of the given size as augmentation.

    return Compose([
        RandSpatialCropd(
            keys=["img", "mask"],
            roi_size=image_size,
            random_size=False,
            random_center=True
        ),
        # resize it to the desired image size
        # in case the cropped region is smaller than the desired
        # size
        Resized(
            keys=["img", "mask"], spatial_size=image_size),

        # now the actual "augmentations"
        RandAffined(
            keys=["img", "mask"], prob=0.5,
            rotate_range=None,
            shear_range=None,
            translate_range=[25, 25, 25],
            scale_range=None,
            mode=["bilinear", "nearest"]),
        RandGaussianNoised(
            keys="img", prob=.5, mean=0, std=0.1),
        RandStdShiftIntensityd(
            keys="img", prob=.5, factors=(-2., 2.)),
        RandAdjustContrastd(
            keys="img", prob=.5, gamma=(.5, 2.5)),
        RandGaussianSmoothd(
            keys="img", prob=0.5,
            sigma_x=(0.25, 0.75),
            sigma_y=(0.25, 0.75),
            sigma_z=(0.25, 0.75)),
        RandRotated(
            keys=["img", "mask"], prob=0.5,
            range_x=np.deg2rad(20),
            # range_y=deg2rad(20),
            # range_z=deg2rad(20),
            mode=["bilinear", "nearest"]),
        # not sure if we should flip all axis
        RandAxisFlipd(
            keys=["img", "mask"], prob=0.5),
        # not sure if we should use that at all
        # Rand3DElasticd(
        #     keys="img", prob=0.5,
        #     sigma_range=(5, 7),
        #     magnitude_range=(50, 150),
        #     padding_mode="zeros"),
    ])
