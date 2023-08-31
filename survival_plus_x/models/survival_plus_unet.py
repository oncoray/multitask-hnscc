import pytorch_lightning as pl
import torch
import torch.nn as nn

from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.networks.layers.factories import Conv
from monai.losses import DiceLoss

from survival_plus_x.models.multitask import MultitaskHead,\
    multitask_metrics_from_step_outputs
from survival_plus_x.models.densenet import DenseNet


# note: this differs from the Unet used in Deep-MTS since there they also use
# residual connections within each Down and Upsampling module
class VariableDepthUNet(nn.Module):
    """
    Largely taken from monai.networks.nets.basic_unet and adapted to allow variable
    length and return intermediate outputs
    """

    def __init__(
        self,
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        features=(32, 32, 64, 128, 256, 32),
        act=("LeakyReLU", {"negative_slope": 0.1, "inplace": True}),
        norm=("instance", {"affine": True}),
        bias=True,
        dropout=0.0,
        upsample="deconv",
    ):
        """
        A UNet implementation with 1D/2D/3D supports.
        Based on:
            Falk et al. "U-Net – Deep Learning for Cell Counting, Detection, and
            Morphometry". Nature Methods 16, 67–70 (2019), DOI:
            http://dx.doi.org/10.1038/s41592-018-0261-2
        Args:
            spatial_dims: number of spatial dimensions. Defaults to 3 for spatial 3D inputs.
            in_channels: number of input channels. Defaults to 1.
            out_channels: number of output channels. Defaults to 2.
            features: variable number of integers as numbers of features.
                Defaults to ``(32, 32, 64, 128, 256, 32)``,
                - the first n-1 values correspond to the encoder feature sizes.
                - the last value corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            bias: whether to have a bias term in convolution blocks. Defaults to True.
                According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                if a conv layer is directly followed by a batch norm layer, bias should be False.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        Examples::
            # for spatial 2D
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128))
            # for spatial 2D, with group norm
            >>> net = BasicUNet(spatial_dims=2, features=(64, 128, 256, 512, 1024, 128), norm=("group", {"num_groups": 4}))
            # for spatial 3D
            >>> net = BasicUNet(spatial_dims=3, features=(32, 32, 64, 128, 256, 32))
        See Also
            - :py:class:`monai.networks.nets.DynUNet`
            - :py:class:`monai.networks.nets.UNet`
        """
        super().__init__()

        fea = features
        self.n_levels = len(fea) - 2
        print(
            f"UNet with {self.n_levels} downsampling/upsampling levels.")

        self.down_ops = torch.nn.ModuleList([
            TwoConv(spatial_dims, in_channels,
                    fea[0], act, norm, bias, dropout)
        ])
        self.up_ops = torch.nn.ModuleList()

        self.downsample_feat_maps = fea[1:-1]

        # i.e. fea = [4, | 8, 12, 16, 20, | 9]
        for i in range(self.n_levels):
            down_filters_in = fea[i]
            down_filters_out = fea[i+1]

            # is self.down_{i+1}, i.e. i=0 -> down_1(4->8), ..., i=3 -> down_4(16 -> 20)
            self.down_ops.append(
                Down(spatial_dims, down_filters_in, down_filters_out,
                     act, norm, bias, dropout))

            # is self.upcat_{self.n_levels-i}, i.e. i=0 -> upcat_4, ..., i=3 -> upcat_1
            up_in_chns = fea[i+1]
            up_cat_chns = fea[i]  #
            halves = False
            if i == 0:
                # the first upsampling layer is a bit special
                up_out_chns = fea[-1]
            else:
                # i > 0 -> upcat_{3}, upcat_{2}, upcat_{1}
                up_out_chns = fea[i]

            self.up_ops.append(
                UpCat(spatial_dims, up_in_chns, up_cat_chns, up_out_chns, act, norm, bias, dropout, upsample, halves))
            # the corresponding upsampling layer that gets the output of the downsampling
            # layer and the previous upsampling result
        # NOTE: in forward we have to process the self.up_ops in reverted way (last module first and first last)

        # NOTE: no output activation is applied here
        self.final_conv = Conv["conv", spatial_dims](
            fea[-1], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: input should have spatially N dimensions
                ``(Batch, in_channels, dim_0[, dim_1, ..., dim_N-1])``, N is defined by `spatial_dims`.
                It is recommended to have ``dim_n % 16 == 0`` to ensure all maxpooling inputs have
                even edge lengths.
        Returns:
            A torch Tensor of "raw" predictions in shape
            ``(Batch, out_channels, dim_0[, dim_1, ..., dim_N-1])``.
        """
        downsampling_feats = []
        for i, down in enumerate(self.down_ops):
            x = down(x)
            downsampling_feats.append(x)
            # print(f"Downsampling block {i+1}, output_shape={x.shape}")

        # note: downsampling_feats[0] is from a Conv module, while the rest is from a Down module

        up_feat = downsampling_feats[-1]
        # process up ops the other way around, i.e. start with the last one
        for i, upcat in enumerate(self.up_ops[::-1]):
            if i == 0:
                # the upcat at the bottom of the U
                x = downsampling_feats[-1]
                # the input from downsampling that gets concatenated
                x_e = downsampling_feats[-2]
            else:
                x = up_feat
                # the input from downsampling that gets concatenated
                x_e = downsampling_feats[self.n_levels - i - 1]

            up_feat = upcat(x=x, x_e=x_e)   # u_{self.n_levels - i} is computed
        logits = self.final_conv(up_feat)

        # of the downsampling feats, skip the first element since this is not
        # after downsampling
        downsampling_feats = downsampling_feats[1:]

        return logits, downsampling_feats


class MultitaskPlusUNET(pl.LightningModule):
    """
    Similar implementation as for Deep-MTS (https://arxiv.org/pdf/2109.07711v1.pdf),
    using a Unet the segmentation model and a DenseNet as the survival net.
    """

    def __init__(self,
                 learning_rate,
                 # multitask related
                 # for the lognormal and the gensheimer head
                 timepoints_cindex,
                 timepoints_brier,
                 training_labels,
                 # for the gensheimer
                 gensheimer_interval_breaks,
                 # UNET related
                 unet_in_channels,
                 unet_features_start=8,
                 unet_feature_size_for_surv=32,
                 # multitask related
                 # for the cox head
                 cox_output_activation="tanh",
                 cox_nll_on_batch_only=True,
                 cox_average_over_events_only=False,
                 cox_training_labels=None,
                 cox_validation_labels=None,
                 cox_memory_bank_init_dict=None,
                 cox_memory_bank_decay_factor=1.,
                 cox_bias=False,
                 # choose which heads and which weights
                 heads_to_use=["cox", "lognormal", "gensheimer", "cindex"],
                 # enable/disable parts of the architecture
                 with_segmentation_loss=True,
                 with_densenet=True
                 ):
        """
        Parameters
        ----------
        timepoints_cindex: list of float
            List of timepoints for which the C-index is computed as a metric for
            models that produce survival functions instead of a single risk output
            (namely 'gensheimer' and 'lognormal')
        timepoints_brier: list of float
            List of timepoints over which the integrated_brier_score is computed
            as a metric for models that produce survival functions
            instead of a single risk output
            (namely 'gensheimer' and 'lognormal')
        training_labels: 2D np.array (shape n_patients x 2) with first column containing
            the event times and the second column containing event indicators
            (1=event, 0=censored)
        gensheimer_interval_breaks: list of float
            Timepoints that define the borders of the intervals used for training
            the gensheimer model (for details see https://peerj.com/articles/6257/).
            Note that the first entry has to be 0!
        heads_to_use: list of str
            The losses used for computing the overall survival loss. Options are
            'cox', 'lognormal', 'gensheimer' and 'cindex'. If more than one is
            specified, losses will be added and each receives equal weight.
        with_segmentation_loss: bool
            Activate or deactivate the segmentation loss (BCE+Dice) for the upsampling
            part of the UNETR that gets added to the multitask survival loss.
            Note that even if it is False, the upsampling part is
            still present in the model but might not receive feedback by a loss and hence
            might not produce sensible output (especially when with_densenet=False, where
            only the downsampling part of UNETR remains to produce features for survival
            and receive feedback by a loss, rendering the upsampling part of UNETR irrelevant)
        with_densenet: bool
            Activate or deactivate the densenet model which receives the segmentation
            produced by UNETR upsampling part in addition to the input image to extract
            features that can be passed to the MultitaskSurvival model in addition to the
            features extracted from the UNETR downsampling path. It has been shown
            to work well in Deep-MTS (https://github.com/MungoMeng/Survival-DeepMTS)

        """
        super().__init__()
        self.save_hyperparameters()

        unet_features = [unet_features_start] + [unet_features_start * 2 **
                                                 i for i in range(4)] + [unet_features_start]

        self.unet = VariableDepthUNet(
            spatial_dims=3,
            in_channels=unet_in_channels,
            out_channels=1,
            features=unet_features)

        # 1 x 1 convolutions to branch from the UNET encoded features to learn
        # features relevant for survival as well
        self.unet_feat_extract = torch.nn.ModuleList()
        self.unet_feat_extract.add_module(
            "surv_unet_enc1_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=self.unet.downsample_feat_maps[0],
                    out_channels=unet_feature_size_for_surv,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )
        self.unet_feat_extract.add_module(
            "surv_unet_enc2_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=self.unet.downsample_feat_maps[1],
                    out_channels=unet_feature_size_for_surv,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )
        self.unet_feat_extract.add_module(
            "surv_unet_enc3_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=self.unet.downsample_feat_maps[2],
                    out_channels=unet_feature_size_for_surv,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )
        self.unet_feat_extract.add_module(
            "surv_unet_enc4_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=self.unet.downsample_feat_maps[3],
                    out_channels=unet_feature_size_for_surv,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )

        # dimensionality of the features for survival coming from the unet
        # downsampling part
        unet_feature_dim = len(self.unet_feat_extract) * \
            unet_feature_size_for_surv

        if with_densenet:
            # the Densenet for making survival predictions based on
            # the image and predicted segmentation mask
            self.densenet = DenseNet(
                spatial_dims=3,
                in_channels=unet_in_channels + 1,  # also the segmentation mask
                out_channels=128,  # dimensionality of the output
                init_features=24,  # number of filters in first conv layer
                growth_rate=16,
                block_config=(4, 8, 16),
                bn_size=4,
                class_layer=False,  # no final output is actually needed
                return_intermediate_outputs=True
            )
            # extract features from the densenet to feed into the
            # multihead survival branch
            self.densenet_convs = torch.nn.ModuleList()
            for i in range(len(self.densenet.out_channels_per_block)):
                channels = self.densenet.out_channels_per_block[i]
                ops = torch.nn.Sequential()
                if i < len(self.densenet.out_channels_per_block) - 1:
                    # apply normalisation first, then 1 x 1 conv
                    # (last layer has already been normalised)
                    ops.add_module(
                        f"denseblock{i+1}_batchnorm", torch.nn.BatchNorm3d(
                            channels))

                # only apply 1 x 1 conv of unetr_feature_size
                ops.add_module(
                    f"denseblock{i+1}_feat_extract", torch.nn.Sequential(
                        torch.nn.Conv3d(
                            in_channels=channels,
                            out_channels=unet_feature_size_for_surv,
                            kernel_size=1),
                        torch.nn.ReLU(),
                        torch.nn.AdaptiveAvgPool3d(1),
                        torch.nn.Flatten()
                    )
                )
                self.densenet_convs.add_module(
                    f"surv_densenet_feats{i+1}", ops)

            densenet_feature_dim = len(
                self.densenet_convs) * unet_feature_size_for_surv
        else:
            self.densenet = None
            densenet_feature_dim = 0

        # we take 4 feature vectors from unetr and apply 1x1 conv of
        # unetr_feature size to give us the first feature set
        # and then also take features from the densenet (optionally)
        surv_feat_dim = unet_feature_dim + densenet_feature_dim

        self.head = MultitaskHead(
            input_shape=surv_feat_dim,
            timepoints_cindex=timepoints_cindex,
            timepoints_brier=timepoints_brier,
            training_labels=training_labels,
            gensheimer_interval_breaks=gensheimer_interval_breaks,
            cox_output_activation=cox_output_activation,
            cox_nll_on_batch_only=cox_nll_on_batch_only,
            cox_average_over_events_only=cox_average_over_events_only,
            cox_training_labels=cox_training_labels,
            cox_validation_labels=cox_validation_labels,
            cox_memory_bank_init_dict=cox_memory_bank_init_dict,
            cox_memory_bank_decay_factor=cox_memory_bank_decay_factor,
            cox_bias=cox_bias,
            heads_to_use=heads_to_use,
            # same weight for each loss
            loss_weights={h: 1. for h in heads_to_use}
        )

        self.dice_loss = DiceLoss(
            sigmoid=True,  # we have to manually do the sigmoid beforehand in forward pass then!
            reduction="mean",
        )
        # expects raw unnormalized scores and combines sigmoid + BCELoss for better
        # numerical stability.
        # expects B x C x H x W x D
        self.ce_loss = torch.nn.BCEWithLogitsLoss(
            reduction="mean")

        self.unet_output_activation = torch.nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.1)

        return optimizer

    def forward(self, x_in):
        _, seg_mask, unet_feats, densenet_feats = self._forward_without_mt_head(
            x_in)
        # feed the features from DenseNet and UNETR to the multitask head
        if densenet_feats is not None:
            mt_feats = torch.cat([unet_feats, densenet_feats], dim=1)
        else:
            mt_feats = unet_feats

        mt_out_dict = self.head(mt_feats)

        mt_out_dict["seg_mask"] = seg_mask

        return mt_out_dict

    def _forward_without_mt_head(self, x_in):
        seg_logits, unet_feats = self.unet(x_in)
        seg_mask = self.unet_output_activation(seg_logits)

        # now apply some convs to the downsampling parts of the UNETR to obtain
        # first part of survival features from enc2, enc3, enc4 and dec4
        # (not enc1 since no vit was involved in its computation)
        unet_feats_for_surv = []
        for i, feat_name in enumerate(["enc1", "enc2", "enc3", "enc4"]):
            # apply 1x1 conv, normalisation and relu
            module = self.unet_feat_extract._modules[f"surv_unet_{feat_name}_feats"]
            unet_feats_for_surv.append(module(unet_feats[i]))
        unet_feats_for_surv = torch.cat(unet_feats_for_surv, dim=1)

        if self.densenet is not None:
            # apply the DenseNet survival network based on input image and predicted
            # segmentation mask
            densenet_in = torch.cat([x_in, seg_mask], dim=1)
            _, densenet_intermediate_outs = self.densenet(densenet_in)

            # note that the last intermediate output has already batchnorm applied, but the
            # rest has not
            dn_feats = []
            for i, dn_out in enumerate(densenet_intermediate_outs):
                module = self.densenet_convs._modules[f"surv_densenet_feats{i+1}"]
                dn_feats.append(module(dn_out))

            dn_feats = torch.cat(dn_feats, dim=1)
        else:
            dn_feats = None

        return seg_logits, seg_mask, unet_feats_for_surv, dn_feats

    def _predict_and_return_dict(self, batch, step_name):
        assert step_name in ["training_step",
                             "validation_step", "predict_step"]

        img = batch["img"]
        seg_logits, seg_mask, unet_feats, densenet_feats = self._forward_without_mt_head(
            img)

        # feed the features from DenseNet and UNETR to the multitask head
        # and compute survival losses and aggregate them
        if densenet_feats is not None:
            mt_feats = torch.cat([unet_feats, densenet_feats], dim=1)
        else:
            mt_feats = unet_feats

        head_fn = getattr(self.head, step_name)
        out_dict = head_fn({
            "img_features": mt_feats,
            "label": batch["label"],
            "label_gensheimer": batch["label_gensheimer"],
            "patient": batch["patient"]
        }, batch_idx=0)

        return {
            "survival": out_dict,
            "segmentation": {
                "img": img.detach().cpu().numpy(),
                "mask": batch["mask"].detach().cpu().numpy(),
                "prediction": seg_mask,
                "prediction_logits": seg_logits
            }
        }

    def _shared_step_trainphase(self, batch, batch_idx, train_or_val):
        step_name = {"train": "training_step",
                     "val": "validation_step"}[train_or_val]

        logging_args = dict(
            on_step=False, logger=True, prog_bar=True,
            on_epoch=True, sync_dist=False,
            batch_size=len(batch["patient"]))

        out_dict = self._predict_and_return_dict(batch, step_name)

        surv_loss = out_dict["survival"]["loss"]
        loss = surv_loss

        # compute segmentation loss
        mask = torch.as_tensor(batch["mask"], dtype=torch.float32)
        # Dont do the sigmoid since it will be done by the losses!
        # print("img:", img.shape, img.dtype)
        # print("mask:", mask.shape, mask.dtype, mask.min(), mask.max())
        # print("pred_mask:", pred_mask.shape, pred_mask.dtype, pred_mask.min(), pred_mask.max())
        seg_logits = out_dict["segmentation"]["prediction_logits"]
        ce_loss = self.ce_loss(seg_logits, mask)
        dice_loss = self.dice_loss(seg_logits, mask)
        seg_loss = 0.5 * (ce_loss + dice_loss)

        if self.hparams.with_segmentation_loss:
            loss += seg_loss

        out_dict["segmentation"]["loss"] = seg_loss.detach()
        out_dict["segmentation"]["ce_loss"] = ce_loss.detach()
        out_dict["segmentation"]["dice_loss"] = dice_loss.detach()
        out_dict["segmentation"]["prediction"] = (
            out_dict["segmentation"]["prediction"].detach())
        out_dict["segmentation"]["prediction_logits"] = (
            out_dict["segmentation"]["prediction_logits"].detach())

        out_dict["survival"]["loss"] = out_dict["survival"]["loss"].detach()
        out_dict["loss"] = loss

        # logging
        # a) total loss
        self.log(f"{train_or_val}_loss", loss, **logging_args)
        # b) aggregated multitask survival loss
        self.log(f"surv/{train_or_val}_loss", surv_loss, **logging_args)
        # c) single survival losses
        for head_name in out_dict["survival"]:
            if head_name == "loss":
                continue
            self.log(
                f"surv/{head_name}_{train_or_val}_loss",
                out_dict["survival"][head_name]["loss"],
                **logging_args)
        # d) aggregated segmentation loss
        self.log(f"segmentation/{train_or_val}_loss",
                 seg_loss, **logging_args)
        # e) single segmentation losses
        self.log(f"segmentation/ce_{train_or_val}_loss",
                 ce_loss, **logging_args)
        self.log(
            f"segmentation/dice_{train_or_val}_loss", dice_loss,
            **logging_args)

        return out_dict

    def training_step(self, batch, batch_idx):
        return self._shared_step_trainphase(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step_trainphase(batch, batch_idx, "val")

    def predict_step(self, batch, batch_idx):
        return self._predict_and_return_dict(batch, "predict_step")

    def _shared_epoch_end(self, step_output_list, train_or_val):
        # log segmentation metrics is not necessary since the dice loss
        # is the negative of the dice score and hence we already have a metric

        # names of the survival heads
        names = list(self.head.heads.keys())
        surv_metrics, _ = multitask_metrics_from_step_outputs(
            step_output_list=[d["survival"] for d in step_output_list],
            task_names=names,
            timepoints_cindex=self.hparams.timepoints_cindex,
            timepoints_brier=self.hparams.timepoints_brier,
            training_labels=self.hparams.training_labels,
            gensheimer_interval_breaks=self.hparams.gensheimer_interval_breaks
        )

        logging_args = dict(
            on_step=False, logger=True,
            prog_bar=True, on_epoch=True, sync_dist=False,
            batch_size=len(step_output_list[0]["survival"][names[0]]["patient"]))
        for head_name, ms in surv_metrics.items():
            for m in ms:
                self.log(f"surv/{head_name}_{train_or_val}_{m}",
                         ms[m], **logging_args)

    def training_epoch_end(self, step_output_list):
        self._shared_epoch_end(step_output_list, "train")

    def validation_epoch_end(self, step_output_list):
        self._shared_epoch_end(step_output_list, "val")

    @ staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser = MultitaskHead.add_model_specific_args(parent_parser)
        parent_parser.add_argument(
            "--with_segmentation_loss",
            action="store_true",
            default=False,
            help="Whether to also train a segmentation alongside the survival losses"
        )
        parent_parser.add_argument(
            "--with_densenet",
            action="store_true",
            default=False,
            help="Whether to also use a Densenet CNN for extracting features for survival, "
                 "similar to Deep-MTS."
        )
        unet_group = parent_parser.add_argument_group("UNET")
        unet_group.add_argument(
            "--unet_features_start",
            type=int,
            default=8,
            help="Number of feature maps after first downsampling block of Unet. "
                 "Doubled in each further downsampling block."
        )
        unet_group.add_argument(
            "--unet_feature_size_for_surv",
            type=int,
            default=32,
            help="Number of feature maps to extract from each downsampling step of Unet "
                 "for survival prediction."
        )

        return parent_parser
