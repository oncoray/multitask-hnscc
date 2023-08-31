import pytorch_lightning as pl
import torch

from monai.networks.blocks import UnetOutBlock, UnetrBasicBlock,\
    UnetrUpBlock, UnetrPrUpBlock
from monai.losses import DiceLoss

from survival_plus_x.models.vit import ViT3D
from survival_plus_x.models.multitask import MultitaskHead,\
    multitask_metrics_from_step_outputs
from survival_plus_x.models.densenet import DenseNet


def proj_feat(x, hidden_size, feat_size):
    """
    Reconvert the Attention layer output to an 'image' format
    that can be processed by conv layers.

    Input
    -----
    x: batch of token features of shape B x NToken x D
    hidden_size: dimensionality of each token (D)
    feat_size: number of patches per spatial dimension (NZ, NY, NX)

    Returns
    -------
    Spatial unfolding of the tokens where the feature dimension
    is interpreted as channels, i.e. B x D x NZ x NY x NX
    """

    new_view = (x.size(0), *feat_size, hidden_size)
    x = x.view(new_view)
    new_axes = (0, len(x.shape) - 1) + \
        tuple(d + 1 for d in range(len(feat_size)))
    x = x.permute(new_axes).contiguous()
    return x


class MultitaskPlusUNETR(pl.LightningModule):
    """
    Similar implementation as for Deep-MTS (https://arxiv.org/pdf/2109.07711v1.pdf),
    but using a vision transformer based UNETR as the segmentation model
    and a DenseNet as the survival net.
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
                 # ViT related
                 vit_image_size,
                 vit_patch_size,
                 vit_dim,
                 vit_depth,
                 vit_heads,
                 vit_dim_head,
                 vit_mlp_dim,
                 vit_channels,
                 vit_dropout=0.,
                 vit_emb_dropout=0.,
                 vit_output_token="cls",
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
                 # UNETR related
                 unetr_res_block=True,
                 unetr_conv_block=True,
                 unetr_norm_name="instance",
                 unetr_feature_size=16,  # n filters in the conv part of UNETR
                 unetr_attention_layer_output_idx=[3, 6, 9],
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

        assert vit_output_token in ["cls", "mean"]
        assert vit_depth >= 4, f"UNETR requires at least 4 layers for the VIT, got {vit_depth}"
        assert len(
            unetr_attention_layer_output_idx) == 3, "need to provide 3 indices of which ViT features to use as intermediate output for UNETR"

        # make sure that all indices are INTERMEDIATE outputs, not including
        # the last layers output
        assert all(
            idx < vit_depth - 1 for idx in unetr_attention_layer_output_idx), f"Intermediate transformer layer indices have to be smaller than {vit_depth-1}, but got {unetr_attention_layer_output_idx}."

        # when doing the upsamling we also want to be able to support patch sizes
        # that differ by dimension (e.g. not only 16x16x16), however, that makes
        # behaviour of the models more complicated and we need different upsamling
        # mechanisms
        upsample_kernel_size = [None] * 3
        for i in range(3):
            # in order to obtain back the original shape after upsampling 4 times
            # we need to have upsample_kernel_size[i]^4 == patch_size[i]
            val = int(vit_patch_size[i]**(0.25))
            if val**4 != vit_patch_size[i]:
                raise ValueError(
                    f"Upsampling for UNETR is only possible if the fourth root of patch_size[{i}] is"
                    " an integer. Try patch size 1, 16, 81, 256, ...")
            upsample_kernel_size[i] = val

        self.vit = ViT3D(
            image_size=vit_image_size,
            patch_size=vit_patch_size,
            dim=vit_dim,
            depth=vit_depth,
            heads=vit_heads,
            mlp_dim=vit_mlp_dim,
            pool=None,  # output will always be (batch, n_patches + 1, dim)
            channels=vit_channels,
            dim_head=vit_dim_head,
            dropout=vit_dropout,
            emb_dropout=vit_emb_dropout)

        # UNETR helpers

        # 1 x 1 convolutions to branch from the UNETR encoded features to learn
        # features relevant for survival as well
        self.unetr_feat_extract = torch.nn.ModuleList()

        hidden_size = vit_dim

        # two 3x3 convs and a res conv where input is
        # mapped to correct channels via a 1x1 conv
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=vit_channels,
            out_channels=unetr_feature_size,
            kernel_size=3,
            stride=1,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        # involves 3 transposed convs for an 8x spatial
        # upsampling (one initial, then one per layer),
        # where each layer involves two convs with a skip
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            conv_block=unetr_conv_block,
            res_block=unetr_res_block,
        )
        self.unetr_feat_extract.add_module(
            "surv_unetr_enc2_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=unetr_feature_size * 2,
                    out_channels=unetr_feature_size,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )

        # same as encoder2, but only 4 times spatial upsampling
        # as one layer less
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            conv_block=unetr_conv_block,
            res_block=unetr_res_block,
        )
        self.unetr_feat_extract.add_module(
            "surv_unetr_enc3_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=unetr_feature_size * 4,
                    out_channels=unetr_feature_size,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )
        # essentially only a transposed convolution
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            conv_block=unetr_conv_block,
            res_block=unetr_res_block,
        )
        self.unetr_feat_extract.add_module(
            "surv_unetr_enc4_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=unetr_feature_size * 8,
                    out_channels=unetr_feature_size,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )
        self.unetr_feat_extract.add_module(
            "surv_unetr_dec4_feats", torch.nn.Sequential(
                torch.nn.Conv3d(
                    in_channels=vit_dim,
                    out_channels=unetr_feature_size,
                    kernel_size=1
                ),
                torch.nn.AdaptiveAvgPool3d(1),
                torch.nn.Flatten()
            )
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=unetr_feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size * 8,
            out_channels=unetr_feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size * 4,
            out_channels=unetr_feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size * 2,
            out_channels=unetr_feature_size,
            kernel_size=3,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=unetr_norm_name,
            res_block=unetr_res_block,
        )
        self.unetr_out = UnetOutBlock(
            spatial_dims=3,
            in_channels=unetr_feature_size,
            out_channels=1  # NOTE: we only want to segment tumor, no different parts of it
        )
        # dimensionality of the features for survival coming from the unetr downsampling
        # part
        unetr_feature_dim = len(self.unetr_feat_extract) * unetr_feature_size

        if with_densenet:
            # the Densenet for making survival predictions based on
            # the image and predicted segmentation mask
            self.densenet = DenseNet(
                spatial_dims=3,
                in_channels=vit_channels + 1,  # also the segmentation mask
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
                        f"denseblock{i+1}_batchnorm", torch.nn.BatchNorm3d(channels))

                # only apply 1 x 1 conv of unetr_feature_size
                ops.add_module(
                    f"denseblock{i+1}_feat_extract", torch.nn.Sequential(
                        torch.nn.Conv3d(
                            in_channels=channels,
                            out_channels=unetr_feature_size,
                            kernel_size=1),
                        torch.nn.ReLU(),
                        torch.nn.AdaptiveAvgPool3d(1),
                        torch.nn.Flatten()
                    )
                )
                self.densenet_convs.add_module(
                    f"surv_densenet_feats{i+1}", ops)

            densenet_feature_dim = len(
                self.densenet_convs) * unetr_feature_size
        else:
            self.densenet = None
            densenet_feature_dim = 0

        # we take 4 feature vectors from unetr and apply 1x1 conv of
        # unetr_feature size to give us the first feature set
        # and then also take features from the densenet (optionally)
        surv_feat_dim = unetr_feature_dim + densenet_feature_dim

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

        self.unetr_output_activation = torch.nn.Sigmoid()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.1)

        return optimizer

    def forward(self, x_in):
        _, seg_mask, unetr_feats, densenet_feats = self._forward_without_mt_head(
            x_in)
        # feed the features from DenseNet and UNETR to the multitask head
        if densenet_feats is not None:
            mt_feats = torch.cat([unetr_feats, densenet_feats], dim=1)
        else:
            mt_feats = unetr_feats

        mt_out_dict = self.head(mt_feats)

        mt_out_dict["seg_mask"] = seg_mask

        return mt_out_dict

    def _forward_without_mt_head(self, x_in):
        vit_out, hidden_states_out = self.vit(x_in)
        # our vit always return the class tokens as well which monais vit
        # does not do, so we have to take care of removing that in order to
        # apply the UNETR code from them
        vit_out = vit_out[:, 1:]
        for k in range(len(hidden_states_out)):
            hidden_states_out[k] = hidden_states_out[k][:, 1:]

        # apply the UNETR codes to get the segmentation mask
        feat_size = self.vit.to_patch_embedding.num_patches_per_dim
        enc1 = self.encoder1(x_in)

        x2 = hidden_states_out[
            self.hparams.unetr_attention_layer_output_idx[0]]
        x2_proj = proj_feat(x2, self.hparams.vit_dim, feat_size)
        enc2 = self.encoder2(x2_proj)

        x3 = hidden_states_out[
            self.hparams.unetr_attention_layer_output_idx[1]]
        x3_proj = proj_feat(x3, self.hparams.vit_dim, feat_size)
        enc3 = self.encoder3(x3_proj)

        x4 = hidden_states_out[
            self.hparams.unetr_attention_layer_output_idx[2]]
        x4_proj = proj_feat(x4, self.hparams.vit_dim, feat_size)
        enc4 = self.encoder4(x4_proj)

        dec4 = proj_feat(vit_out, self.hparams.vit_dim, feat_size)

        dec3 = self.decoder5(dec4, enc4)

        dec2 = self.decoder4(dec3, enc3)

        dec1 = self.decoder3(dec2, enc2)

        out = self.decoder2(dec1, enc1)

        seg_logits = self.unetr_out(out)

        seg_mask = self.unetr_output_activation(seg_logits)

        # now apply some convs to the downsampling parts of the UNETR to obtain
        # first part of survival features from enc2, enc3, enc4 and dec4
        # (not enc1 since no vit was involved in its computation)
        unetr_feats = []
        for feat_name, feat_map in zip(["enc2", "enc3", "enc4", "dec4"], [enc2, enc3, enc4, dec4]):
            # apply 1x1 conv, normalisation and relu
            module = self.unetr_feat_extract._modules[f"surv_unetr_{feat_name}_feats"]
            unetr_feats.append(module(feat_map))
        unetr_feats = torch.cat(unetr_feats, dim=1)

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

        return seg_logits, seg_mask, unetr_feats, dn_feats

    def _predict_and_return_dict(self, batch, step_name):
        assert step_name in ["training_step",
                             "validation_step", "predict_step"]

        img = batch["img"]
        seg_logits, seg_mask, unetr_feats, densenet_feats = self._forward_without_mt_head(
            img)

        # feed the features from DenseNet and UNETR to the multitask head
        # and compute survival losses and aggregate them
        if densenet_feats is not None:
            mt_feats = torch.cat([unetr_feats, densenet_feats], dim=1)
        else:
            mt_feats = unetr_feats

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
        unetr_group = parent_parser.add_argument_group("UNETR")
        unetr_group.add_argument(
            "--unetr_norm_name",
            type=str,
            default="instance"
        )
        unetr_group.add_argument(
            "--unetr_feature_size",
            type=int,
            default=16
        )
        unetr_group.add_argument(
            "--unetr_attention_layer_output_idx",
            type=int,
            nargs=3,
            default=[3, 6, 9],
        )
        return parent_parser
