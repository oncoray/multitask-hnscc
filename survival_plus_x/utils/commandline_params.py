from argparse import ArgumentParser


def add_vit_commandline_args(parser):
    vit_group = parser.add_argument_group(
        "VisionTransformer architecture")

    # this will be handled in the config
    # since the transformations will take care of it
    # vit_group.add_argument(
    #     "--image_size",
    #     type=int,
    #     nargs=3,
    #     help="Size of input images (Z, Y, X)",
    # )
    # vit_group.add_argument(
    #     "--patch_size",
    #     type=int,
    #     nargs=3,
    #     help="Size of image region that gets mapped to a single patch with a "
    #     "single embedding (Z, Y, X)",
    # )
    vit_group.add_argument(
        "--patch_size",
        type=int,
        nargs=3,
        # required=True,
        help="Patch size of Vision transformer."
    )
    vit_group.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Number of channels of input images"
    )
    vit_group.add_argument(
        "--dim",
        type=int,
        help="Dimensionality of feature vector for each patch embedding.",
    )
    vit_group.add_argument(
        "--depth", type=int, help="Number of transformer blocks to use."
    )
    vit_group.add_argument(
        "--heads", type=int,
        help="Number of attention heads."
    )
    vit_group.add_argument(
        "--dim_head",
        type=int,
        help="Dimensionality of features processed by each attention head",
    )
    vit_group.add_argument(
        "--mlp_dim",
        type=int,
        help="Dimensionality of MLP within each tranformer block",
    )
    vit_group.add_argument(
        "--pool",
        type=str,
        choices=["cls", "mean"],
        help="Which output token to use after the transformer.",
    )
    vit_group.add_argument(
        "--emb_dropout",
        type=float,
        help="Dropout rate after patch embedding and positional encoding"
    )
    vit_group.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate in transformer blocks."
    )
    vit_group.add_argument(
        "--no_bias_at_output",
        action="store_true",
        default=False,
        help="Disable bias parameter in the output layer of the ViT."
    )

    return parser


def add_common_args(parser):
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        metavar='S',
        help='random seed (default: 1)')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1)
    parser.add_argument(
        '--num_workers',
        type=int,
        default=0)
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.e-4)
    parser.add_argument(
        "--reduce_lr_on_plateau",
        action="store_true",
        default=False,
        help="Whether to reduce learning rate when validation C-index does not improve anymore (in our case reduces)"
    )
    parser.add_argument(
        "--balance_nevents",
        action="store_true",
        default=False,
        help="Try to incorporate about same number of events and censors within each batch. Note this might repeat the same patient multiple times and leave out others during an epoch. Shown to work in DeepMTS paper."
    )

    io_group = parser.add_argument_group("Input/Output")
    io_group.add_argument(
        '--image_size',
        type=int,
        nargs=3,
        required=True,
        help="Size of the Image crop around tumor center of mass."
    )
    io_group.add_argument(
        '--train_id_file',
        type=str,
        help="Full path to a csv file containing the ids used for training.")
    io_group.add_argument(
        '--valid_id_file',
        type=str,
        help="Full path to a csv file containing the ids used for validation during training.")
    io_group.add_argument(
        '--test_id_file',
        type=str,
        help="Full path to a csv file containing the ids used for testing during inference.")
    io_group.add_argument(
        '--input',
        type=str,
        nargs="+",
        help="list of input directories that each contain directories for each patient")
    io_group.add_argument(
        '--outcome',
        type=str,
        help="Path to the outcome file (*.csv)")
    io_group.add_argument(
        '--outcome_sep',
        type=str,
        default=";",
        help="Column separator token for the outcome csv file.")
    io_group.add_argument(
        '--event_col',
        type=str,
        help="Column name within the outcome that contains event marker")
    io_group.add_argument(
        '--time_col',
        type=str,
        help="Column name within the outcome that contains event time")
    io_group.add_argument(
        '--id_col',
        type=str,
        help="Column name within the outcome that contains patient ids")
    io_group.add_argument(
        "--img_filename",
        type=str,
        help="a filename for the images within patient directories of input")
    io_group.add_argument(
        "--mask_filename",
        type=str,
        help="a filename for the segmentation mask within patient directories of input")
    io_group.add_argument(
        "--num_best_checkpoints",
        type=int,
        default=1,
        help="Number of best models to save as checkpoints."
    )
    io_group.add_argument(
        "--checkpoint_every_n_epochs",
        type=int,
        default=None,
        help="Frequency to write out checkpoints."
    )

    return parser


def parser_with_common_args(title):
    parser = ArgumentParser(title)

    parser = add_common_args(parser)
    parser = add_vit_commandline_args(parser)

    return parser
