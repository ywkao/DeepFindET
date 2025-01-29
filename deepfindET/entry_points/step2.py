from deepfindET.training_copick import Train
import deepfindET.utils.common as cm
from typing import List, Tuple, Optional
import copick, click, os

def parse_filters(ctx, param, value):
    try:
        # Split the comma-separated string and convert each part to an integer
        filters = [int(x) for x in value.split(",")]
        return filters
    except ValueError:
        raise click.BadParameter("Filters must be a comma-separated list of integers.")

@click.group()
@click.pass_context
def cli(ctx):
    pass

@cli.command()
@click.option(
    "--path-train",
    type=str,
    required=True,
    help="Path to the copick config file (if --path-valid is provided as well, this is the training split).",
)
@click.option(
    "--path-valid",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Path to the copick config file (if provided, this is the validation split).",
)
@click.option(
    "--train-voxel-size",
    type=float,
    required=True,
    help="Voxel size of the tomograms.",
)
@click.option(
    "--train-tomo-type",
    type=str,
    required=True,
    help="Type of tomograms used for training.",
)
@click.option(
    "--target",
    type=(str, str, str),
    required=False,
    default = None,
    help="Tuples of object name, user id and session id (Default for Data Portal: 'data-portal' and).",
    multiple=True,
)
@click.option(
    "--output-path",
    type=str,
    required=True,
    help="Path to store the training results.",
)
@click.option(
    "--n-class",
    type=int,
    required=True,
    help="Number of classes.",
)
@click.option(
    "--model-name",
    type=str,
    required=False,
    default='res_unet',
    show_default=True,
    help="Model Architecture Name to Load For Training",
)
@click.option(
    "--model-pre-weights",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="Pre-Trained Model Weights To Load Prior to Training",
)
@click.option(
    "--model-filters",
    type=str,
    required=False,
    default="48,64,80",
    show_default=True,
    callback=parse_filters,
    help="Comma-separated list of filters for the model architecture.",
)
@click.option(
    "--model-dropout",
    type=float,
    required=False,
    default=0,
    show_default=True,
    help="Dropout for Model Architecture.",
)
@click.option(
    "--dim-in",
    type=int,
    required=False,
    default=52,
    show_default=True,
    help="Patch size.",
)
@click.option(
    "--n-sub-epoch",
    type=int,
    required=False,
    default=10,
    show_default=True,
    help="Number of epochs to train on a subset of the data.",
)
@click.option(
    "--sample-size",
    type=int,
    required=False,
    default=15,
    show_default=True,
    help="Size of the subset of tomos to load into memory.",
)
@click.option(
    "--batch-size",
    type=int,
    required=False,
    default=15,
    show_default=True,
    help="Batch size.",
)
@click.option(
    "--epochs",
    type=int,
    required=False,
    default=65,
    show_default=True,
    help="Number of epochs.",
)
@click.option(
    "--steps-per-epoch",
    type=int,
    required=False,
    default=250,
    show_default=True,
    help="Number of steps per epoch.",
)
@click.option(
    "--n-valid",
    type=int,
    required=False,
    default=20,
    show_default=True,
    help="Number of steps per validation.",
)
@click.option(
    "--target-name",
    type=str,
    required=False,
    default="spheretargets",
    show_default=True,
    help="Name of the segmentation target.",
)
@click.option(
    "--target-user-id",
    type=str,
    required=False,
    default="train-deepfinder",
    show_default=True,
    help="User ID of the segmentation target.",
)
@click.option(
    "--target-session-id",
    type=str,
    required=False,
    default="0",
    show_default=True,
    help="Session ID of the segmentation target",
)
@click.option(
    "--valid-tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="List of validation tomoIDs.",
)
@click.option(
    "--train-tomo-ids",
    type=str,
    required=False,
    default=None,
    show_default=True,
    help="List of training tomoIDs.",
)
@click.option(
    "--class-weights",
    type=(str, float),
    multiple=True,
    required=False,
    default=None,
    show_default=True,
    help="Class weights.",
)
def train(
    path_train: str,
    train_voxel_size: int,
    train_tomo_type: str,
    target: List[Tuple[str, str, str]],
    output_path: str,
    n_class: int,
    model_name: str,
    model_pre_weights: str,
    model_filters: List[int],
    model_dropout: float,
    path_valid: str = None,
    dim_in: int = 52,
    n_sub_epoch: int = 10,
    sample_size: int = 15,
    batch_size: int = 15,
    epochs: int = 65,
    steps_per_epoch: int = 250,
    n_valid: int = 20,
    target_name: str = "spheretargets",
    target_user_id: str = "train-deepfinder",
    target_session_id: str = "0",    
    valid_tomo_ids: str = None,
    train_tomo_ids: str = None,
    class_weights: Optional[List[Tuple[str,float]]] = None,
    ):

    train_model(path_train, train_voxel_size, train_tomo_type, target, output_path, 
                model_name, model_pre_weights, n_class, path_valid, dim_in, n_sub_epoch, 
                sample_size, batch_size, epochs, steps_per_epoch, n_valid, model_filters, 
                model_dropout, target_name, target_user_id, target_session_id, valid_tomo_ids, 
                train_tomo_ids, class_weights)

def train_model(
    path_train: str,
    train_voxel_size: int,
    train_tomo_type: str,
    target: List[Tuple[str, str, str]],
    output_path: str,
    model_name: str,
    model_pre_weights: str,
    n_class: int,
    path_valid: str = None,
    dim_in: int = 52,
    n_sub_epoch: int = 10,
    sample_size: int = 15,
    batch_size: int = 15,
    epochs: int = 65,
    steps_per_epoch: int = 250,
    n_valid: int = 20,
    model_filters: List[int] = [48, 64, 80],
    model_dropout: float = 0,
    target_name: str = "spheretargets",
    target_user_id: str = "train-deepfinder",
    target_session_id: str = "0",    
    valid_tomo_ids: str = None,
    train_tomo_ids: str = None,
    class_weights: Optional[List[Tuple[str,float]]] = None,        
    lr_scheduler: str = "default", # "exp_decay" | "cosine_decay" | "linear_decay" | "cosine_restart" | "cosine_restart_12_3"
    learning_rate: float = 0.0001,
    optimizer: str = "Adam",
    ):

    # Parse input parameters
    if valid_tomo_ids is not None:
        valid_tomo_ids = valid_tomo_ids.split(",")
    if train_tomo_ids is not None:
        train_tomo_ids = train_tomo_ids.split(",")

    # Copick Input Parameters
    trainVoxelSize = train_voxel_size
    trainTomoAlg = train_tomo_type

    # Input parameters:
    Nclass = n_class

    # Initialize training task:
    trainer = Train(Ncl=Nclass,
                    dim_in=dim_in,
                    total_epochs=epochs,
                    constant_epochs=5,
                    learning_rate=learning_rate,
                    final_learning_rate=0.00001,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler)
    trainer.path_out = output_path  # output path
    trainer.batch_size = batch_size
    trainer.epochs = epochs
    trainer.steps_per_epoch = steps_per_epoch
    trainer.Nvalid = n_valid  # steps per validation

    batch_bootstrap = True
    direct_read = False    

    # Segmentation Target Name And Corresponding UserID
    trainer.labelName = target_name
    trainer.labelUserID = target_user_id
    trainer.sessionID = target_session_id

    # Assign Class Weights
    if class_weights is not None:   trainer.create_class_weights(class_weights, path_train)
      
    # Load Specified Model Architecture and Potential Pre-Trained Weights
    trainer.load_model(model_name, model_pre_weights, model_filters, model_dropout)

    # A Certain Number of Tomograms are Loaded Prior to Training (sample_size)
    # And picks from these tomograms are trained for a specified number of epochs (NsubEpoch)
    trainer.NsubEpoch = n_sub_epoch
    trainer.sample_size = sample_size

    # Assign targets as None if None Are Provided - Will Query from config file
    if not target:
        targets = None
    else:
        targets = {}
        for t in target:
            info = {
                "user_id": t[1],
                "session_id": t[2],
            }
            targets[t[0]] = info
    trainer.targets = targets

    # Create output Path
    os.makedirs(output_path, exist_ok=True)

    # Copick Input Parameters
    trainer.voxelSize = trainVoxelSize
    trainer.tomoAlg = trainTomoAlg

    # Finally, launch the training procedure:
    if path_valid is None and valid_tomo_ids is None and train_tomo_ids is None:
        # Option 1:
        # Split the Entire Copick Project into Train / Validation / Test
        tomo_ids = [r.name for r in copick.from_file(path_train).runs]
        (trainList, validationList, testList) = cm.split_datasets(
            tomo_ids,
            train_ratio=0.9,
            val_ratio=0.10,
            test_ratio=0.1,
            savePath=output_path,
        )
        # Swap if Test Runs is Larger than Validation Runs
        if len(testList) > len(validationList):
            testList, validationList = validationList, testList

        # Pass the Run IDs to the Training Class
        trainer.validTomoIDs = validationList
        trainer.trainTomoIDs = trainList

        trainer.launch(path_train)

    elif path_valid is None and valid_tomo_ids is not None and train_tomo_ids is not None:
        # Option 2:
        # train and valid tomoIDs are provided
        trainer.trainTomoIDs = train_tomo_ids
        trainer.validTomoIDs = valid_tomo_ids

        trainer.launch(path_train)

    elif path_valid is not None:
        # Option 3:
        # The Data is Already Split into two Copick Projects

        # Train
        trainer.launch(path_train, path_valid)
