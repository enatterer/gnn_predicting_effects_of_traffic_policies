import os
import sys

import torch

# Add the 'scripts' directory to Python Path
scripts_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
if scripts_path not in sys.path:
    sys.path.append(scripts_path)

from training.help_functions import *

from gnn.help_functions import EIGN_Loss, compute_baseline_of_mean_target, compute_baseline_of_no_policies
from gnn.models.eign import EIGNLaplacianConv

project_root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))

# Please adjust as needed
dataset_path = os.path.join(
    project_root, "data", "train_data", "edge_features_with_net_flow_aggregated"
)
base_dir = os.path.join(project_root, "data")

PARAMETERS = [
    "project_name",
    "predict_mode_stats",
    "in_channels",
    "use_all_features",
    "out_channels",
    "loss_fct",
    "use_weighted_loss",
    "point_net_conv_layer_structure_local_mlp",
    "point_net_conv_layer_structure_global_mlp",
    "gat_conv_layer_structure",
    "use_bootstrapping",
    "num_epochs",
    "batch_size",
    "lr",
    "early_stopping_patience",
    "use_dropout",
    "dropout",
    "gradient_accumulation_steps",
    "use_gradient_clipping",
    "device_nr",
    "unique_model_description",
]


def get_parameters(args):

    params = {
        # KEEP IN MIND: IF WE CHANGE PARAMETERS, WE NEED TO CHANGE THE NAME OF THE RUN IN WANDB (for the config)
        "project_name": "IDP",
        "predict_mode_stats": args.predict_mode_stats,
        "in_channels": args.in_channels,
        "use_all_features": args.use_all_features,
        "out_channels": args.out_channels,
        "loss_fct": args.loss_fct,
        "use_weighted_loss": args.use_weighted_loss,
        "point_net_conv_layer_structure_local_mlp": [
            int(x) for x in args.point_net_conv_layer_structure_local_mlp.split(",")
        ],
        "point_net_conv_layer_structure_global_mlp": [
            int(x) for x in args.point_net_conv_layer_structure_global_mlp.split(",")
        ],
        "gat_conv_layer_structure": [
            int(x) for x in args.gat_conv_layer_structure.split(",")
        ],
        "use_bootstrapping": args.use_bootstrapping,
        "num_epochs": args.num_epochs,
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "early_stopping_patience": args.early_stopping_patience,
        "use_dropout": args.use_dropout,
        "dropout": args.dropout,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "use_gradient_clipping": args.use_gradient_clipping,
        "device_nr": args.device_nr,
    }

    params["unique_model_description"] = "eign 1.3M params unsigned"
    return params

datalist = []
batch_num = 1
while True:
    print(f"Processing batch number: {batch_num}")
    batch_file = os.path.join(dataset_path, f"datalist_batch_{batch_num}.pt")
    if not os.path.exists(batch_file):
        break
    batch_data = torch.load(batch_file, map_location="cpu")
    if isinstance(batch_data, list):
        datalist.extend(batch_data)
    batch_num += 1
print(f"Loaded {len(datalist)} items into datalist")

# Replace the argparse section with this:
args = {
    "in_channels": 5,
    "use_all_features": False,
    "out_channels": 1,
    "loss_fct": "mse",
    "use_weighted_loss": True,
    "predict_mode_stats": False,
    "point_net_conv_layer_structure_local_mlp": "256",
    "point_net_conv_layer_structure_global_mlp": "512",
    "gat_conv_layer_structure": "128,256,512,256",
    "use_bootstrapping": False,
    "num_epochs": 600,
    "batch_size": 8,
    "lr": 0.001,
    "early_stopping_patience": 40,
    "use_dropout": True,
    "dropout": 0.3,
    "gradient_accumulation_steps": 3,
    "use_gradient_clipping": True,
    "lr_scheduler_warmup_steps": 10000,
    "device_nr": 0,
}


# Convert the dictionary to an object with attributes
class Args:
    def __init__(self, **entries):
        self.__dict__.update(entries)


args = Args(**args)
set_random_seeds()

gpus = get_available_gpus()
best_gpu = select_best_gpu(gpus)
set_cuda_visible_device(best_gpu)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params = get_parameters(args)

# Create directory for the run
unique_run_dir = os.path.join(
    base_dir, params["project_name"], params["unique_model_description"]
)
os.makedirs(unique_run_dir, exist_ok=True)

model_save_path, path_to_save_dataloader = get_paths(
    base_dir=os.path.join(base_dir, params["project_name"]),
    unique_model_description=params["unique_model_description"],
    model_save_path="trained_model/model.pth",
)
train_dl, valid_dl, scalers_train, scalers_validation = (
    prepare_data_with_graph_features(
        datalist=datalist,
        batch_size=params["batch_size"],
        path_to_save_dataloader=path_to_save_dataloader,
        use_all_features=params["use_all_features"],
        use_bootstrapping=params["use_bootstrapping"],
    )
)

config = setup_wandb({param: params[param] for param in PARAMETERS})

def create_model(architecture: str, config: object, device: torch.device):
    return EIGNLaplacianConv(
            in_channels_signed=1,
            out_channels_signed=1,
            in_channels_unsigned=5,
            out_channels_unsigned=1,
            hidden_channels_signed=64,
            hidden_channels_unsigned=64,
            num_blocks=32,
        ).to(device)

gnn_instance = create_model("eign", config, device)

model = gnn_instance.to(device)
loss_fct = EIGN_Loss(
    config.loss_fct, datalist[0].x.shape[0], device, config.use_weighted_loss
)

baseline_loss_mean_target = compute_baseline_of_mean_target(
    dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train
)
baseline_loss = compute_baseline_of_no_policies(
    dataset=train_dl, loss_fct=loss_fct, device=device, scalers=scalers_train
)
print("baseline loss mean " + str(baseline_loss_mean_target))
print("baseline loss no  " + str(baseline_loss))

early_stopping = EarlyStopping(patience=params["early_stopping_patience"], verbose=True)

# Count and print the number of parameters in the model
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Non-trainable parameters: {total_params - trainable_params:,}")

# Print parameter breakdown by layer
print("\nParameter breakdown by layer:")
for name, param in model.named_parameters():
    print(f"{name}: {param.numel():,} parameters, shape: {param.shape}")

best_val_loss, best_epoch = gnn_instance.train_model(
    config=config,
    loss_fct=loss_fct,
    optimizer=torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-4),
    train_dl=train_dl,
    valid_dl=valid_dl,
    device=device,
    early_stopping=early_stopping,
    model_save_path=model_save_path,
    scalers_train=scalers_train,
    scalers_validation=scalers_validation,
    use_signed=False,
)

print(
    f"Best model saved to {model_save_path} with validation loss: {best_val_loss} at epoch {best_epoch}"
)