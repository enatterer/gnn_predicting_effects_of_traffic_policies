#This is a script to generate the command line command for scripts/training/run_models.py

#arguments:
# --gnn_arch: the GNN architecture to use
# --project_name: the name of the project
# --unique_model_description: a unique description of the model
# --in_channels: the number of input channels
# --use_all_features: whether to use all features
# --out_channels: the number of output channels
# --model_kwargs: additional model parameters (as defined in the class) in JSON format (path to the file)
# --loss_fct: the loss function to use
# --use_weighted_loss: whether to use weighted loss (based on vol_base_case) or not
# --predict_mode_stats: whether to predict mode stats or not
# --use_bootstrapping: whether to use bootstrapping for train-validation split
# --num_epochs: the number of epochs to train for
# --batch_size: the batch size for training
# --lr: the learning rate for the model
# --early_stopping_patience: the patience for early stopping
# --use_dropout: whether to use dropout
# --dropout: the dropout rate
# --gradient_accumulation_steps: the number of steps to accumulate gradients
# --use_gradient_clipping: whether to use gradient clipping
# --device_nr: the device number (0 or 1 for Retina Roaster's two GPUs)
# --continue_training: whether to continue training from a checkpoint
# --base_checkpoint_path: the path to the checkpoint to continue training from
# --learning_variant: the learning variant to use
# --split_variant: the variant of the train-validation-test data split
# --sampler_variant: the variant of the sampler
# --run_name: the name of the run from data preprocessing

gnn_arch = "gatv2"
#project_name = None
unique_model_description = "gatv2_transductive_only_rosenheim"
in_channels = 5
use_all_features = False
out_channels = 1
#model_kwargs = None
loss_fct = "mse"
use_weighted_loss = True
predict_mode_stats = False
use_bootstrapping = False
batch_size = 4
num_epochs = 1
lr = 0.003
early_stopping_patience = 25
use_dropout = True
dropout = 0.3
gradient_accumulation_steps = 3
use_gradient_clipping = True
device_nr = 0
continue_training = False
#base_checkpoint_path = None
learning_variant = "transductive"
split_variant = "non_uniform"
sampler_variant = "nested_fixed_proportion"
run_name = "1"

print(f"python ../../scripts/training/run_models.py --gnn_arch {gnn_arch} --unique_model_description {unique_model_description} --in_channels {in_channels} --use_all_features {use_all_features} --out_channels {out_channels} --loss_fct {loss_fct} --use_weighted_loss {use_weighted_loss} --predict_mode_stats {predict_mode_stats} --use_bootstrapping {use_bootstrapping} --num_epochs {num_epochs} --batch_size {batch_size} --lr {lr} --early_stopping_patience {early_stopping_patience} --use_dropout {use_dropout} --dropout {dropout} --gradient_accumulation_steps {gradient_accumulation_steps} --use_gradient_clipping {use_gradient_clipping} --device_nr {device_nr} --continue_training {continue_training} --learning_variant {learning_variant} --split_variant {split_variant} --sampler_variant {sampler_variant} --run_name {run_name}")