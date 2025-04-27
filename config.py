# File paths
train_csv     = 'data/mnist_train.csv'
test_csv      = 'data/anom.csv'

# Model parameters
input_dim     = 784
latent_dim    = 16

# Training hyperparameters
batch_size    = 32
lr            = 1e-2
weight_decay  = 1e-5
momentum      = 0.9
epochs        = 15

# Threshold for anomaly decision
threshold     = 0.3

# DataLoader settings
torch_num_workers = 4
pin_memory         = True
