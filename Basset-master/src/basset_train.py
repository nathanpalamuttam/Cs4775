import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import h5py

class ConvNet(nn.Module):
    def __init__(self, init_depth, seq_len, num_targets, job_params):
        super().__init__()
        self.job_params = job_params
        self.target_type = job_params.get('target_type', 'binary')
        self.build(init_depth, seq_len, num_targets)
        
    def build(self, init_depth, seq_len, num_targets):
        # Construct conv layers
        conv_layers = []
        in_channels = init_depth
        for i, (filters, filter_size, pool_width) in enumerate(zip(
            self.job_params.get('conv_filters', [300, 300, 500]),
            self.job_params.get('conv_filter_sizes', [21, 6, 4]),
            self.job_params.get('pool_width', [4, 4, 4])
        )):
            conv_layer = nn.Sequential(
                nn.Conv2d(in_channels, filters, kernel_size=(1, filter_size)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(1, pool_width))
            )
            conv_layers.append(conv_layer)
            in_channels = filters

        # Calculate flattened size
        test_input = torch.zeros(1, init_depth, 1, seq_len)
        x = test_input
        for conv_layer in conv_layers:
            x = conv_layer(x)
        flattened_size = x.numel() // x.size(0)

        # Construct hidden layers
        hidden_layers = []
        hidden_units = self.job_params.get('hidden_units', [800])
        hidden_dropouts = self.job_params.get('hidden_dropouts', [0.5])
        
        prev_layer_size = flattened_size
        for units, dropout_rate in zip(hidden_units, hidden_dropouts):
            hidden_layer = nn.Sequential(
                nn.Linear(prev_layer_size, units),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            hidden_layers.append(hidden_layer)
            prev_layer_size = units

        # Output layer
        output_layer = nn.Linear(prev_layer_size, num_targets)

        self.features = nn.Sequential(*conv_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            *hidden_layers,
            output_layer
        )

        self.optimizer = optim.Adam(self.parameters())
        
        if self.target_type == 'binary':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            self.criterion = nn.MSELoss()

        return True

    def forward(self, x):
        return self.classifier(self.features(x))

class Batcher:
    def __init__(self, sequences, targets, batch_size, shuffle=True):
        self.sequences = torch.from_numpy(sequences)
        self.targets = torch.from_numpy(targets)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = torch.randperm(len(sequences)) if shuffle else torch.arange(len(sequences))

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            end = min(start + self.batch_size, len(self.indices))
            batch_indices = self.indices[start:end]
            yield (
                self.sequences[batch_indices],
                self.targets[batch_indices]
            )

def main():
    parser = argparse.ArgumentParser(description='DNA ConvNet training')
    parser.add_argument('data_file', help='HDF5 data file')
    parser.add_argument('-cuda', action='store_true', help='Run on CUDA')
    parser.add_argument('-drop_rate', action='store_true', help='Decrease learning rate when training loss stalls')
    parser.add_argument('-job', default='', help='Job parameters file')
    parser.add_argument('-max_epochs', type=int, default=1000, help='Maximum training epochs')
    parser.add_argument('-rc', action='store_true', help='Alternate forward and reverse complement epochs')
    parser.add_argument('-restart', default='', help='Restart an interrupted training run')
    parser.add_argument('-result', default='', help='Write loss value to this file')
    parser.add_argument('-save', default='dnacnn', help='Prefix for saved models')
    parser.add_argument('-seed', default='', help='Seed the model with parameters of another')
    parser.add_argument('-rand', type=int, default=1, help='Random number generator seed')
    parser.add_argument('-stagnant_t', type=int, default=10, help='Allowed epochs with stagnant validation loss')

    args = parser.parse_args()
    torch.manual_seed(args.rand)

    # Load data
    with h5py.File(args.data_file, 'r') as data_file:
        train_targets = data_file['train_out'][:]
        train_seqs = data_file['train_in'][:]
        valid_targets = data_file['valid_out'][:]
        valid_seqs = data_file['valid_in'][:]

    # Parse job parameters
    job_params = {}
    if args.job:
        with open(args.job, 'r') as f:
            for line in f:
                key, value = line.strip().split()
                try:
                    value = float(value)
                except ValueError:
                    pass
                job_params[key] = value

    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Initialize model
    convnet = ConvNet(
        init_depth=train_seqs.shape[1],
        seq_len=train_seqs.shape[3],
        num_targets=train_targets.shape[1],
        job_params=job_params
    ).to(device)

    # Training
    best_acc = 0
    best_epoch = 0

    for epoch in range(1, args.max_epochs + 1):
        print(f"Epoch #{epoch}")

        # Training
        convnet.train()
        train_loss = train_epoch(convnet, train_seqs, train_targets, batch_size=64, device=device)
        print(f"Train loss = {train_loss:.3f}")

        # Validation
        convnet.eval()
        valid_loss, valid_acc = test_epoch(convnet, valid_seqs, valid_targets, device=device)

        # Update best
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_epoch = epoch
            torch.save(convnet.state_dict(), f'{args.save}_best.pth')

        print(f"Validation loss = {valid_loss:.3f}, Validation accuracy = {valid_acc:.4f}")

        # Early stopping
        if epoch - best_epoch > args.stagnant_t:
            break

    # Optional result output
    if args.result:
        with open(args.result, 'w') as f:
            f.write(str(best_acc))

def train_epoch(model, sequences, targets, batch_size=64, device=None):
    if device is None:
        device = torch.device('cpu')
    
    model.to(device)
    sequences = torch.from_numpy(sequences).to(device)
    targets = torch.from_numpy(targets).to(device)
    
    batcher = Batcher(sequences, targets, batch_size)
    total_loss = 0

    for batch_seqs, batch_targets in batcher:
        batch_seqs = batch_seqs.to(device)
        batch_targets = batch_targets.to(device)

        model.optimizer.zero_grad()
        outputs = model(batch_seqs)
        loss = model.criterion(outputs, batch_targets)
        loss.backward()
        model.optimizer.step()

        total_loss += loss.item()

    return total_loss / len(sequences) * batch_size

def test_epoch(model, sequences, targets, device=None):
    if device is None:
        device = torch.device('cpu')
    
    model.to(device)
    sequences = torch.from_numpy(sequences).to(device)
    targets = torch.from_numpy(targets).to(device)

    with torch.no_grad():
        outputs = model(sequences)
        loss = model.criterion(outputs, targets)
        
        if model.target_type == 'binary':
            predictions = torch.sigmoid(outputs)
            accuracy = (predictions.round() == targets).float().mean()
        else:
            accuracy = torch.nn.functional.mse_loss(outputs, targets)

    return loss.item(), accuracy.item()

if __name__ == '__main__':
    main()