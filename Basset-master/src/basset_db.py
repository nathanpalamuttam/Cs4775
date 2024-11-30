#!/usr/bin/env python3

import argparse
import h5py
import torch
import torch.nn as nn
import numpy as np
import time
from collections import defaultdict


class ConvNet(nn.Module):
    def __init__(self, job, input_depth, seq_len, num_targets):
        super(ConvNet, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_depth, job['conv_filters'][0], job['conv_filter_sizes'][0]),
            nn.ReLU(),
            nn.MaxPool1d(job['pool_width'][0]),
            nn.Conv1d(job['conv_filters'][0], job['conv_filters'][1], job['conv_filter_sizes'][1]),
            nn.ReLU(),
            nn.MaxPool1d(job['pool_width'][1]),
            nn.Conv1d(job['conv_filters'][1], job['conv_filters'][2], job['conv_filter_sizes'][2]),
            nn.ReLU(),
            nn.MaxPool1d(job['pool_width'][2])
        )

        # Dynamically calculate output size after convolutional layers
        conv_out_size = self._get_conv_output(seq_len, input_depth, job)
        self.fc_layers = nn.Sequential(
            nn.Linear(conv_out_size, job['hidden_units'][0]),
            nn.ReLU(),
            nn.Dropout(job['hidden_dropouts'][0]),
            nn.Linear(job['hidden_units'][0], num_targets)
        )

    def _get_conv_output(self, seq_len, input_depth, job):
        x = torch.zeros(1, input_depth, seq_len)  # Dummy input
        x = self.conv_layers(x)
        return x.numel()  # Total number of features after convolutions

    def forward(self, x):
        x = x.squeeze(2)  # Ensure input shape is [Batch, Channels, Sequence Length]
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x


def load_hdf5_data(data_file):
    """Load training and validation data from HDF5 file."""
    with h5py.File(data_file, 'r') as f:
        train_seqs = torch.tensor(f['train_in'][:], dtype=torch.float32)
        train_targets = torch.tensor(f['train_out'][:], dtype=torch.float32)
        valid_seqs = torch.tensor(f['valid_in'][:], dtype=torch.float32)
        valid_targets = torch.tensor(f['valid_out'][:], dtype=torch.float32)

    # Debug: Check dataset statistics
    print(f"Training data shape: {train_seqs.shape}, Targets shape: {train_targets.shape}")
    print(f"Validation data shape: {valid_seqs.shape}, Targets shape: {valid_targets.shape}")
    print(f"Training targets (first 10): {train_targets[:10]}")

    return train_seqs, train_targets, valid_seqs, valid_targets


def check_data_leakage(train_seqs, valid_seqs):
    """Check for data leakage between training and validation datasets."""
    # Convert each sequence to a tuple of tuples (to make it hashable)
    train_set = set(tuple(map(tuple, seq.tolist())) for seq in train_seqs)
    valid_set = set(tuple(map(tuple, seq.tolist())) for seq in valid_seqs)

    overlap = len(train_set.intersection(valid_set))
    print(f"Data overlap between train and validation sets: {overlap}")
    if overlap > 0:
        print("Warning: Data leakage detected!")



def validate_data_variance(data, name):
    """Validate that input sequences have meaningful variance."""
    mean = data.mean().item()
    std = data.std().item()
    print(f"{name} - Mean: {mean}, Std Dev: {std}")


def train_epoch(model, criterion, optimizer, train_loader, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate(model, criterion, valid_loader, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            correct += (outputs.argmax(1) == targets.argmax(1)).sum().item()
    return total_loss / len(valid_loader), correct / len(valid_loader.dataset)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='DNA ConvNet Training')
    parser.add_argument('data_file', help='Path to the HDF5 data file')
    parser.add_argument('--cuda', action='store_true', help='Use GPU for training')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--save', default='dnacnn', help='Prefix for saved models')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Load data
    train_seqs, train_targets, valid_seqs, valid_targets = load_hdf5_data(args.data_file)

    print(f"Unique training targets: {torch.unique(train_targets)}")
    print(f"Unique validation targets: {torch.unique(valid_targets)}")

    # Ensure correct loss function
    if train_targets.shape[1] == 1:  # Binary classification
        criterion = nn.BCEWithLogitsLoss()
        train_targets = train_targets.squeeze(1)  # Remove extra dimension if needed
        valid_targets = valid_targets.squeeze(1)
    else:  # Multi-class classification
        criterion = nn.CrossEntropyLoss()
    # Debug: Check for data leakage and input variance
    check_data_leakage(train_seqs, valid_seqs)
    validate_data_variance(train_seqs, "Train Sequences")
    validate_data_variance(valid_seqs, "Validation Sequences")

    # Hyperparameters
    job = {
        'conv_filters': [64, 128, 256],  # Reduced model complexity
        'conv_filter_sizes': [15, 5, 3],
        'pool_width': [4, 4, 2],
        'hidden_units': [128],
        'hidden_dropouts': [0.5]
    }

    # Model, criterion, optimizer
    model = ConvNet(job, train_seqs.shape[1], train_seqs.shape[2], train_targets.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_seqs, train_targets), batch_size=args.batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(valid_seqs, valid_targets), batch_size=args.batch_size
    )

    # Training loop
    best_valid_acc = 0
    for epoch in range(1, args.max_epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, criterion, optimizer, train_loader, device)
        valid_loss, valid_acc = validate(model, criterion, valid_loader, device)

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Valid Loss={valid_loss:.4f}, Valid Acc={valid_acc:.4f}, Time={epoch_time:.2f}s")

        # Save best model
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(model.state_dict(), f"{args.save}_best.pth")
            print("Saved best model!")

    print(f"Best Validation Accuracy: {best_valid_acc:.4f}")


if __name__ == '__main__':
    main()
