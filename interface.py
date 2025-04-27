from model import AE as TheModel
from train import train_model as the_trainer
from predict import detect_anomalies as the_predictor
from dataset import MNISTTrainDataset as TheDataset, get_train_dataloader as the_dataloader, get_anomaly_dataloader as the_anomaly_dataloader
from config import batch_size as the_batch_size, epochs as total_epochs, lr, momentum, weight_decay, input_dim, latent_dim, threshold

import torch
import torch.nn as nn
import torch.optim as optim


def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data loaders
    train_loader = the_dataloader(the_batch_size)
    anomaly_loader = the_anomaly_dataloader(the_batch_size)

    # Model, loss, optimizer
    model = TheModel(input_dim=input_dim, latent_dim=latent_dim)
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Train
    print("Starting training...")
    train_metrics = the_trainer(model, total_epochs, train_loader, loss_fn, optimizer, device)

    # Evaluate anomalies
    print("Evaluating anomalies...")
    scores, true_labels, preds = the_predictor(model, anomaly_loader, nn.MSELoss(reduction='none'), device, threshold)

    # Compute metrics
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score
        acc = accuracy_score(true_labels, preds)
        auc = roc_auc_score(true_labels, scores)
        print(f"Anomaly Detection Accuracy: {acc:.4f}")
        print(f"Anomaly Detection AUC: {auc:.4f}")
    except ImportError:
        print("scikit-learn not installed: install it to compute accuracy and AUC metrics.")

if __name__ == "__main__":
    main()
