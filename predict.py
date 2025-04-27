import torch


def detect_anomalies(model, anomaly_loader, loss_fn, device, threshold):
    """Generates reconstruction loss scores and binary predictions for anomalies."""
    model.to(device)
    model.eval()
    scores, true_labels, preds = [], [], []

    with torch.no_grad():
        for features, label in anomaly_loader:
            features = features.to(device)
            recon = model(features)
            # MSE over features per sample
            loss = loss_fn(recon, features)
            per_sample = loss.mean(dim=1).cpu().numpy()
            scores.extend(per_sample.tolist())
            true_labels.extend(label.numpy().tolist())

    preds = [1 if s >= threshold else 0 for s in scores]
    return scores, true_labels, preds
