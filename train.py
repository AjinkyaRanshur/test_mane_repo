import torch
import time
from datetime import timedelta


def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device):
    """Runs training loop and saves final_weights.pth in checkpoints/."""
    model.to(device)
    metrics = {'train_loss': []}
    start = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            recon = model(batch)
            loss = loss_fn(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader.dataset)
        metrics['train_loss'].append(epoch_loss)
        print(f"[EPOCH {epoch+1}/{num_epochs}] LOSS: {epoch_loss:.4f} TIME: {timedelta(seconds=time.time()-epoch_start)}")

    print(f"Training complete in {timedelta(seconds=time.time()-start)}")
    torch.save(model.state_dict(), 'checkpoints/final_weights.pth')
    return metrics
