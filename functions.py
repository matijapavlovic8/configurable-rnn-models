from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.nn.utils import clip_grad_norm_
import torch


def train(model, data_loader, optimizer, criterion, device, grad_clip=0.25):
    model.train()
    total_loss = 0

    for batch in data_loader:
        x, y = batch
        x, y = x.to(device), y[0].to(device)
        optimizer.zero_grad()
        logits = model(x).squeeze()
        loss = criterion(logits, y.float())
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(data_loader)


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x, y = x.to(device), y[0].to(device)
            logits = model(x).squeeze()
            loss = criterion(logits, y.float())
            total_loss += loss.item()
            all_logits.append(logits)
            all_labels.append(y)

    avg_loss = total_loss / len(data_loader)
    all_logits = torch.cat(all_logits).cpu()
    all_labels = torch.cat(all_labels).cpu()

    preds = torch.round(torch.sigmoid(all_logits))
    accuracy = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds)
    conf_matrix = confusion_matrix(all_labels, preds)

    return avg_loss, accuracy, f1, conf_matrix
