import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.nn.utils import clip_grad_norm_


def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs=5, gradient_clip=0.25):
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels[0].float())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
            optimizer.step()
            train_running_loss += loss.item()
        train_epoch_loss = train_running_loss / len(train_loader)

        model.eval()
        valid_running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels[0].float())
                valid_running_loss += loss.item()
        valid_epoch_loss = valid_running_loss / len(valid_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_epoch_loss:.4f}, Valid Loss: {valid_epoch_loss:.4f}")


def evaluate(model, dataloader, criterion, output_file):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels[0].float())
            running_loss += loss.item()
            predictions.extend(torch.round(torch.sigmoid(outputs)).cpu().numpy())
            true_labels.extend(labels[0].cpu().numpy())
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    cm = confusion_matrix(true_labels, predictions)
    with open(output_file, 'w') as f:
        f.write(f'Average Loss: {running_loss / len(dataloader)}\n')
        f.write(f'Accuracy: {accuracy}\n')
        f.write(f'F1 Score: {f1}\n')
        f.write('Confusion Matrix:\n')
        for row in cm:
            f.write(' '.join(map(str, row)) + '\n')
    return running_loss / len(dataloader), accuracy, f1, cm
