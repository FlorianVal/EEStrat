import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.model import BranchyResNet
from src.utils import build_path_count_dict, build_probs_dict
from torchvision import datasets, transforms


def train_loop(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, device
):
    model.to(device)
    best_val_loss = float("inf")

    # Send dummy input to model to get dimensions of outputs (number of heads)
    dummy_input = torch.randn(2, 3, 32, 32).to(device)  # Assuming input shape is (batch_size, channels, height, width)
    with torch.no_grad():
        dummy_outputs = model(dummy_input)
    num_heads = len(dummy_outputs)
    print(f"Number of heads in the model: {num_heads}")

    # Get batch sizes from train and val loaders
    train_batch_size = train_loader.batch_size
    val_batch_size = val_loader.batch_size
    print(f"Train batch size: {train_batch_size}")
    print(f"Validation batch size: {val_batch_size}")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = torch.zeros(num_heads, device=device)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        model.yield_mode = False
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # Compute loss for each head
            losses = [criterion(output, labels) for output in outputs]
            loss = sum(losses)  # Sum up all losses

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            predicted = torch.argmax(outputs, dim=2)
            train_correct += (predicted == labels.unsqueeze(0)).sum(dim=1).float()

            # Update tqdm with current loss and accuracies
            accuracies = [correct / labels.size(0) for correct in (predicted == labels.unsqueeze(0)).sum(dim=1).float().tolist()]
            pbar.set_postfix(
                Losses=f"[{', '.join([f'{loss.item():.2f}' for loss in losses])}]",
                Accuracies=f"[{', '.join([f'{acc:.2f}' for acc in accuracies])}]",
            )

        train_loss /= len(train_loader)
        train_acc = [correct / (len(train_loader) * train_batch_size) for correct in train_correct]

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_correct = torch.zeros(num_heads, device=device)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                # Compute loss for each head
                losses = [criterion(output, labels) for output in outputs]
                loss = sum(losses)  # Sum up all losses

                val_loss += loss.item()

                predicted = torch.argmax(outputs, dim=2)
                val_correct += (predicted == labels.unsqueeze(0)).sum(dim=1).float()

        val_loss /= len(val_loader)
        val_acc = [correct / (len(val_loader) * val_batch_size) for correct in val_correct]

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Acc: {', '.join([f'{acc:.4f}' for acc in train_acc])}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Acc: {', '.join([f'{acc:.4f}' for acc in val_acc])}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # After training, build the path count and probability dictionaries
    model.eval()

    return model, probs_dicts


model = BranchyResNet(num_classes=10, base_model="resnet18")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using device: {device}")
# Load CIFAR-10 dataset
transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
val_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)

trained_model, probs_dicts = train_loop(
    model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=device
)
