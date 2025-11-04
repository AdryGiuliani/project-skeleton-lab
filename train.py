import torchvision.transforms as transforms
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import torch
import wandb

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

# Validation loop
def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0

    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    return val_accuracy, val_loss

if __name__ == '__main__':
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    wandb.init(project="tiny-imagenet-training")
    config = wandb.config
    config.lr = 0.001
    config.batch_size = 32

    # root/{classX}/x001.jpeg
    tiny_imagenet_dataset_train = ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=config.batch_size, shuffle=False, num_workers=8)

    best_acc = 0
    from models.model import CustomNet
    model = CustomNet().cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    for epoch in range(1, 11):
        train(epoch, model, train_loader, criterion, optimizer)
        val_acc, val_loss = validate(model, val_loader, criterion)
        wandb.log({'epoch': epoch, 'val_accuracy': val_acc, 'val_loss': val_loss})
        best_acc = max(val_acc, best_acc)

    print(f'Best Validation Accuracy: {best_acc:.2f}%')
    torch.save(model.state_dict(), './model_checkpoint.pth')
    wandb.finish()
