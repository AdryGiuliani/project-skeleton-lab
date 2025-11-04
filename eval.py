import torch
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from models.model import CustomNet
from train import validate


if __name__ == '__main__':
    #evalute the model
    transform = T.Compose([
        T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = ImageFolder(root='./data/tiny-imagenet-200/val', transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    model = CustomNet()
    model.load_state_dict(torch.load('./model_checkpoint.pth'))
    model = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    val_accuracy, val_loss = validate(model, val_loader, criterion)
    print(f'Final Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')