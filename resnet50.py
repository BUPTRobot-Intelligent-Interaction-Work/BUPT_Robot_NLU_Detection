# Load model directly
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import AutoImageProcessor, AutoModelForImageClassification, AdamW, TrainingArguments, Trainer
from torch.optim.lr_scheduler import StepLR


# 数据预处理和加载
transform = Compose([
    Resize((224, 224)),  # 调整图像大小
    ToTensor(),  # 将图像转换为tensor
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 假设你的数据目录结构为：
# /path/to/images/train
# /path/to/images/val
train_dataset = ImageFolder('emotions/images/train', transform=transform)
val_dataset = ImageFolder('emotions/images/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-18", num_labels=3,
                                                        ignore_mismatched_sizes=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3)
# scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

# 训练循环
for epoch in range(300):  # 训练3轮
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        # print(outputs.logits.shape)
        loss = criterion(outputs.logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # scheduler.step()

    # 简单的验证循环
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Accuracy after epoch {epoch + 1}: {100 * correct / total:.2f}%')
    torch.save(model.state_dict(), f"./resnet_model/resnet18_epoch{epoch + 1}.pth")

print("Training complete!")
