import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import json
from PIL import Image

# 数据目录与模型保存路径（根据实际项目修改）
DATA_DIR = "./data/images"          # 建议结构：data/images/library/xxx.jpg
NUM_CLASSES = 13                      # 你给定的地点数量
SAVE_PATH = "./models/best_model.pth"
CLASS_MAP_PATH = "./data/class_map.json"  # 保存类别映射

# 支持的图片格式
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


def is_valid_image_file(file_path):
    """检查文件是否为有效的图片文件"""
    # 检查扩展名
    _, ext = os.path.splitext(file_path.lower())
    if ext not in SUPPORTED_FORMATS:
        return False
    
    # 尝试打开文件以验证它确实是图片
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证图片文件的完整性
        return True
    except:
        return False


def build_dataloaders(batch_size: int = 32):
    """构建训练用 DataLoader，过滤掉空的类别文件夹。"""
    # 获取所有子文件夹（候选类别）
    candidate_classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    # 过滤掉空的类别文件夹
    valid_classes = []
    for cls in candidate_classes:
        cls_path = os.path.join(DATA_DIR, cls)
        # 检查是否包含有效图片文件
        has_valid_images = any(is_valid_image_file(os.path.join(cls_path, f)) 
                              for f in os.listdir(cls_path))
        if has_valid_images:
            valid_classes.append(cls)
    
    print(f"找到 {len(valid_classes)} 个包含有效图片的类别: {valid_classes}")
    
    # 更新全局类别数量
    global NUM_CLASSES
    NUM_CLASSES = len(valid_classes)
    print(f"更新 NUM_CLASSES = {NUM_CLASSES}")
    
    # 创建自定义数据集，只包含有效类别
    class CustomImageFolder(Dataset):
        def __init__(self, root, transform=None):
            self.root = root
            self.transform = transform
            self.samples = []
            self.class_to_idx = {cls: i for i, cls in enumerate(valid_classes)}
            self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
            
            # 收集所有有效样本
            for cls in valid_classes:
                cls_path = os.path.join(root, cls)
                for f in os.listdir(cls_path):
                    file_path = os.path.join(cls_path, f)
                    if is_valid_image_file(file_path):
                        self.samples.append((file_path, self.class_to_idx[cls]))
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path, label = self.samples[idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])

    train_set = CustomImageFolder(DATA_DIR, transform=transform_train)
    loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 把类别映射保存成 json，推理时会用到
    os.makedirs(os.path.dirname(CLASS_MAP_PATH), exist_ok=True)
    with open(CLASS_MAP_PATH, "w", encoding="utf8") as f:
        json.dump(train_set.class_to_idx, f, ensure_ascii=False, indent=2)

    return loader, train_set.class_to_idx


def build_model(num_classes: int) -> nn.Module:
    """构建 ResNet18 基础模型，只训练最后一层全连接。"""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 冻结 backbone，只微调分类头（基础版本够用了）
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(512, num_classes)
    return model


def train(num_epochs: int = 50, lr: float = 1e-3, batch_size: int = 32):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    loader, class_map = build_dataloaders(batch_size=batch_size)
    model = build_model(NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)

    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0

        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch + 1}/{num_epochs} - loss: {loss.item():.4f}, acc: {acc:.4f}")

        # 保存最优模型
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print("✓ Saved best model.")

    print("Training finished. Best acc =", best_acc)


if __name__ == "__main__":
    train()
