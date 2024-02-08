# 导入需要的库
import os
from PIL import Image
import random
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# DataSet
class MyDataset(Dataset):
    def __init__(self, images, labels):
        super(MyDataset, self).__init__()
        self.images = images[:]
        self.labels = labels[:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        # print('1:\n',img)
        img = Image.open(img).convert('RGB')
        # print('2:\n',img)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        img = tf(img)
        label = torch.tensor(label)
        return img, label

# 获取数据
def getData(data_dir):
    image_names = []
    print("getData函数接收到的参数：", data_dir)
    for root, sub_folder, file_list in os.walk(data_dir):
        # 每张图片的地址的数组
        image_names += [os.path.join(root, image_path) for image_path in file_list]
    print("所有image的名字：", image_names)
    labels = []
    for file_name in image_names:
        labels.append(1)
    return image_names, labels

# 模型
class OCR_model(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Conv2d(256, 512, 3, 1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, padding=1),
            nn.Flatten(),
            nn.Linear(12800, num_classes, bias=True)
        )

    def forward(self, x):
        x = self.backbone(x)
        return x

# 测试模型的函数
def test_model():
    test_path = 'image/'  # 测试集路径
    batch_size = 1
    lr = 0.01
    try:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    except:
        return -1, -2
    try:
        print("已经进行到了getData")
        img, label = getData(test_path)
    except:
        return -1, -3
    # img = Image.open(img)
    if len(img) == 0:
        return -1, -1
    try:
        dataset = MyDataset(img, label)
        dataloader = DataLoader(dataset, batch_size)
        model = OCR_model(6495).to(device)
        # params = filter(lambda p: p.requires_grad, model.parameters())
        # criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(params, lr, weight_decay=1e-4)
        model.load_state_dict(torch.load(f'./model/model.pt'))
        model.eval()
    except:
        return -1, -4
    try:
        with torch.no_grad():
            for x, y in dataloader:
                # dataset = dataset.to(device)
                x, y = x.to(device), y.to(device)
                pred = model(x)
                char = pred.argmax(1)
                score = int(pred[0][char].item()) + 1
                return char, score
    except:
        return -1, -5

# if __name__ == '__main__':
# epoch_input = int(input("请输入epoch"))
# test_model(epoch_input)
