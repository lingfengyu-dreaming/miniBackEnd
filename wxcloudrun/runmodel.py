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
    pick = {0: '一', 1: '乙', 2: '二', 3: '十', 4: '丁', 5: '厂', 6: '七', 7: '卜', 8: '八', 9: '人', 10: '入',
            11: '儿', 12: '匕', 13: '几', 14: '九', 15: '刁', 16: '了', 17: '刀', 18: '力', 19: '乃', 20: '又',
            21: '三', 22: '干', 23: '于', 24: '亏', 25: '工', 26: '土', 27: '士', 28: '才', 29: '下', 30: '寸',
            31: '大', 32: '丈', 33: '与', 34: '万', 35: '上', 36: '小', 37: '口', 38: '山', 39: '巾', 40: '千',
            41: '乞', 42: '川', 43: '亿', 44: '个', 45: '夕', 46: '久', 47: '么', 48: '勺', 49: '凡', 50: '丸',
            51: '及', 52: '广', 53: '亡', 54: '门', 55: '丫', 56: '义', 57: '之', 58: '尸', 59: '己', 60: '已',
            61: '巳', 62: '弓', 63: '子', 64: '卫', 65: '也', 66: '女', 67: '刃', 68: '飞', 69: '习', 70: '叉',
            71: '马', 72: '乡', 73: '丰', 74: '王', 75: '开', 76: '井', 77: '天', 78: '夫', 79: '元', 80: '无',
            81: '云', 82: '专', 83: '丐', 84: '扎', 85: '艺', 86: '木', 87: '五', 88: '支', 89: '厅', 90: '不',
            91: '犬', 92: '太', 93: '区', 94: '历', 95: '歹', 96: '友', 97: '尤', 98: '匹', 99: '车', 100: '巨',
            101: '牙', 102: '屯', 103: '戈', 104: '比', 105: '互', 106: '切', 107: '瓦', 108: '止', 109: '少',
            110: '曰', 111: '日', 112: '中', 113: '贝', 114: '冈', 115: '内', 116: '水', 117: '见', 118: '午',
            119: '牛', 120: '手', 121: '气', 122: '毛', 123: '壬', 124: '升', 125: '夭', 126: '长', 127: '仁',
            128: '什', 129: '片', 130: '仆', 131: '化', 132: '仇', 133: '币', 134: '仍', 135: '仅', 136: '斤',
            137: '爪', 138: '反', 139: '介', 140: '父', 141: '从', 142: '仑', 143: '今', 144: '凶', 145: '分',
            146: '乏', 147: '公', 148: '仓', 149: '月', 150: '氏', 151: '勿', 152: '欠', 153: '风', 154: '丹',
            155: '匀', 156: '乌', 157: '勾', 158: '凤', 159: '六', 160: '文', 161: '亢', 162: '方', 163: '火',
            164: '为', 165: '斗', 166: '忆', 167: '计', 168: '订', 169: '户', 170: '认', 171: '冗', 172: '讥',
            173: '心', 174: '尺', 175: '引', 176: '丑', 177: '巴', 178: '孔', 179: '队', 180: '办', 181: '以',
            182: '允', 183: '予', 184: '邓', 185: '劝', 186: '双', 187: '书', 188: '幻', 189: '玉', 190: '刊',
            191: '未', 192: '末', 193: '示', 194: '击', 195: '打', 196: '巧', 197: '正', 198: '扑', 199: '卉',
            200: '扒', 201: '功', 202: '扔', 203: '去', 204: '甘', 205: '世', 206: '艾', 207: '古', 208: '节',
            209: '本', 210: '术', 211: '可', 212: '丙', 213: '左', 214: '厉', 215: '石', 216: '右', 217: '布',
            218: '夯', 219: '戊', 220: '龙', 221: '平', 222: '灭', 223: '轧', 224: '东', 225: '卡', 226: '北',
            227: '占', 228: '凸', 229: '卢', 230: '业', 231: '旧', 232: '帅', 233: '归', 234: '旦', 235: '目',
            236: '且', 237: '叶', 238: '甲', 239: '申', 240: '叮', 241: '电', 242: '号', 243: '田', 244: '由',
            245: '只', 246: '叭', 247: '史', 248: '央', 249: '兄', 250: '叽', 251: '叼', 252: '叫', 253: '叩',
            254: '叨', 255: '另', 256: '叹', 257: '冉', 258: '皿', 259: '凹', 260: '囚', 261: '四', 262: '生',
            263: '矢', 264: '失', 265: '乍', 266: '禾', 267: '丘', 268: '付', 269: '仗', 270: '代', 271: '仙',
            272: '们', 273: '仪', 274: '白', 275: '仔', 276: '他', 277: '斥', 278: '瓜', 279: '乎', 280: '丛',
            281: '令', 282: '用', 283: '甩', 284: '印', 285: '尔', 286: '乐', 287: '句', 288: '匆', 289: '册',
            290: '卯', 291: '犯', 292: '外', 293: '处', 294: '冬', 295: '鸟', 296: '务', 297: '包', 298: '饥',
            299: '主', 300: '市', 301: '立', 302: '冯', 303: '玄', 304: '闪', 305: '兰', 306: '半', 307: '汁',
            308: '汇', 309: '头', 310: '汉', 311: '宁', 312: '穴', 313: '它', 314: '讨', 315: '写', 316: '让',
            317: '礼', 318: '训', 319: '议', 320: '必', 321: '讯', 322: '记', 323: '永', 324: '司', 325: '尼',
            326: '民', 327: '弗', 328: '弘', 329: '出', 330: '辽', 331: '奶', 332: '奴', 333: '召', 334: '加',
            335: '皮', 336: '边', 337: '孕', 338: '发', 339: '圣', 340: '对', 341: '台', 342: '矛', 343: '纠',
            344: '母', 345: '幼', 346: '丝', 347: '邦', 348: '式', 349: '迂', 350: '刑', 351: '戎', 352: '动',
            353: '扛', 354: '寺', 355: '吉', 356: '扣', 357: '考', 358: '托', 359: '老', 360: '巩', 361: '圾',
            362: '执', 363: '扩', 364: '扫', 365: '地', 366: '场', 367: '扬', 368: '耳', 369: '芋', 370: '共',
            371: '芒', 372: '亚', 373: '芝', 374: '朽', 375: '朴', 376: '机', 377: '权', 378: '过', 379: '臣',
            380: '吏', 381: '再', 382: '协', 383: '西', 384: '压', 385: '厌', 386: '戌', 387: '在', 388: '百',
            389: '有', 390: '存', 391: '而', 392: '页', 393: '匠', 394: '夸', 395: '夺', 396: '灰', 397: '达',
            398: '列'}
    try:
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    except:
        return -1, -2
    try:
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
        model.load_state_dict(torch.load(f'model/model.pt', map_location=torch.device('cpu')))
        model.eval()
    except:
        return -1, -4
    try:
        with torch.no_grad():
            for x, y in dataloader:
                # dataset = dataset.to(device)
                x, y = x.to(device), y.to(device)
                try:
                    pred = model(x)
                except:
                    return -1, -6
                try:
                    char = pred.argmax(1)
                    score = int(pred[0][char].item()) + 1
                    char = pick[char[0].item()]
                except:
                    return -1, -7
                return char, score
    except:
        return -1, -5

# if __name__ == '__main__':
# epoch_input = int(input("请输入epoch"))
# test_model(epoch_input)
