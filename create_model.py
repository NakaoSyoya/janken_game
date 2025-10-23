import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import os
import torchvision
from torchvision import transforms
import sklearn

print("インポート完了！！\n")

#学習、検証のデータセット分割
def setup_train_val_split(labels, dryrun=False, seed=0):
    x = np.arange(len(labels))
    y = np.array(labels)

    #n_splitsは何回分割するか
    splitter = sklearn.model_selection.StratifiedShuffleSplit(
        n_splits=1, train_size=0.8, random_state=seed
    )
    train_indices, val_indices = next(splitter.split(x, y))

    if dryrun:
        train_indices = np.random.choice(train_indices, 100, replace = False)
        val_indices = np.random.choice(val_indices, 100, replace = False)

    return train_indices, val_indices

#前処理のルール定義
def setup_totensor_normalize():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),#
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )

def get_labels(dataset):
    """ImageFolderデータセットからラベルリストを抽出する"""
    return dataset.targets

#学習データと検証データを返す関数
def setup_train_val_datasets(data_dir, dryrun=False):
    dataset = torchvision.datasets.ImageFolder(
        "archive/Rock-Paper-Scissors/train",#ファイルパスゲット
        transform=setup_totensor_normalize(), #前処理のルール取得
    )
    labels = get_labels(dataset)
    train_indices, val_indices = setup_train_val_split(labels, dryrun) #学習、検証のインデックス

    train_dataset = torch.utils.data.Subset(dataset, train_indices) #学習データ
    val_dataset = torch.utils.data.Subset(dataset, train_indices) #検証データ

    return train_dataset, val_dataset

#一つずつバッチを取り出す仕組みを返す
def setup_train_val_loaders(data_dir, batch_size, dryrun):
    train_dataset, val_dataset = setup_train_val_datasets(
        data_dir, dryrun = dryrun
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = 0
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, num_workers = 0
    )

    return train_loader, val_loader

#学習の1エポック
def train_1epoch(model, train_loader, lossfun, optimizer, device):
    model.train() #訓練モードに
    total_loss, total_acc = 0.0, 0.0
    for x, y in tqdm(train_loader):
        x = x.to(device) #GPUに送る
        y = y.to(device)
        optimizer.zero_grad() #勾配リセット
        out = model(x) #順伝播
        loss = lossfun(out, y) #誤差を計算
        _, pred = torch.max(out.detach(), 1) #最大スコアの予測クラスを取得
        loss.backward() #逆伝播
        optimizer.step() #パラメータ更新

        total_loss += loss.item() * x.size(0)
        total_acc += torch.sum(pred == y)
    
    avg_loss = total_loss / len(train_loader.dataset)
    avg_acc = total_acc / len(train_loader.dataset)
    return avg_acc, avg_loss

#検証の1エポック
def validate_1epoch(model, val_loader, lossfun, device):
    model.eval()
    total_loss, total_acc = 0.0, 0.0

    with torch.no_grad():
        for x,y in tqdm(val_loader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = lossfun(out.detach(), y)
            _, pred = torch.max(out, 1)

            total_loss += loss.item() * x.size(0)
            total_acc += torch.sum(pred == y)

    avg_loss = total_loss / len(val_loader.dataset)
    avg_acc = total_acc / len(val_loader.dataset)
    return avg_acc, avg_loss

#学習
def train(model, optimizer, train_loader, val_loader, n_epochs, device):
    lossfun = torch.nn.CrossEntropyLoss()

    for epoch in tqdm(range(n_epochs)):
        train_acc, train_loss = train_1epoch(
            model, train_loader, lossfun, optimizer, device
        )
        val_acc, val_loss = validate_1epoch(model, val_loader, lossfun, device)
        print(
            f"epoch={epoch}, train loss={train_loss}, train accuracy={train_acc}\n"
            f"val loss={val_loss}, val accuracy={val_acc}"
        )

def train_subsec5(data_dir, batch_size, dryrun=False, device='cpu'):
    model = torchvision.models.resnet50(pretrained = True)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    train_loader, val_loader = setup_train_val_loaders(
        data_dir, batch_size, dryrun
    )
    train(
        model, optimizer, train_loader, val_loader, n_epochs=1, device=device
    )

    return model

#実際の実行
#dryrunをTrueにするとお試し実行
model = train_subsec5(
    data_dir="archive/Rock-Paper-Scissors",
    batch_size=32,
    dryrun=False,
    device="cpu"
)

torch.save(model.state_dict(), "model/best_model.pth")
print(f"✅ モデルを保存しました")

#ここからはテストデータでの精度
def evaluate(model, data_loader, device="cpu"):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for x, y in tqdm(data_loader):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == y).sum().item()
            total_samples += y.size(0)

    accuracy = total_correct / total_samples
    print(f"✅ Test Accuracy: {accuracy:.4f}")
    return accuracy
test_dir = "archive/Rock-Paper-Scissors/test"
test_dataset = torchvision.datasets.ImageFolder(test_dir, transform=setup_totensor_normalize())
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_acc = evaluate(model, test_loader, "cpu")
print("コード実行完了！！")