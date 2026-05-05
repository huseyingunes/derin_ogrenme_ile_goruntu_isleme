"""
MNIST - PyTorch ile Yapay Sinir Ağı
====================================
Dataset: https://www.kaggle.com/datasets/hojjatk/mnist-dataset

Kullanım:
    python mnist_pytorch.py                  # varsayılan (mevcut klasör)
    python mnist_pytorch.py --data ./data    # farklı klasör
    python mnist_pytorch.py --epochs 20 --lr 0.001
"""

import argparse
import os
import struct
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# ──────────────────────────────────────────────
# 1. IDX formatından veri okuma (bellekten)
# ──────────────────────────────────────────────

def parse_images(b: bytes) -> np.ndarray:
    """IDX3 bytes → (N, H, W) uint8"""
    magic, n, h, w = struct.unpack(">IIII", b[:16])
    assert magic == 2051, f"Beklenen magic=2051, gelen={magic}"
    return np.frombuffer(b[16:], dtype=np.uint8).reshape(n, h, w)


def parse_labels(b: bytes) -> np.ndarray:
    """IDX1 bytes → (N,) uint8"""
    magic, n = struct.unpack(">II", b[:8])
    assert magic == 2049, f"Beklenen magic=2049, gelen={magic}"
    return np.frombuffer(b[8:], dtype=np.uint8)


CANDIDATES = {
    "train_images": ["train-images-idx3-ubyte", "train-images.idx3-ubyte"],
    "train_labels": ["train-labels-idx1-ubyte", "train-labels.idx1-ubyte"],
    "test_images":  ["t10k-images-idx3-ubyte",  "t10k-images.idx3-ubyte"],
    "test_labels":  ["t10k-labels-idx1-ubyte",  "t10k-labels.idx1-ubyte"],
}


def load_mnist(data_dir: str):
    """
    Zip arşivini DISKE AÇMADAN doğrudan bellekten okur.
    Zip yoksa data_dir içindeki IDX dosyalarını kullanır.
    """

    # ── A) ZIP varsa bellekten oku ────────────────────────────────
    zip_files = [f for f in os.listdir(data_dir) if f.endswith(".zip")]
    if zip_files:
        zip_path = os.path.join(data_dir, zip_files[0])
        print(f"Zip bellekten okunuyor: {zip_path}")

        with zipfile.ZipFile(zip_path, "r") as z:
            # basename → zipinfo eşlemesi (klasörleri atla)
            members = {}
            for info in z.infolist():
                if not info.is_dir():
                    members[os.path.basename(info.filename)] = info.filename
            print(f"  Zip içindeki dosyalar: {sorted(members.keys())}")

            def read_zip(key):
                for name in CANDIDATES[key]:
                    if name in members:
                        return z.read(members[name])
                raise FileNotFoundError(
                    f"Zip içinde bulunamadı: {CANDIDATES[key]}\n"
                    f"  Mevcut dosyalar: {sorted(members.keys())}"
                )

            X_train = parse_images(read_zip("train_images"))
            y_train = parse_labels(read_zip("train_labels"))
            X_test  = parse_images(read_zip("test_images"))
            y_test  = parse_labels(read_zip("test_labels"))

    # ── B) Zip yok, klasördeki dosyaları oku ─────────────────────
    else:
        def find_file(key):
            for name in CANDIDATES[key]:
                p = os.path.join(data_dir, name)
                if os.path.isfile(p):
                    return p
            raise FileNotFoundError(
                f"Bulunamadı: {CANDIDATES[key]}\n"
                f"  Klasör: {os.listdir(data_dir)}"
            )

        with open(find_file("train_images"), "rb") as f: X_train = parse_images(f.read())
        with open(find_file("train_labels"), "rb") as f: y_train = parse_labels(f.read())
        with open(find_file("test_images"),  "rb") as f: X_test  = parse_images(f.read())
        with open(find_file("test_labels"),  "rb") as f: y_test  = parse_labels(f.read())

    print(f"  Eğitim seti: {X_train.shape} | Test seti: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ──────────────────────────────────────────────
# 2. DataLoader
# ──────────────────────────────────────────────

def make_loaders(X_train, y_train, X_test, y_test, batch_size=64):
    def prep(X, y):
        X_t = torch.tensor(X, dtype=torch.float32).reshape(-1, 784) / 255.0
        y_t = torch.tensor(y, dtype=torch.long)
        return TensorDataset(X_t, y_t)

    train_loader = DataLoader(prep(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(prep(X_test,  y_test),  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# 3. Model  784 → 512 → 256 → 128 → 10
# ──────────────────────────────────────────────

class MNISTNet(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 4. Eğitim & değerlendirme
# ──────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out  = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        out  = model(X)
        loss = criterion(out, y)
        total_loss += loss.item() * len(y)
        correct    += (out.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return total_loss / n, correct / n


# ──────────────────────────────────────────────
# 5. Ana akış
# ──────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    X_train, y_train, X_test, y_test = load_mnist(args.data)
    train_loader, test_loader = make_loaders(
        X_train, y_train, X_test, y_test, batch_size=args.batch_size
    )

    model     = MNISTNet(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Test Loss':>9} | {'Test Acc':>8}")
    print("-" * 60)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        te_loss, te_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        print(f"{epoch:>6} | {tr_loss:>10.4f} | {tr_acc:>8.2%} | {te_loss:>9.4f} | {te_acc:>7.2%}")
        if te_acc > best_acc:
            best_acc = te_acc
            torch.save(model.state_dict(), args.save)

    print(f"\n✓ En iyi test doğruluğu : {best_acc:.2%}")
    print(f"✓ Model kaydedildi      : {args.save}")

    # Örnek tahminler
    print("\n── Örnek Tahminler (ilk 10 test örneği) ──")
    model.load_state_dict(torch.load(args.save, map_location=device))
    model.eval()
    X_s = torch.tensor(X_test[:10], dtype=torch.float32).reshape(-1, 784) / 255.0
    with torch.no_grad():
        preds = model(X_s.to(device)).argmax(1).cpu().numpy()
    print(f"  Tahmin : {preds.tolist()}")
    print(f"  Gerçek : {y_test[:10].tolist()}")


# ──────────────────────────────────────────────
# 6. CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST PyTorch YSA")
    parser.add_argument("--data",       default=".",              help="Dataset klasörü (zip veya IDX dosyaları)")
    parser.add_argument("--epochs",     type=int,   default=15,   help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int,   default=64,   help="Batch boyutu")
    parser.add_argument("--lr",         type=float, default=1e-3, help="Öğrenme oranı")
    parser.add_argument("--dropout",    type=float, default=0.3,  help="Dropout oranı")
    parser.add_argument("--save",       default="mnist_best.pth", help="Model kayıt yolu")
    args = parser.parse_args()
    main(args)