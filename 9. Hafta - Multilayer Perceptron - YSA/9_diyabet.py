import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════
#  BÖLÜM 1: VERİ YÜKLEME VE KEŞİF
# ══════════════════════════════════════════════
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
sutunlar = ['gebelik', 'glikoz', 'kan_basinci', 'deri_kalinligi',
            'insulin', 'bmi', 'soy_gecmis', 'yas', 'diyabet']

df = pd.read_csv(url, header=None, names=sutunlar)

print("Veri boyutu:", df.shape)
print("\nİlk 5 satır:")
print(df.head())
print("\nEksik değer kontrolü (0 gerçekte eksik olabilir):")
sifir_sutunlar = ['glikoz', 'kan_basinci', 'deri_kalinligi', 'insulin', 'bmi']
for s in sifir_sutunlar:
    sifir_sayisi = (df[s] == 0).sum()
    print(f"  {s}: {sifir_sayisi} sıfır değer ({sifir_sayisi / len(df) * 100:.1f}%)")

# ══════════════════════════════════════════════
#  BÖLÜM 2: ÖN İŞLEME
# ══════════════════════════════════════════════
# Fizyolojik açıdan sıfır olamayacak değerleri medyan ile doldur
for s in sifir_sutunlar:
    df[s] = df[s].replace(0, np.nan)
    df[s] = df[s].fillna(df[s].median())

X = df.drop('diyabet', axis=1).values
y = df['diyabet'].values

# Train / Validation / Test bölünmesi (%70 / %15 / %15)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# Standartlaştırma — SADECE train üzerinde fit et!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


# ══════════════════════════════════════════════
#  BÖLÜM 3: MODEL
# ══════════════════════════════════════════════
class DiyabetModeli(nn.Module):
    def __init__(self):
        super().__init__()
        self.ag = nn.Sequential(
            # Gizli katman 1: 8 → 32
            nn.Linear(8, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Gizli katman 2: 32 → 16
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Çıktı: 16 → 1 (ikili sınıflandırma)
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.ag(x)


model = DiyabetModeli()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
kayip_fn = nn.BCEWithLogitsLoss()

print(f"\nModel parametreleri: {sum(p.numel() for p in model.parameters()):,}")


# ══════════════════════════════════════════════
#  BÖLÜM 4: EĞİTİM
# ══════════════════════════════════════════════
# Veri yükleyiciler
def tensor_yukleyici(X, y, batch_size, karistir=False):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=karistir)


train_loader = tensor_yukleyici(X_train, y_train, batch_size=32, karistir=True)
val_loader = tensor_yukleyici(X_val, y_val, batch_size=32)

tarih = {'train': [], 'val': []}
EN_IYI_VAL = float('inf')
PATIENCE = 15
sabir = 0

for epoch in range(1, 201):
    # Eğitim
    model.train()
    train_kayip = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        kayip = kayip_fn(model(Xb).squeeze(), yb)
        kayip.backward()
        optimizer.step()
        train_kayip += kayip.item()

    # Validasyon
    model.eval()
    val_kayip = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            val_kayip += kayip_fn(model(Xb).squeeze(), yb).item()

    tarih['train'].append(train_kayip / len(train_loader))
    tarih['val'].append(val_kayip / len(val_loader))

    if tarih['val'][-1] < EN_IYI_VAL:
        EN_IYI_VAL = tarih['val'][-1]
        sabir = 0
        torch.save(model.state_dict(), 'diyabet_en_iyi.pth')
    else:
        sabir += 1
        if sabir >= PATIENCE:
            print(f"Erken durdurma — epoch {epoch}")
            break

    if epoch % 20 == 0:
        print(f"Epoch {epoch}: Train={tarih['train'][-1]:.4f}, Val={tarih['val'][-1]:.4f}")

# ══════════════════════════════════════════════
#  BÖLÜM 5: TEST DEĞERLENDİRME
# ══════════════════════════════════════════════
model.load_state_dict(torch.load('diyabet_en_iyi.pth'))
model.eval()

with torch.no_grad():
    test_logits = model(torch.FloatTensor(X_test)).squeeze()
    test_probs = torch.sigmoid(test_logits).numpy()
    test_preds = (test_probs >= 0.5).astype(int)

print("\n" + "=" * 50)
print("TEST SETİ SONUÇLARI")
print("=" * 50)
print(classification_report(y_test, test_preds,
                            target_names=["Diyabetik Değil", "Diyabetik"]))
print(f"ROC-AUC: {roc_auc_score(y_test, test_probs):.4f}")