import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1. XOR veri seti
# -----------------------------
X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0]
])

y = torch.tensor([
    [0.0],
    [1.0],
    [1.0],
    [0.0]
])

# -----------------------------
# 2. Basit sinir ağı modeli
# -----------------------------
class SimpleXORNet(nn.Module):
    def __init__(self):
        super(SimpleXORNet, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 4),   # 2 giriş -> 4 gizli nöron
            nn.Sigmoid(),      # aktivasyon fonksiyonu
            nn.Linear(4, 1),   # 4 gizli nöron -> 1 çıkış
            nn.Sigmoid()       # çıktı 0 ile 1 arasında olsun
        )

    def forward(self, x):
        return self.model(x)

# Modeli oluştur
model = SimpleXORNet()

# -----------------------------
# 3. Loss ve optimizer
# -----------------------------
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.05)

# -----------------------------
# 4. Eğitim döngüsü
# -----------------------------
epochs = 5000

for epoch in range(epochs):
    # Tahmin
    outputs = model(X)

    # Hata hesaplama
    loss = criterion(outputs, y)

    # Eski gradyanları sıfırla
    optimizer.zero_grad()

    # Geri yayılım
    loss.backward()

    # Ağırlıkları güncelle
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.6f}")

# -----------------------------
# 5. Modeli test et
# -----------------------------
print("\nModel tahminleri:")

with torch.no_grad():
    predictions = model(X)

    for i in range(len(X)):
        print(f"Girdi: {X[i].tolist()} -> Tahmin: {predictions[i].item():.4f}")