"""
Kahve Dükkanı Müşteri Memnuniyeti - Yapay Sinir Ağı Örneği
-----------------------------------------------------------
Tek perceptron ile çözülemeyen, MLP ile çözülebilen gerçekçi bir problem.

Girişler:
  x1 = Kahve sıcaklığı (0-1 arası normalize)
  x2 = Bekleme süresi (0-1 arası normalize)

Çıkış:
  y = 1 (memnun) eğer kahve ılık-sıcak aralıkta (0.4-0.8) VE bekleme kısa (<0.5)
  y = 0 (memnun değil) diğer durumlarda
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. VERİ SETİ OLUŞTURMA ----------
np.random.seed(42)
N = 400  # toplam örnek sayısı

# Rastgele sıcaklık ve bekleme süresi değerleri
x1 = np.random.uniform(0, 1, N)  # sıcaklık
x2 = np.random.uniform(0, 1, N)  # bekleme süresi

# Kural: ideal sıcaklık aralığı [0.4, 0.8] VE bekleme < 0.5
y = ((x1 >= 0.4) & (x1 <= 0.8) & (x2 < 0.5)).astype(int)

# Gerçekçilik için %5 gürültü (bazı müşteriler kural dışı tepki verir)
noise_mask = np.random.rand(N) < 0.05
y[noise_mask] = 1 - y[noise_mask]

X = np.column_stack([x1, x2])

print(f"Toplam örnek: {N}")
print(f"Memnun müşteri sayısı: {y.sum()}")
print(f"Memnun değil: {N - y.sum()}")
print("\nİlk 5 örnek:")
print("sıcaklık | bekleme | memnun?")
for i in range(5):
    print(f"  {X[i,0]:.2f}   |  {X[i,1]:.2f}   |   {y[i]}")

# ---------- 2. BASİT MLP (Çok Katmanlı Perceptron) ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(a):
    return a * (1 - a)

# Ağ mimarisi: 2 giriş -> 4 gizli nöron -> 1 çıkış
np.random.seed(0)
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 0.5
epochs = 5000
y_col = y.reshape(-1, 1)

for epoch in range(epochs):
    # İleri yayılım
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # Kayıp (binary cross-entropy)
    loss = -np.mean(y_col * np.log(a2 + 1e-8) + (1 - y_col) * np.log(1 - a2 + 1e-8))

    # Geri yayılım
    dz2 = a2 - y_col
    dW2 = a1.T @ dz2 / N
    db2 = dz2.mean(axis=0, keepdims=True)
    dz1 = (dz2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ dz1 / N
    db1 = dz1.mean(axis=0, keepdims=True)

    # Güncelleme
    W1 -= lr * dW1; b1 -= lr * db1
    W2 -= lr * dW2; b2 -= lr * db2

    if epoch % 1000 == 0:
        preds = (a2 > 0.5).astype(int)
        acc = (preds.flatten() == y).mean()
        print(f"Epoch {epoch:4d} | Kayıp: {loss:.4f} | Doğruluk: {acc:.2%}")

# Son performans
preds = (a2 > 0.5).astype(int).flatten()
acc = (preds == y).mean()
print(f"\nFinal doğruluk: {acc:.2%}")

# ---------- 3. KARAR SINIRINI GÖRSELLEŞTİR ----------
xx, yy = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200))
grid = np.column_stack([xx.ravel(), yy.ravel()])
a1_g = sigmoid(grid @ W1 + b1)
a2_g = sigmoid(a1_g @ W2 + b2).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Sol: Ham veri
axes[0].scatter(X[y==1, 0], X[y==1, 1], c='green', label='Memnun', alpha=0.7)
axes[0].scatter(X[y==0, 0], X[y==0, 1], c='red', label='Memnun değil', alpha=0.5)
axes[0].set_xlabel('Kahve sıcaklığı')
axes[0].set_ylabel('Bekleme süresi')
axes[0].set_title('Veri Seti')
axes[0].legend()

# Sağ: Öğrenilen karar sınırı
axes[1].contourf(xx, yy, a2_g, levels=20, cmap='RdYlGn', alpha=0.6)
axes[1].scatter(X[y==1, 0], X[y==1, 1], c='green', edgecolor='k', s=20, label='Memnun')
axes[1].scatter(X[y==0, 0], X[y==0, 1], c='red', edgecolor='k', s=20, label='Memnun değil')
axes[1].set_xlabel('Kahve sıcaklığı')
axes[1].set_ylabel('Bekleme süresi')
axes[1].set_title(f'MLP Karar Sınırı (Doğruluk: {acc:.1%})')
axes[1].legend()

plt.tight_layout()
plt.savefig('kahve_sonuc.png', dpi=100)
print("\nGrafik kaydedildi: kahve_sonuc.png")