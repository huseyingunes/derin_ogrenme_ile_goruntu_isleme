import random

girdi_ara_sinav = [15, 80, 90, 50, 20, 95, 75]
girdi_final = [75, 45, 95, 50, 20, 65, 25]

cikti = [1, 0, 1, 1, 0, 1, 0]

agirliklar = [0.5, 0.5]

def aktivasyon_fonksiyonu(toplam):
    if toplam >= 50:
        return 1
    else:
        return 0

for i in range(0, 100000):
    agirlik_ara_sinav = random.Random().randint(0, 100) / 100
    agirlik_final = random.Random().randint(0, 100) / 100
    dogru = 0
    for j in range(0, len(girdi_ara_sinav)):
        toplam = girdi_ara_sinav[j] * agirlik_ara_sinav + girdi_final[j] * agirlik_final
        sonuc = aktivasyon_fonksiyonu(toplam)
        if cikti[j] == sonuc:
            dogru += 1

    if dogru == 6:
        print("Ara Sınav ve Final Ağırlıkları :", agirlik_ara_sinav, agirlik_final)
