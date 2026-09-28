import os
os.environ["OMP_NUM_THREADS"] = "1"
"""
ML Machine Learning
Supervised Learning (Denetimli Öğrenme = Feature(+) + Label(+))
Modelin hem giriş verileri(x)
hem de bu verilere ait doğru cevapları(Etiketleri[Label]: Y)

Temel Yapı:
    X(Features/Özellikler) --> Model --> y(Label/Etiketler)

Örneğin:
    - E-posta spam mı değil mi?
    - Ev fiyatları ne kadar olur?
    - Bu müşteriye borç beyaz eşya verilir mi?

Öğrenme türü olan label vardır.

"""

"""
Bir öğrencinin;
    1-) Günlük çalışma saati.
    2-) Derse katılım yüzdesi.
bu bilgilere bakarak sınavı geçip geçmediğini tahmin edelim.

Label:
    0: Kaldı
    1: Geçti
    
Kullanılan algoritma:
    Logistic Regression
    
Kurulum:
    pip install numpy scikit-learn
    pip install -r requirements.txt

"""

import numpy as np
from sklearn.linear_model import LogisticRegression

def main():
#  X(Features/Özellikler) --> MODEL --> y (Label/Etiketler)
    x = np.array([
        [1, 30],
        [2, 40],
        [2, 50],
        [3, 55],
        [4, 60],
        [5, 65],
        [6, 75],
        [7, 85],
        [8, 90],
        [9, 95]
    ])

    #  y (Label/Etiketler)
    # 0 = Kaldı, 1= Geçti

    # NOT: Supervised Learning'in en önemli özelliği ==>
    # X verileriyle birlikte y etiketlerinin bulunmasıdır(SL)
    y = np.array([
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1
    ])

    print("=== SUPERVISED LEARNING (Features(+) Label(+))")
    print("\nX - Öğrenci Özellikler(Features)")
    print(x)

    print("\ny - Label(Etiketler)")
    print(y)

    """
    - Model Oluşturma
    - LogisticRegression bir sınıflandırma algoritmasıdır.
    - 0 -> Kaldı
    - 1 -> Geçti

        LogisticRegression, iki veya daha fazla sınıfın hangisine ait olduğunu tahmin etmek için kullanılan
    sınıfın algoritmasıdır.
    """
    model = LogisticRegression()

    # Modeli Eğitme
    # Model hem özellikleri hem de doğru cevapları görsün.
    # Bu ilişkide; çalışma saati + katılım oranı --> Geçti/Kaldı
    model.fit(x, y)

    # Instance
    # Örnek; Öğrenci 6 saat çalışıyor, derse katılım oranı %80

    new_student = np.array([[6, 80]])
    prediction = model.predict(new_student)[0]

    # Tahmin olasılıkları
    probabilities = model.predict_proba(new_student)[0]

    print("\nYeni Öğrenci")
    print("Çalışma Saati: 6 Saat")
    print("Derse Katılım: %80")
    print("\nModel Tahmini: ", prediction)

    # Conditional
    if prediction == 1:
        print("Sonuç: Öğrencinin GEÇMESİ bekleniyor")
    else:
        print("Sonuc: Öğrencinin KALMASI bekleniyor")

    print("\nOlasılıklar:")
    print(f"Kalma Olasılığı: %{probabilities[0] * 100:.2f}")
    print(f"Geçme Olasılığı: %{probabilities[1] * 100:.2f}")

    print("=============== OZET ==============")
    print("Supervised Learning LABEL vardır.")
    print("NOT: !!Model geçmişteki doğru cevapları öğrenir.!!")
    print("Bo örnekte label; 0: Kaldı, 1: Geçti")

if __name__ == "__main__":
    main()

"""
Logistic Regression: 

z = 0.1422 x çalışma saati + 0.7122 x katılım yüzdesi - 45.1082
P(Geçti): 1 / (1 + e^(-z))
P >= 50 --> Geçti ==> z = 0
%99.9997
P(Kaldı)

"""