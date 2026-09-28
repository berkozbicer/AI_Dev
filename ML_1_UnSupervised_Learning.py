import os
os.environ["OMP_NUM_THREADS"] = "1"
"""
ML Machine Learning
UnSupervised Learning (Denetimsiz Öğrenme = Feature(+) + Label(-))
Modelin giriş verilerinin(x) bulunduğu ancak bu verilere ait doğru cevapların etiketlerinin
bulunmadığı ML öğrenme türüdür.

Temel Yapı:
    X(Features/Özellikler) --> Model --> Gruplar/Desenler

Bu öğrenmede LABEL yok.

Model:
    - Benzer kayıtları gruplandırabilir.
    - Verideki gizli desenleri bulabilir.
    - Müşteri segmentleri oluşturabilir.
    - Anormal verileri tespit etmeye yardımcı olabilir.

"""

"""
Bir örnekte;
    Müşterilerin:
        1-) Yıllık gelir.
        2-) Aylık harcama.
bu bilgilere bakarak müşterileri 3 gruba ayıracağız.

Ancak Modele;
    Bu müşteri A grubundadır.
    Bu müşteri B grubundadır.
gibi hiçbir doğru cevap vermesin.

Kullanılan algoritma:
    K-Means Clustering
    
Kurulum:
    pip install numpy scikit-learn
    pip install -r requirements.txt

"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    # X(Features/Özellikler) --> MODEL --> Gruplar/Desenler
    # x = Müşteri Özelliği
    # [yillik_gelir, aylik_harcama]
    # DİKKAT!! Burada y=LABEL yoktur.
    x = np.array([
        [20, 10],
        [22, 12],
        [25, 15],

        [50, 45],
        [52, 48],
        [55, 50],

        [85, 80],
        [88, 85],
        [90, 88],
    ])

    print("=== UNSUPERVISED Learning")
    print("\nMüşteri verileri:")
    print(x)

    # --------------------------------------------
    # ÖLÇEKLEME
    # StandartScaler, farklı büyüklükteki sayıları benzer ölçeğe getirir.
    # K-Means, uzaklık hesabı yaptığı için ölçekleme faydalıdır.

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    # --------------------------------------------
    # K_Means Modeli
    # 3 = Kümeleme
    # 42 = Rastgele yapılan işlemlerin her çalışmada aynı sonucu vermesini sağlar.
    # 10 = 10 farklı başlangıcı dene ve en iyisini seç
    model = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10,
    )

    # Model hem öğrenir hem de veri için bir cluster numarasını üretir.
    clusters = model.fit_predict(x_scaled)
    print(clusters)

    print("\nModelin oluşturduğu gruplar")

    for i, customer in enumerate(x):
        income = customer[0]
        spending = customer[1]
        cluster = clusters[i]

        print(
            f"Müşteri: {i + 1}: ",
            f"Gelir: {income}",
            f"Harcama: {spending}",
            f"--> Cluster = {cluster}"
        )

if __name__ == "__main__":
    main()

"""
2 tane feature var. (çalışma saati,  katılım oranı)
z = b +w1*x1 + w2*x2

3 tane feature (çalışma saati,  katılım oranı, sınav_notu)
z= b +w1*x1 +w2*x2 +w3*x3

z = b + W(wi*xi)
"""