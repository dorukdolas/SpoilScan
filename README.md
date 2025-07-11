keras dosyam: https://drive.google.com/file/d/10N-_A3a2pzjtLQYSz0vDUGOGTA5byso8/view?usp=sharing
veri setim: https://drive.google.com/file/d/1gUTwGQR945k17gttbx7-Z8bNSnVqmRQ_/view?usp=sharing


Burada laerning_rate değerini deniyorum: <img width="949" height="181" alt="Ekran görüntüsü 2025-07-11 185017" src="https://github.com/user-attachments/assets/33b39109-c8c3-48f7-9f6b-2e519283202e" />
<img width="956" height="882" alt="image" src="https://github.com/user-attachments/assets/540e74c4-8d13-4b15-ba14-55fe5dd023f9" />
<img width="961" height="205" alt="Ekran görüntüsü 2025-07-11 193457" src="https://github.com/user-attachments/assets/bbe3007f-23dd-40f4-8a36-e09a34cbaaeb" />
<img width="623" height="442" alt="Ekran görüntüsü 2025-07-10 205704" src="https://github.com/user-attachments/assets/b478b260-b4ce-42d9-8e68-4ee70fdabeab" />
<img width="1101" height="441" alt="Ekran görüntüsü 2025-07-10 184419" src="https://github.com/user-attachments/assets/bbe790f0-3fd2-471d-b759-260a0ce2db3d" />
Normalizasyon yapmadan önce: <img width="1092" height="127" alt="Ekran görüntüsü 2025-07-10 182701" src="https://github.com/user-attachments/assets/f1e66550-2279-435b-acb5-430fcf1b96f5" />
<img width="948" height="699" alt="Ekran görüntüsü 2025-07-11 025319" src="https://github.com/user-attachments/assets/88ddcff3-d8b0-48fb-9c69-8da07f2b87c6" />

Bunları süreç boyunca screenshot alıp kaydetmişim fakat üstünden zaman geçtiği için hangisinde tam ne yapıyordum hatırlayamadım. Her bir eğitimde yeni bir keras dosyası oluşturmayı unuttum ama görsellerde son değerler mevcut belki o kısım yardımcı olabilir.

Açıklama:
Google Colab kullandığım için öncelikle veri setimi drive a yükledim ve .zip formatındki veriyi içeriye aktardım.
zip dosyası content/dataset/ klasörüne açılıyor ve eğitim için hazır hale geliyor.
İki veri seti birleştirilerek toplamda yaklaşık 52.000 görsel içeren bir sınıflandırma veri seti oluşturulmuştur.
Görseller 224x224 olacak şekilde resize edildi. (Github için yeterli olacağını düşünmüştüm ama yükleme sınırları düşündüğümden azmış)
CNN modeli oluştururken, görselin küçük parçalarını tarayarak kenarları, çizgileri ve dokuları yakalamaya çalışan bir katman ekledim.
Filtre sayısını 32 yaptım çünkü başlangıçta bu kadarı yeterli oluyordu. 3x3 boyutu ise CNN’lerde genellikle kullanılan standart bir filtredir.
ReLU aktivasyonunu kullandım çünkü bu sayede negatif değerleri sıfıra çekerek modelin öğrenmesini hızlandırıyor. Daha önce bu aktivasyonun 1. ara raporu yazarken yaptığım araştırmalarda CNN’de kullanıldığını görmüştüm.
Sonrasında maxpooling kullandım çünkü hesaplama yükünü azaltmak ve önemli bilgileri tutup gereksizleri atmak istedim.
İkinci katmanda filtre sayısını artırdım çünkü model biraz daha karmaşık yapıları öğrenmeye hazır hale geliyor. Her ne kadar çok karmaşık bir görüntüm olmasa da renk geçişleri, meyvenin şekli gibi niteliklerle çürük meyve tespitinin daha iyi yapılacağını düşündüm.
Flatten ile 2 boyutlu filtre çıktısını 1 boyutlu diziye indirdim. Bunu yapmak gerekiyor çünkü bundan sonra gelecek tam bağlantılı katmanlar (Dense) sadece 1 boyutlu verilerle çalışıyor.
'Dense' katmanı öğrenilen öznitelikleri kullanarak hangi sınıfa ait olduğunu belirlemek için çalışıyor.
Dense katmanındaki sayı, modelin öğrenilen bilgileri yorumlamasını sağlıyor diyebilirim. Çok teknik detayını bilmesem de, bu sayının hem doğruluk hem de overfitting konusunda etkili olduğunu gözlemledim. Deneyerek 64’ün dengeli bir sonuç verdiğini gördüm.
İkili sınıflandırma yaptığım için binary_crossentropy kullandım. Çünkü hem Fresh (0) hem de Rotten (1) olacak şekilde net bir ayrım var. 
Bu fonksiyon, modelin tahminlerinin ne kadar yanlış olduğunu hesaplıyor.
Callback, modelin eğitim sırasında bazı koşullara göre otomatik olarak müdahale etmesini sağlıyor. Kodda iki tane callback kullandım:
EarlyStopping: Modeli eğitirken bazen validation loss değeri bir noktadan sonra düzelmiyor hatta kötüleşebiliyor. O yüzden EarlyStopping kullanarak modelin gereksiz yere eğitilmesini engelledim.
Burada "val_loss"u izliyor ve 3 epoch boyunca gelişme olmazsa eğitimi otomatik durduruyor. Böylece zaman kazanıyorum ve overfitting riskini azaltıyorum.
Bunu ilk başta cpu ile eğitim yaptığımdan kullandım çünkü her bir epoch için yaklaşık 7-8dk beklemem gerekiyordu ve sonuç artık sabitleşirse bu fonksiyon sayesinde otomatik durdurma yapabiliyorum.
ModelCheckpoint: Modeli eğitirken her epoch'ta çıkan sonuçları kaydetmek istemediğimden sadece en iyi sonucu verdiği haliyle saklamak istedim. Bu yüzden ModelCheckpoint kullandım.
bu fonksiyon da "val_loss" değeri en düşük olduğunda, model otomatik olarak "model.keras" adıyla kaydediliyor. Böylece eğitim bitse bile elimde en iyi hali kalıyor.
Bu iki fonksiyon ilk başta cpu kullandığım için benim için çok önemliydi çünkü başka bir işle uğraşırken bir yandan da modelin eğitimini takip etmem kolaylaşıyordu.
