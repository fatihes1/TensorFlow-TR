![](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white)
![](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white)
![](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)


# TensorFlow-TR
![Capture1](https://user-images.githubusercontent.com/54971670/127469293-869b2ae8-e12f-47f4-81af-050421f1582b.PNG)

Udacity platformu üzerinde Tensorflow tarafından sunulan `Intro to TensorFlow for Deep Learninh` eğitimi sırasında üzerinde çalıştığım colab dosyalarını içermektedir.
Eğitim toplamda 10 bölümden oluşmaktadır. Eğitime göz atmak isterseniz [buraya](https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187) tıklayabilirsiniz. Bölümler ve içeriği şu şekildedir:

1. **Kursa Hoş Geldiniz** (*Welcome to the Course*) : Bu bölümde yapay zeka, makine öğrenmesi, sinir ağları gibi konulara teorik olarak değinilmiş ve `Python` ve `Colab` için hızlı bir giriş niteliğine sahiptir.
2. **Makine Öğrenimine Giriş** (*Introduction to Machine Learning*) : Makine öğreniminin ne olduğu teorik olarak anlatılmıştır bu teorik bilginin ardından her ne kadar halihazırda bir formülü olmasına rağmen santigrat dereceden fahrenhayta çevirmek üzerine basit bir modelin geliştirildiği bir `colab` dosyası vardır (`B02_01_Santigrattan_Fahrenhayta_Dönüşüm.ipynb`). Birçok makine öğrenimi teriminin tanımı ve ne işe yaradığından bahsedilmiştir (`Loss`, `Weights and biases` gibi). Son olarak yoğun katmanlar (`Dense Layers`) hakkında bilgi sunmaktadır.
3. **İlk Modelimiz : Fashion MNIST** (*Your First Model - Fashion MNIST*) : Giysi görüntüleri ile basit bir görüntü işleme modeli yapılmıştır. Bununla birlikte `ReLu` işlevi hakkında bilgilendirme yapılmış ve `Dense` katmanları ile nasıl çalıştığına dair teorik/pratik bilgiler verilmiştir. Son olarak yaptığımız ilk sıcaklık modeli ile görüntü işleme modeli arasında farkların neler olduğuna değinilmiştir.
4. **Evrişimli Sinir Ağlarına Giriş** (*Introduction to CNNs*) : Evirişimin (`Convolutions`) ne olduğundan bahsedilip nasıl çalıştığı aktarılmıştır. `MaxPooling` yöntemi hakkında da teorik bilgi sağlandıktan sonra daha önceden eğitilen `Fashion MNIST` veri kümesini CNN ile yeni bir model oluşturuldu.
5. **CNN ile Daha İleri Gitmek** (*Going Further with CNNs*) : Bu bölüm içerisinde `Dogs vs. Cats` veri seti ile sınıflandırma üzerine görüntü işleme modeli (RGB-Renkli Görüntüler ile) yapılmıştır. `Softmax` ve `Sigmoid` aktivasyon işlevleri hakkında bilgi sağlanmıştır. Veri seti büyütme (`Image Augmentation`) yöntemleri ile veri seti genişletilmiş ve modele olan etkisi izlenmiştir. Son olarak 5 farklı çiçek türünü içeren bir görüntü sınıflandırmaya daha yer verilmiştir.
6. **Transfer Öğrenme** (*Transfer Learning*) : Daha önceden eğitilmiş modeller üzerinden model eğitmenin verdiği güçten bahsedilmiştir.`MobileNet` kullanılarak daha önceden eğittiğimiz modellere etkisi gözlemlenmiştir. Son olarak önceki bölümde yapılan çiçeklerle görüntü sınıflandırma aktarımlı (*transfer*) öğrenme ile tekrar gerçekleşmiştir.
7. **Model Kaydetme ve Yükleme** (*Saving and Loading Models*) : Bu bölüm içerisinde eğittiğimiz bir modeli nasıl kaydedebileceğimiz konusuna yer verilip `Keras` tarafından desteklenen `.h5` uzantılı dosyalar olarak kaydedip sonrasında modeli tekrar eğitme işlemleri yapılmıştır. En son platform bağımsızlığı için `TensorFlow Saved Model`  olarak kaydedilip yerel makineye indirilmiştir.
8. **Zaman Serisi Tahmini** (*Time Series Forecasting*) : Gürültü, trend, mevsimsellik gibi durumlar ile bir zaman serisi oluşturulmuş ve eğitim boyunca bu zaman serisi üzerinde birçok farklı model denenmiştir. Saf (Naif) tahmin, hareketli ortalama, makine öğrenmesi, RNN, LSTM ve CNN modelleri ile zaman serisindeki en iyi tahmin başarısı için hepsi denenmiştir.
9. **Doğal Dil İşleme : Tokenize Etme ve Gömme** (*NLP: Tokenization and Embeddings*) : Bu bölümde bir metnin makine öğrenmesi modeline hazır hale getirilme sürecine yer verilmiştir.
10. **Doğal Dil İşleme : Yinelenen Sinir Ağları** (*NLP: Recurrent Neural Networks*) : Önceki bölümde makine öğrenimi modeli için hazır hale gelen girdiler (*metinler*) bu bölümde makine öğrenmesi modelleri tarafından işlenmiştir. RNN hakkında bilgi verildikten sonra LSTM üzerinde model geliştirilmiştir. Sonraki aşama olarak LSTM, CNN ve GRUs modellerinin başarılı kıyaslanmıştır. Eğitimin ve bölümün son kısmında ise geliştirilen modeller ile metin oluşturucu (*Text Generation*) oluşturulmuştur. Oluşturulurken Kaggle tarafından sunulan şarkı sözleri veri kümesi kullanılmıştır.

### Colab Dosyalarının Çalıştırılması
- `Colab Files` içerisindeki `.ipynb` uzantılı yerel diskinize indirdikten sona isterseniz **Anaconda** tarafından da sunulan [Jupyter Notebook](https://jupyter.org/) ile açabilirsiniz.
- Diğer opsiyon ise tarayıcı üzerinden *Google Colab* sayfasına gidip indirdiğiniz `.ipybn` uzantılı dosyaları yüklemek olacaktır. Colab sayfasına [**buradan**](https://colab.research.google.com/notebooks/intro.ipynb#recent=true) ulaşabilirsiniz.

