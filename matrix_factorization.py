#############################
# Model-Based Collaborative Filtering: Matrix Factorization
#############################
#Boşlukları doldurmak için user'lar ve movie'ler için var olduğu varsayılan latent feature'ların ağırlıkları var olan veri ve üzerinden bulunur 
#ve bu ağırlıklar ile var olmayan gözlemler için tahmin yapılır

#*User-Item matrisini 2 tane daha az boyutlu matrise ayrıştırır
#*2 matristen user-ıtem matrisine gidişin latent factor'ler ile gerçekleştiği varsayımında bulunur
#*dolu olan gözlemler üzerinden latent factor'lerin ağırlıklarını bulur.
#*bulunan ağırlıklar ile boş olan gözlemler doldurulur.


# !pip install surprise #bu kütüphane yeni kurman lazım
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy #yeni kurduk
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate
pd.set_option('display.max_columns', None)

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: Modelleme
# Adım 3: Model Tuning
# Adım 4: Final Model ve Tahmin

#############################
# Adım 1: Veri Setinin Hazırlanması
#############################

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") #yukarıdaki iki farklı datasetini burada birleştirdik
df.head()

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)] #movie ıds datafreminin içinde varmı (isin) yukarıdaki movie_ids'lerimiz
sample_df.head()

sample_df.shape

user_movie_df = sample_df.pivot_table(index=["userId"], #sample df üzerinden satırlar user sütun filmler kesişimlerinde puanlar olsun yaptık pivot table'la
                                      columns=["title"],
                                      values="rating")

user_movie_df.shape #out:(76918, 4) bu çıktıdaki 76918 değeri nedir? kullanıcılarımızı ifade ediyor. 4 te filmleri ifade ediyor

reader = Reader(rating_scale=(1, 5)) #surpries kütüphanesinde reader kardeşe dedik ki rate oranımızı bil bak böyle puan skalamız var

data = Dataset.load_from_df(sample_df[['userId',
                                       'movieId',
                                       'rating']], reader) #bu surprise'ın istekleri bitmiyor diyor ki benim kendi özel veri yapım var gel verilerini tanıt bana o şekle çevirelim
#üst satırdaki datayı alıp tipine bakmak istersen okutamazsın mesela tipi özeldir
##############################
# Adım 2: Modelleme
##############################
#makine öğrenmesi ben geliyorum selamın aleyküm dedik inceden bi giriş yaptık

#nasıl yaparız modelleri makine öğrenmesinin öğrendiği test setleri üzerinde kurarız ve daha önce görmediği test seti üzerinde test ederiz
#elimizdeki veriyi aşırı iyi öğrenipte diğer verilerde başarılı olamıyor muyuz?u yanıtlama yolumuzdur bu
trainset, testset = train_test_split(data, test_size=.25) #test setimiz ve train setimiz oranlarını belirledik datamızın içinden
#yukarıdaki kodda scikit değil surprise içinden çağırdığımız  train_test_split ile yaptık işlemi
svd_model = SVD() #bu nedir? matris faktorizasyon yöntemini kullanacağımız fonksiyon
svd_model.fit(trainset) #svd modeli ile trainseti üzerinden fit ederek öğren dedik
#öğrenimi yaptık şimdi test edelim
#yukarıdaki kodda p ve q değerlerini bulduk diyoruz ki şimdi aşağıda hadi bakalım p ve q ne kadar doğru davranıcak diğer verilerde
predictions = svd_model.test(testset)

#tamam yukarıda yaptık kodumuz çalıştı ama ne kadar tutarlıyız reis? hatalarının farklarının karesinin ortalamasının karekökünü alıp öğrenelim?:)
#kare alma sebebi farklar sonucu bazen eksi çıkar
accuracy.rmse(predictions)


svd_model.predict(uid=1.0, iid=541, verbose=True)

svd_model.predict(uid=1.0, iid=356, verbose=True) # bu koda yazıp baktık modelimizle ne tahmin etmişiz puanı diye


sample_df[sample_df["userId"] == 1]
#blade runner'a 4.16 tahmin yaptık adam 4 vermiş
##############################
# Adım 3: Model Tuning
##############################
#optimize etmek performansı iyileştirmek dışsal parametrelere bak ipok sayısı iterasyon sayısına bak mesela 
#mesela svd ye gir bak yukarıda n factors the number of factor default is 20 girili dışsal bir parametre bu belki değiştirebilirsin
#n epochs bir de mesela
#lr_all learning rate
#reg_all bunlar da işte hiperparametre örnekleri dışsal parametredir modelin içinden öğrenilmez
#bu yukarı örnek biraz ileri seviye ama bilgin olsun işte


param_grid = {'n_epochs': [5, 10, 20],
              'lr_all': [0.002, 0.005, 0.007]}
#yukarıdaki kodumuzda dedik ki kardeş bu hiperparametreleri al 5 10 20 dene epoch için 
#parametre gridi ızgarası oluşturduk biz burada diğer dışsal değişkenlerin için de deneyebilirsin

gs = GridSearchCV(SVD,
                  param_grid,
                  measures=['rmse', 'mae'],
                  cv=3,
                  n_jobs=-1,
                  joblib_verbose=True)
# bu yukarıdaki kod kardeşimizle ne diyoruz bir önceki param_grid var ya heh onun tüm kombinasyonlarını hesapla diyoruz 
#o da diyor ki tamam kardeş sen hatanı nasıl değerlendirmek istersin
#biz de diyoruz ki "mae" ile gerçek değerlerle tahmin edilen değerlerin farklarının ortalamalarının karelerini al ya da "rmse" o ortalamanın karekökünü al diyoruz
#cv=3 ne peki çapraz doğrulama nedir bu veri setini 3'e böl 2 parçasıyla model kur 1 parçasıyla test et bunu da işte her parça için yap
#n_jobs da tam performansla yap işlemciyi
#joblid de bana işlemleri yaparken raporlama yap demek

gs.fit(data) #işte fit ettik datamıza

gs.best_score['rmse']
gs.best_params['rmse']


##############################
# Adım 4: Final Model ve Tahmin
##############################

dir(svd_model) #svd modelin içinden neleri alabiliriz diye baktık
svd_model.n_epochs

#alt satırdaki kısımlara tekrar bakarsın gerekirse çok odaklanamadım
svd_model = SVD(**gs.best_params['rmse'])

data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid=1.0, iid=541, verbose=True)






