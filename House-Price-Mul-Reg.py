import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)

# Multiple Linear Regression
# Birden fazla bağımsız değişken ile bağımlı değişkeni tahmin edeceğiz.

df = pd.read_csv("Housing.csv")
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
check_df(df)

df.drop(["mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning","furnishingstatus", "prefarea", "parking", "bathrooms"], axis=1, inplace=True)
df.head()

# Regresyondan önce bir korelasyona ihtiyaç vardır. Çünkü değişkenler arasında bir ilişki olması gerekmektedir.
# iki değişken arasında ki korelasyon 0 olursa bağımsız değişken bağımlı değişkeni yordamayacaktır. Yani bir değişim
# oluşturmayacaktır.

# Değişkenler arasındaki korelasyon

corr = df.corr()
corr

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
plt.show(block=True)

# Regresyon modeli için değişkenleri oluşturacağız.
df.head()

X = df.drop("price", axis=1) # bağımsız değişkenler
y = df[["price"]] # bağımlı değişken

# Model

# burada eğitim ve test seti oluşturacağız. test split yöntemiyle test setini %20 eğitim setinide
# %80 olacak şekilde rastgele bir örneklem oluşturuyoruz.
# random state'i kullanarak aynı rassalıkta testleri ayırarak aynı sonuçlara ulaşmayı amaçlıyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
df.shape
X_train.shape
X_test.shape
y_train.shape
y_test.shape

# eğitim setiyle model kuracağız, kurduğumuz modelide test seti ile test edeceğiz.

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_
# 222766.05799787
# coefficients (w - weights)
reg_model.coef_
# 3.95249027e+02, 4.17845694e+05, 6.97671091e+05


##########################
# Tahmin(Çoklu Doğrusal Regresyon)
##########################

df.describe().T
# Aşağıdaki gözlem değerlerine göre satışın beklenen değeri nedir?

# area: 3600
# bedrooms: 3
# stories: 2

# 222766.05799787 sabit
# 3.95249027e+02, 4.17845694e+05, 6.97671091e+05 katsayılar


222766.05799787 + 3.95249027e+02 * 3600 + 4.17845694e+05 * 3 + 6.97671091e+05 * 2

yeni_veri = [[4000], [5], [3]]
yeni_veri = pd.DataFrame(yeni_veri).T

reg_model.predict(yeni_veri)
# 5986003.90858105


# Tahmin Başarısını Değerlendirme

# Train RMSE değeri
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))
# 1349775.1377570685

# TRAIN RKARE r**2
reg_model.score(X_train, y_train)
# 0.45046518476787833


# Test RMSE
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
# 1515904.1347880652

# Test RKARE
reg_model.score(X_test, y_test)
# 0.45276057253382995

np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=10,
                                 scoring="neg_mean_squared_error")))

# 1382616.6364054163

# 5 Katlı CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model,
                                 X,
                                 y,
                                 cv=5,
                                 scoring="neg_mean_squared_error")))
# 1597243.7764936232
