
#####KURAL TABANLI MÜŞTERİ SEGMENTASYONU#####

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı



#############################################



# GÖREV 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.

import pandas as pd
import numpy as np
import seaborn as sns

df=pd.read_csv(r'C:\Users\SARU\Desktop\VBO AFTER\Python Programming for Data Science\persona.csv')

df.head()
df.shape
df.info()
df.columns
df.dtypes
df.describe().T
df.isnull().sum()
df['COUNTRY'].value_counts()
df.nunique()

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?

df['SOURCE'].nunique()    ----->#2
df['SOURCE'].value_counts()  ----->#android:2974,ios:2026


# Soru 3: Kaç unique PRICE vardır?

df['PRICE'].unique()
df['PRICE'].nunique()  ----->#6

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?

df.groupby('PRICE')['PRICE'].count()
df['PRICE'].value_counts()


# Soru 5: Hangi ülkeden kaçar tane satış olmuş?

df.groupby('COUNTRY')['PRICE'].count()
df[['COUNTRY','PRICE']].value_counts() ----->#ülkelere göre hangi fiyattan kaçar urun satıldı


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?

df.groupby('COUNTRY')['PRICE'].sum()


# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?

df.groupby('SOURCE')['PRICE'].count()



# Soru 8: Ülkelere göre PRICE ortalamaları nedir?

df.groupby('COUNTRY')['PRICE'].mean()


# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?

df.groupby('SOURCE')['PRICE'].mean()


# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?


df.groupby(['COUNTRY','SOURCE'])['PRICE'].mean()



#############################################



# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?

df.groupby(['COUNTRY','SOURCE','SEX','AGE'])['PRICE'].mean()

df.groupby(['COUNTRY','SOURCE','SEX','AGE']).agg({'PRICE':'mean'}).T



#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız. Çıktıyı agg_df olarak kaydediniz.

agg_df=(df.groupby(['COUNTRY','SOURCE','SEX','AGE']).agg({'PRICE':'mean'})).sort_values('PRICE', ascending= False)
print(agg_df)


# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.

agg_df=(df.groupby(['COUNTRY','SOURCE','SEX','AGE']).agg({'PRICE':'mean'})).sort_values('PRICE', ascending= True)
print(agg_df)




#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.

# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.

agg_df.shape ----->#(348,1)

df.shape  ----->#(5000,5)

#burada df içerisinde 5 kolon vardıancak agg_df 'te 1 kolongörünüyor.
# agg_df içerisindeki değişkenler index olarak algılanıyor. Bunların  değişkne ismine çevirilmesi gerekiyor.

agg_df=agg_df.reset_index()
agg_df.shape  ----->#(348,5)



#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.

# Age sayısal değişkenini kategorik degiskene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'


agg_df['AGE'].dtype  ----->#int64
agg_df['AGE'].nunique()
agg_df['AGE'].unique()

agg_df['AGE_NEW']=pd.cut(agg_df['AGE'],[0,18,25,30,40,70],labels=['0_18','19_25','26_30','27_40','41_70'])
agg_df.head()



#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.

# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18


agg_df['CUSTOMER_LEVEL_BASED']=[col[0].upper() + '_' + col[1].upper() + '_' + col[2].upper() + '_' + col[5].upper() for col in agg_df.values]
agg_df.head()

agg_df=agg_df.groupby('CUSTOMER_LEVEL_BASED')['PRICE'].mean()  ---->#aynı yas grubuna giren farklı yaşlarda kişiler olabilir.Guba yönelik bir fiyat verebilmek için ortamlama alıyoruz



#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.


# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz


agg_df.shape ---->#tüm sutunlar index olarak gözüktüğü için indexleri değişkene ceviriyoruz

agg_df=agg_df.reset_index()

agg_df['SEGMENT']=pd.qcut(agg_df['PRICE'],4,labels=['D','C','B','A'])

agg_df.columns

# agg_df.drop('index',axis=1,inplace=True)

agg_df.groupby('SEGMENT').agg({'PRICE':['sum','max','mean']})

agg_df.head()




#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

#agg_df['CUSTOMER_LEVEL_BASED'].unique()
new_user='TUR_ANDROID_FEMALE_26_30'
agg_df[agg_df["CUSTOMER_LEVEL_BASED"] ==  new_user ]  ---->#TUR_ANDROID_FEMALE_26_30  30.785714       D


###############################################################################################################################

