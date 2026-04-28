import pandas as pd # pra manipular os dados obvio
from sklearn.preprocessing import MinMaxScaler# pra fazer nosso minmax duh
from sklearn.cluster import KMeans # nosso clusterizador
from scipy.spatial.distance import cdist # vai ser usado para calcular a distancia entre os nossos centroides
import pickle # pra salvar tudo
#import matplotlib.pyplot as plt eu ia usar, mas o windows nn tava afim
'''ERROR: Could not install packages due to an OSError: [WinError 5] Acesso negado: ''
Consider using the `--user` option or check the permissions.
'''
import math, numpy as np # pra usar na hora de aplicar o codigo de otimizacao de clusters
#primeira coisa q eu fiz foi ler a documentacao do csv, e meu deus do ceu vc judiou da gente 2k de row
#907543
dados = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
#sempre importante copiar o relative path, como eu to fazendo agora, msm que de na mesma pq tem so 1 pasta aq eu prefiro, pq mo medo de ferrar tudo
#por diretorio, que e um habito terrivel meu
#e tudo bem pouco importa, mas eu peguei o costume de anotar isso enquanto estudava
#agora, ja da pra instanciar as nossas ferramentas de trabalho, que nese caso vai ser o scaler, infelizmente n vo poder usar o inputer mesmo
#tendo me afeicoado por ele

Scaler = MinMaxScaler()

#agora que o nosso csv ta ingerido, a gente pode começar a tratar ele
'''Gender,Age,Height,Weight,family_history_with_overweight,FAVC,FCVC,NCP,CAEC,SMOKE,CH2O,SCC,FAF,TUE,CALC,MTRANS,NObeyesdad
Female,21,1.62,64,yes,no,2,3,Sometimes,no,2,no,0,1,no,Public_Transportation,Normal_Weight'''
#aqui eu ja me liguei de droppar o target q é algo q eu geralmente esqueço, mas aqui vai ser importante ja que se eu deixar ele ali vai inflar
# e muito a acuracia de um jeito artificial, q eu notei antes mas nn achei nada nos meus outros codigos de estudo, mas a lanna me corrigiu ent suave


colunas_para_dropar_categoricos = [ #eu prefiro sempre separar as colunas antes de droppar pra visualizar, depois eu deixo do nojo que eu quiser
    #mas nessa hora eu prefiro visualizar
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS", "NObeyesdad"
]
dados_numericos = dados.drop(columns=colunas_para_dropar_categoricos)
# e agora podemos normalizar esses dados, mas ants vamos treinar
Scaler_Treinado = Scaler.fit(dados_numericos)
#ja aproveito pra salvar o meu scaler treinado, seguindo o .pkl por bons costumes
pickle.dump(Scaler_Treinado,open("Scaler_Treinado_OBS.pkl", "wb"))
#agora a gente pode dar nosso transform
dados_numericos_tratados = Scaler_Treinado.transform(dados_numericos)
dados_numericos_tratados = pd.DataFrame(dados_numericos_tratados, columns=dados_numericos.columns)

#print(dados_numericos_tratados)

'''           Age    Height    Weight  FCVC       NCP      CH2O       FAF       TUE
0     0.148936  0.320755  0.186567   0.5  0.666667  0.500000  0.000000  0.500000
1     0.148936  0.132075  0.126866   1.0  0.666667  1.000000  1.000000  0.000000
2     0.191489  0.660377  0.283582   0.5  0.666667  0.500000  0.666667  0.500000
3     0.276596  0.660377  0.358209   1.0  0.666667  0.500000  0.666667  0.000000
4     0.170213  0.622642  0.379104   0.5  0.000000  0.500000  0.000000  0.000000
...        ...       ...       ...   ...       ...       ...       ...       ...
2106  0.148443  0.491943  0.689616   1.0  0.666667  0.364070  0.558756  0.453124
2107  0.169850  0.563366  0.707037   1.0  0.666667  0.502565  0.447130  0.299635
2108  0.181362  0.570200  0.706637   1.0  0.666667  0.527097  0.471403  0.323144
2109  0.220467  0.546132  0.704079   1.0  0.666667  0.926170  0.379702  0.293017
2110  0.205632  0.544974  0.705020   1.0  0.666667  0.931757  0.342151  0.357069

[2111 rows x 8 columns]'''

#ok e ta perfeito, temos os nossos numericos hihihihi, e ainda sao 7 e 26 ta bem de buenas
#mas enfim, com isso eu tmbm ja vo anotar que tem 2111 rows, logo o cluster vai ficar imenso pra compilar, mas paciencia
#feito isso temos os campos numericos limpos e separados


#agora pra parte mais chata pra mim que é tratar os categoricos
dados_categoricos_ja_limpos = dados.select_dtypes(include=["object"])
#essa linha aqui eu comecei a usar depois de ter feito alguns datasets a mais do que a gente ja tinha, feito em aula
#eu gosto dela por ser facil pra datasets um pouco maiores, eu usei no do titanic.csv por exemplo e gostei bastante
#print(dados_categoricos_ja_limpos) 
'''
      Gender_Female  Gender_Male  ...  NObeyesdad_Overweight_Level_I  NObeyesdad_Overweight_Level_II
0                 1            0  ...                              0                               0
1                 1            0  ...                              0                               0
2                 0            1  ...                              0                               0
3                 0            1  ...                              1                               0
4                 0            1  ...                              0                               1
...             ...          ...  ...                            ...                             ...
2106              1            0  ...                              0                               0
2107              1            0  ...                              0                               0
2108              1            0  ...                              0                               0
2109              1            0  ...                              0                               0
2110              1            0  ...                              0                               0
'''
# No be yes dad?, nn consigo ler esse target
#mas tanto faz
Dados_Categoricos_Normalizados = pd.get_dummies(dados_categoricos_ja_limpos, prefix_sep="_", dtype=int)
Dados_Normalizados  = dados_numericos_tratados.join(Dados_Categoricos_Normalizados, how='left')


# print(Dados_Normalizados)

'''          Age    Height    Weight  ...  NObeyesdad_Obesity_Type_III  NObeyesdad_Overweight_Level_I  NObeyesdad_Overweight_Level_II
0     0.148936  0.320755  0.186567  ...                            0                              0
  0
1     0.148936  0.132075  0.126866  ...                            0                              0'''

#e perfeito, o dificil ja foi
print("a")
#indo pra clusterizacao agora, como eu disse eu ja anotei aquele numero de rows pra usar aqui
distorcoes = [] #as distorcoes pros calculos, normal
K = range(2, 500) # vai ficar imenso e meio chato de lidar, mas é o ideal, como um sabio uma vez disse
# gente parece q vcs tem medo de dado(a gente tem mesmo)
#mas tudo bem,  

for i in K: # preferi nn inventar moda nenhuma e só usar oq eu ja aprendi pra testar as clusters
    print(i)
    cluster_OBS  = KMeans(n_clusters=i,random_state=42).fit(Dados_Normalizados) #importante fittar pra treinar sempre
    #em seguida a gente pode calcular as distorcoes
    distorcoes.append(
    sum(
        np.min(
            cdist(Dados_Normalizados, cluster_OBS.cluster_centers_, 'euclidean'),
            axis=1  #eu admito que nn peguei mt bem a aritimetica dos calculos, mas ta funcionando ent oq mais a gente pode pedir
        ) / Dados_Normalizados.shape[0]

    )
)

#agora pra parte mais bizarra que é calcular o otimo de clusters, que tmbm vai demorar pra rodar ent
    #to tomando cuidado extra extra extra pra só precisas compilar 1 unica vez

x0 = K[0]
y0 = distorcoes[0]
xn = K[-1]
yn = distorcoes[-1]
distancias = []
for i in range(len(distorcoes)):
    x= K[i]
    y= distorcoes[i]
    numerador = abs(
        (yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0
    )
    denominador = math.sqrt(
        (yn-y0)**2 + (xn-x0)**2
    )
    distancias.append(numerador/denominador)
numero_clusters_otimo = K[distancias.index(np.max(distancias))]
print("numero de clusters mais melhor: ", numero_clusters_otimo)
#sim mais melhor de proposito

#credo djo que ta tudo certo ent vo partir
#brigado por liberar pra fazer com 500 ao inves de 2k no kmeans se nn eu ia ficar aq ate amanha
cluster_OBS = KMeans(n_clusters=numero_clusters_otimo, random_state=42).fit(Dados_Normalizados)
print(cluster_OBS)
pickle.dump(cluster_OBS, open("cluster_OBS.pkl", 'wb'))