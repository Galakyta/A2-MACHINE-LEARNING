import pickle
import pandas as pd
import numpy as np

#a partir do codigo de cluster a gente pode começar a proxima etapa de descrever nossos centroides, pra isso vale a pena pensar em centroides como os nossos grupos, o que eles tem em comum, porque eles se agrupam
# ou de forma mais branda, oq os dados podem nos dizer

#pra isso a gente carrega nossos dados processados anteriormente
cluster = pickle.load(open("cluster_OBS.pkl", "rb"))

#em seguida a gente vai transformar ele em algo legivel de novo, assim a gente pode analisar isso de forma mais apropriada e identificar e nomear os padroes contidos nos dados
#primeiro a gente precisa nomear as colunas novamente, da pra extrair direto do csv

dados_originais = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
# a gegnte vai usar elas na hora de converter os centroides que a gente tem, em um dataframe legivel
'''NUMERICOS PRIMEIRO E CATEGORICOS DEPOIS'''
#eu nn sei se é exatamente uma regraaaaaaaa issso, mas eu tenho usado assim pra manter uma logica entre os meus codigos
#particularmente eu tenho mt disso de ter que firmar uma lei pra mim mesmo ou eu acabo me perdendo bastante

nomes_pra_one_quente = [] #nn caçoar dos meus nomes estupidos
for col in colunas_categoricas:
    for cat in sorted(dados_originais[col].unique()):
        nomes_pra_one_quente.append(f"{col}_{cat}")
#eu gosto bastante de fazer isso, ja que assim eu consigo distinguir melhor na hora de apresentar, apesar de ser
        #bem chato na hora de implementar, mas eu ja tinha pronto na consulta ent tudo certo

nomes_das_colunas = colunas_numericas + nomes_pra_one_quente
print(f"colunax no total = {len(nomes_das_colunas)}")

centroides = pd.DataFrame(cluster.cluster_centers_, columns=nomes_das_colunas)
#assim a gente transforma os centroides em um dataframe com as colunas corretas
#mas agora como a gente tinha tratado anteriormente, é preciso segmentar o nosso dataframe em 2 partes, 1 para numericos e outra para categoricos 

#eu ja vo explicar pq eu tirei o target calma la
colunas_categoricas_sem_target = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
nomes_onehot_sem_target = [nome for nome in nomes_pra_one_quente if not nome.startswith('NObeyesdad_')] #eu tmbm decidi tirar o target
#pq eu achei q poderia influenciar no treinamento e acabar deixando o trabalho todo do onehot de nn criar nenhum padrao ficticio inutil

dados_numericos_normalizados = centroides[colunas_numericas]
dados_categoricos_normalizados = centroides[nomes_onehot_sem_target]

#sim a gente puxa o scaler
Scaler = pickle.load(open('Scaler_Treinado_OBS.pkl', 'rb'))
# a partir dele, a gente pode fazer uma operação inversa de normalização, que é pegar os nossos dados e reconstruir eles, dessa forma a gente vai ter um dataframe com os nossos dados originais, mas classificados em um cluster
dados_num = Scaler.inverse_transform(dados_numericos_normalizados)
dados_num = pd.DataFrame(dados_num, columns=dados_numericos_normalizados.columns)
#em seguida a gente desfaz o get dummies, e finalmente enquadra os clusters em alguma classificacao categorica

categorias_resultado = []

for indice in range(len(centroides)):
    categorias_de_cluster = []
    pos = 0
    for col in colunas_categoricas_sem_target:
        n = len(sorted(dados_originais[col].unique()))
        cat_idx = np.argmax(dados_categoricos_normalizados.iloc[indice, pos:pos+n])
        categorias_de_cluster.append(sorted(dados_originais[col].unique())[cat_idx])
        pos += n
    categorias_resultado.append(", ".join(categorias_de_cluster))
dados_cat = pd.DataFrame(categorias_resultado, columns=['Class'])

#ok alguns detalhes importantes, inicialmente na hora da descrição. não tem como a gente saber pra qual cluster cada dado categorico ira pertencer, mas vale lembrar que a gente deu um int pra cada um deles no treinamento
#com o asntype, com isso agora a gente pode reverter o processo, round 0 vai fazer todo mundo ir ou pra 0, ou pra 1, nesse caso o maior valor vira 1, e o resto 0, como se fosse uma eleição ou leilão pra ver qual dado
#categorico fica com cada cluster, exemplo: pra um dos clusters tem os seguintes valores: setosa 0,8 versicolor 0,2 e virginica 0,3 // esses valores são os valores predominantes nos centroides, então o maior vira 1 e o resto 0
#dessa forma a gente consegue classificar com precisão
#isso basicamente tira as variaveis "ficticias" do getdummies, transforma tudo em 0, e transforma elas em um int, ou seja elas viram categoria 1 2 3 4 etc, por fim a ggente coloca a coluna class nelas como classificacao
#em seguida a gente junta ambos novamente
cluster_obesidade_dados = dados_num.join(dados_cat)
print(cluster_obesidade_dados)

'''colunax no total = 38
          Age    Height      Weight      FCVC       NCP      CH2O       FAF       TUE                                              Class
0   23.297535  1.593686   65.606684  2.566483  2.291351  1.865957  1.018967  0.841990  Female, no, yes, Sometimes, no, no, Sometimes,...
1   25.765281  1.637855  110.103601  3.000000  3.000000  2.619039  0.054077  0.339578  Female, yes, yes, Sometimes, no, no, Sometimes...
2   31.909327  1.766731  116.870964  2.296472  2.973601  2.100067  0.514833  0.693515  Male, yes, yes, Sometimes, no, no, Sometimes, ...
3   20.750000  1.626250   58.054167  2.250000  2.750000  1.625000  0.916667  0.791667  Female, yes, yes, Frequently, no, no, no, Publ...
4   18.372584  1.739040   52.497908  2.115456  3.103276  2.003274  0.796313  1.540140  Male, yes, yes, Sometimes, no, no, Sometimes, ...
..        ...       ...         ...       ...       ...       ...       ...       ...                                                ...
76  20.818182  1.578182   53.863636  2.000000  3.454545  1.909091  1.545455  0.909091  Female, no, yes, Frequently, no, no, Sometimes...
77  23.353619  1.822119   85.186406  2.234015  2.396037  1.568128  0.965161  0.644195  Male, yes, yes, Sometimes, no, no, Sometimes, ...
78  27.000000  1.673333   69.833333  2.333333  3.166667  2.500000  1.666667  1.166667  Female, yes, no, Frequently, no, yes, Frequent...
79  25.531996  1.796229  117.875683  1.904186  2.998243  2.046712  1.311596  0.404613  Male, yes, yes, Sometimes, no, no, Sometimes, ...
80  20.414692  1.770531  140.105453  3.000000  3.000000  2.595106  1.431889  0.780099  Female, yes, yes, Sometimes, no, no, Sometimes...
'''
# e o resultado esperado deve ser esse 
# q por cima pelo menos ta certo