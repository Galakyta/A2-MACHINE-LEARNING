import pandas as pd
import pickle
import numpy as np
#finalmente a gente chega na parte de inferencia, que é na verdade talvez a parte realmente mais pratica e "util" quando você quer usar dados em cluster, que é além de apenas classificar pessoas ou qualquer coisa em grupos
#tambem conseguir pegar um individuo ou objeto novo, e classificar ele dentro dos seus dados, mas isso é até bem simples de fazer
#pra começar a gente vai precisar de um dataframe vazio, mas que tenha a mesma estrutura que a dos nossos centroides.

# Carregar dados originais para gerar os nomes das colunas one-hot
dados_originais = pd.read_csv("ObesityDataSet_.csv")
colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']
#eu taria chamando isso aqui de estrutura do zumano a ser classificado, mas esse dataset fico gande de mais ent eu vo me perder
#se continuar nomeando igual uma besta


#mesmo codigo que eu ja tinha usado anteriormente, nada de mais tmb
nomes_pra_one_quente = []
for col in colunas_categoricas:
    for cat in sorted(dados_originais[col].unique()):
        nomes_pra_one_quente.append(f"{col}_{cat}")

nomes_das_colunas = colunas_numericas + nomes_pra_one_quente

Estrutura_De_Pessoa_Classificada = pd.DataFrame(columns=nomes_das_colunas)
#pra isso aqui a gente só quer a estrutura pra servir de template pro que a gente for passar como nova pessoa a ser classificada

#aqui vem a outra parte importante da gente ter salvado os nossos pkls, a gente precisa normalizar as pessoas novas antes de classificar, porque elas precisam passar pelo scanner exatamente da mesma forma que a gente
#fez com os dados originais, a gente não vai repetir o kmeans pois a gente ja tem o nosso modelo treinado
#o nosso modelo treinado que vai fazer uma previsao de a qual cluster a nova pessoa pertence
# logo a gente loada o nosso normalizador, e depois o nosso cluster ja treinado
Scaler = pickle.load(open('Scaler_Treinado_OBS.pkl', 'rb'))
Cluster = pickle.load(open('cluster_OBS.pkl', 'rb'))

#pra testar eu vou fazer uma amostra nova viciada me baseando no que a gente ja fez no descrevendo centroides
# Valores de exemplo baseados no cluster que vimos anteriormente
Nova_Pessoa = pd.DataFrame([[21.0, 1.62, 64.0, 2.0, 3.0, 2.0, 0.0, 1.0]], columns=colunas_numericas)

Nova_Pessoa = Scaler.transform(Nova_Pessoa)
#A gennte retifica esses dados com o scaler

#agora que ela ta normalizada, é só transformar em um dataframe de novo
Nova_Pessoa = pd.DataFrame(Nova_Pessoa, columns=colunas_numericas)

#agora a gente pode concatenar os dois dataframes, formando a estrutura perfeita pra usar o predict
Nova_Pessoa_Scalada = pd.concat([Nova_Pessoa, Estrutura_De_Pessoa_Classificada]).fillna(0) # esse fill na serve pra transformar em 0 qualquer valor nulo passado, que forma a nossa tabela de one hot pros categoricos

#agora finalmente a gente pode descobrir aonde essa pessoa cai
cluster_da_nova_pessoa = Cluster.predict(Nova_Pessoa_Scalada)
print(f"cluster da nova pessoa: {cluster_da_nova_pessoa[0] + 1}")  # +1 para ficar de 1 a 81