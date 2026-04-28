import pandas as pd
import pickle
import numpy as np

# Carregar dados originais para gerar os nomes das colunas pra one hot
dados_originais = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")
colunas_numericas = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

nomes_pra_one_quente = [] #como eu a tinha dito anteriormente, eu ja tinho isso pronto entao nada de mais para comentar
for col in colunas_categoricas:
    for cat in sorted(dados_originais[col].unique()):
        nomes_pra_one_quente.append(f"{col}_{cat}")

#simplesmente mais estrutura
nomes_das_colunas = colunas_numericas + nomes_pra_one_quente
Estrutura_De_Pessoa_Classificada = pd.DataFrame(columns=nomes_das_colunas)
#eu sei q a nomeclatura ta confusa mas confia
Scaler = pickle.load(open('Scaler_Treinado_OBS.pkl', 'rb'))
Cluster = pickle.load(open('cluster_OBS.pkl', 'rb'))
def inferir_paciente(dados_numericos, dados_categoricos):
    #aqui cria os dados numericos pro paciente
    df_num = pd.DataFrame([dados_numericos])
    df_num_scaled = Scaler.transform(df_num)
    df_num_scaled = pd.DataFrame(df_num_scaled, columns=colunas_numericas)
    #aqui faz os categoricos
    df_cat = pd.DataFrame([dados_categoricos])
    df_cat_onehot = pd.get_dummies(df_cat)

   #isso aq é pra garantir q todas as colunas existam na hora de fazer o one hot
    #acabei de lembrar que eu pesquisei qq era cat, e qq tinha a ver com gato quando vi isso a primeira vez
    #mas era só a categoria
    for col in nomes_pra_one_quente:
        if col not in df_cat_onehot.columns:
            df_cat_onehot[col] = 0
    df_cat_onehot = df_cat_onehot[nomes_pra_one_quente]
    paciente_completo = pd.concat([df_num_scaled, df_cat_onehot], axis=1)
    cluster = Cluster.predict(paciente_completo)
    
    return cluster[0] + 1
#aq eu peguei direto do da descricao e simplesmente fiz um que na teoria deve cair ctz no cluster 10
dados_num = {
    'Age': 21.0,
    'Height': 1.62,
    'Weight': 64.0,
    'FCVC': 2.0,
    'NCP': 3.0,
    'CH2O': 2.0,
    'FAF': 0.0,
    'TUE': 1.0
}
dados_cat = {
    'Gender': 'Female',
    'family_history_with_overweight': 'yes',
    'FAVC': 'no',
    'CAEC': 'Sometimes',
    'SMOKE': 'no',
    'SCC': 'no',
    'CALC': 'no',
    'MTRANS': 'Public_Transportation',
    'NObeyesdad': 'Normal_Weight'
}
cluster_paciente = inferir_paciente(dados_num, dados_cat)
print(f"cluster do paciente = {cluster_paciente}")
#Cluster do paciente: 10
#CAIU NO 10