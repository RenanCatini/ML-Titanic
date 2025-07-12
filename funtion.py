import pandas as pd
import numpy as np

def atualizacaoTabela(tabela, modeloIdade): 
    # Atualização de tipos (Genero) 
    # Homem: 0, Mulher: 1
    tabela['Sex'] = tabela['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # Atualização de tipos (Local de Embarque)
    # S: 0, C: 1, Q: 2
    tabela = tabela.dropna(subset=['Embarked'])
    tabela['Embarked'] = tabela['Embarked'].map({'S': 0, 'C': 1, 'Q':2})

    tabela = tabela.drop(columns=['Name', 'PassengerId', 'Ticket', "Cabin", 'Embarked'])

    # from sklearn.preprocessing import StandardScaler
    tabela['Fare'] = tabela['Fare'].fillna(0)
    # scaler = StandardScaler()
    # tabela['Fare'] = scaler.fit_transform(tabela[['Fare']])

    idadeNaN = tabela[tabela['Age'].isnull()].copy()

    # Previsão valores NaN
    xA_NaN = idadeNaN.drop(columns=['Age'])
    idadeNaN['Age'] = modeloIdade.predict(xA_NaN)

    idadeNaN.loc[idadeNaN['Age'] > 0, 'Age'] = np.ceil(idadeNaN.loc[idadeNaN['Age'] > 0, 'Age'])

    # Atualizar valores NaN na tabela original
    tabela = tabela.dropna(subset=['Age'])
    tabela = pd.concat([tabela,idadeNaN]).sort_index()
    tabela['Fare'] = tabela['Fare'].fillna(0)

    return tabela


def saidaSobreviventes(listaSobre):
    passageiros = list(range(892, 892 + len(listaSobre)))
    resultado = pd.DataFrame({
        'PassengerId' : passageiros,
        'Survived': listaSobre
    })
    return resultado

