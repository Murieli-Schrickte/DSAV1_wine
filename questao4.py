import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Carregando o dataset
df = pd.read_csv('wine_dataset.csv')

# --- ITEM (A): Amostragem Estratificada Proporcional (20% por cultivar) ---
amostra_estratificada = df.groupby('cultivar', group_keys=False).apply(
    lambda x: x.sample(frac=0.2, random_state=42)
)
n_amostra = len(amostra_estratificada)
print(f'Tamanho da amostra estratificada (n): {n_amostra}')


# --- ITEM (B): Comparação visual (População vs Amostra) para 'proline' ---
plt.figure(figsize=(10, 5))

plt.hist(
    df['proline'],
    bins=12,
    color='gray',
    alpha=0.5,
    edgecolor='black',
    label='População Completa (178)',
)
plt.hist(
    amostra_estratificada['proline'],
    bins=12,
    color='purple',
    alpha=0.7,
    edgecolor='black',
    label=f'Amostra Estratificada (n={n_amostra})',
)

plt.title('Comparação da Variável Proline: População vs Amostra Estratificada')
plt.xlabel('Teor de Proline')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# --- ITEM (C): 1.000 Reamostragens com reposição (Bootstrap) e TCL ---
np.random.seed(42)
medias_reamostragem = []

for _ in range(1000):
  amostra_bootstrap = np.random.choice(
      df['proline'], size=n_amostra, replace=True
  )
  medias_reamostragem.append(np.mean(amostra_bootstrap))

# Cálculos estatísticos para o TCL
std_observado_medias = np.std(medias_reamostragem, ddof=1)
std_populacao = np.std(df['proline'], ddof=1)
erro_padrao_teorico = std_populacao / np.sqrt(n_amostra)
proporcao = std_populacao / std_observado_medias

print(f'Desvio-padrão observado das médias: {std_observado_medias:.2f}')
print(f'Erro-padrão teórico (TCL): {erro_padrao_teorico:.2f}')
print(f'O desvio populacional é cerca de {proporcao:.1f} vezes maior.')

# Plotando o histograma das 1.000 médias
plt.figure(figsize=(9, 5))
plt.hist(
    medias_reamostragem,
    bins=25,
    color='teal',
    edgecolor='black',
    alpha=0.8,
)
plt.axvline(
    np.mean(df['proline']),
    color='red',
    linestyle='dashed',
    linewidth=2,
    label='Média da População',
)
plt.title('Histograma das Médias de 1.000 Reamostragens (Proline)')
plt.xlabel('Média Amostral de Proline')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()