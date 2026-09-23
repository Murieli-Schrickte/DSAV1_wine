import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene, f_oneway, kruskal

# Carregando os dados originais
df = pd.read_csv('wine_dataset.csv')

# ==========================================
# LETRA A: Amostra Estratificada (60%, random_state=7)
# ==========================================
amostra_60 = df.groupby('cultivar').sample(frac=0.6, random_state=7)

print("--- LETRA A: CONTAGEM DA AMOSTRA (60%) ---")
print(amostra_60['cultivar'].value_counts())

# ==========================================
# LETRA B: Matriz de Correlação e Variáveis Mais Redundantes
# ==========================================
colunas_num = amostra_60.select_dtypes(include=[np.number]).columns
corr_matrix = amostra_60[colunas_num].corr()

pares = corr_matrix.abs().unstack().dropna()
pares = pares[pares != 1.0].drop_duplicates()

# Excluindo os pares já usados na Questão 7
pares_excluir = [
    ('flavanoids', 'total_phenols'), ('total_phenols', 'flavanoids'),
    ('flavanoids', 'od280/od315_of_diluted_wines'), ('od280/od315_of_diluted_wines', 'flavanoids')
]
pares_filtrados = pares.drop(labels=pares_excluir, errors='ignore')

par_redundante = pares_filtrados.idxmax()
r_max = pares_filtrados.max()

print("\n--- LETRA B: PAR MAIS REDUNDANTE ---")
print(f"Variáveis: {par_redundante[0]} e {par_redundante[1]} (r = {r_max:.4f})")

# ==========================================
# LETRA C: Teste estatístico para o 'alcohol'
# ==========================================
alc_A = amostra_60[amostra_60['cultivar'] == 'Cultivar A']['alcohol']
alc_B = amostra_60[amostra_60['cultivar'] == 'Cultivar B']['alcohol']
alc_C = amostra_60[amostra_60['cultivar'] == 'Cultivar C']['alcohol']

alfa = 0.05
todos_normais = True

print("\n--- LETRA C: PRESSUPOSTOS DO TEOR ALCOÓLICO ---")
grupos_alc = {'Cultivar A': alc_A, 'Cultivar B': alc_B, 'Cultivar C': alc_C}

for nome, dados in grupos_alc.items():
    stat, p_val = shapiro(dados)
    if p_val >= alfa:
        print(f"Shapiro ({nome}): p-valor = {p_val:.6f} -> Normal")
    else:
        print(f"Shapiro ({nome}): p-valor = {p_val:.6f} -> Não Normal")
        todos_normais = False

stat_lev, p_levene = levene(alc_A, alc_B, alc_C)
if p_levene >= alfa:
    print(f"Levene: p-valor = {p_levene:.6f} -> Variâncias Homogêneas")
else:
    print(f"Levene: p-valor = {p_levene:.6f} -> Variâncias Não Homogêneas")

print("\n--- LETRA C: TESTE ESCOLHIDO ---")
if todos_normais and p_levene >= alfa:
    print("Veredito: Usando ANOVA.")
    stat_test, p_test = f_oneway(alc_A, alc_B, alc_C)
    print(f"ANOVA p-valor: {p_test:.2e}")
else:
    print("Veredito: Usando KRUSKAL-WALLIS.")
    stat_test, p_test = kruskal(alc_A, alc_B, alc_C)
    print(f"Kruskal-Wallis p-valor: {p_test:.2e}")