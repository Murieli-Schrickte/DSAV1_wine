import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, shapiro, spearmanr, kendalltau

# Carregando os dados
df = pd.read_csv('wine_dataset.csv')
colunas_numericas = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[colunas_numericas].corr()

# ========================================================
# LETRA A: Filtrando pares fortes
# ========================================================
pares = corr_matrix.unstack().dropna()
pares_fortes = pares[(abs(pares) > 0.7) & (pares != 1.0)]
pares_unicos = pares_fortes.drop_duplicates()

# Excluindo o par proibido
pares_finais = pares_unicos.drop(
    labels=[('flavanoids', 'total_phenols'), ('total_phenols', 'flavanoids')], 
    errors='ignore'
)

# Descobrindo automaticamente o par com a MAIOR correlação
par_maior_corr = pares_finais.abs().idxmax()
var1, var2 = par_maior_corr
r_max = pares_finais[par_maior_corr]

print("--- RESULTADOS DA QUESTÃO 7 ---")
print(f"Par de maior correlação: {var1} e {var2} (r = {r_max:.4f})\n")

# ========================================================
# LETRA B: Estatística t e Scipy
# ========================================================
n = len(df)
t_manual = r_max * np.sqrt(n - 2) / np.sqrt(1 - (r_max ** 2))
r_scipy, p_pearson = pearsonr(df[var1], df[var2])

print("--- LETRA B ---")
print(f"Estatística t (Manual): {t_manual:.4f}")
print(f"P-valor (SciPy): {p_pearson:.2e}\n")

# ========================================================
# LETRA C: Normalidade, Spearman e Kendall
# ========================================================
print("--- LETRA C ---")
stat1, p1 = shapiro(df[var1])
stat2, p2 = shapiro(df[var2])

print(f"Shapiro-Wilk ({var1}): p-valor = {p1:.6f}")
print(f"Shapiro-Wilk ({var2}): p-valor = {p2:.6f}")

if p1 < 0.05 or p2 < 0.05:
    r_spearman, _ = spearmanr(df[var1], df[var2])
    r_kendall, _ = kendalltau(df[var1], df[var2])
    print(f"\n-> Pelo menos uma variável não é normal. Calculando alternativas:")
    print(f"-> Spearman: {r_spearman:.4f}")
    print(f"-> Kendall: {r_kendall:.4f}\n")

# ========================================================
# LETRA D: Heatmap e Scatter Plot
# ========================================================
print("--- LETRA D ---")
print("Gerando gráficos... (Feche a janela para encerrar)")

paleta = ['#D1C4E9', '#F8BBD0', '#B3E5FC']
plt.figure(figsize=(14, 6))

# 1. Heatmap
plt.subplot(1, 2, 1)
sns.heatmap(corr_matrix, annot=False, cmap='Purples', linewidths=0.5)
plt.title('Heatmap da Matriz de Correlação', weight='bold')

# 2. Scatter Plot com reta
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=df, x=var1, y=var2, hue='cultivar', 
    palette=paleta, edgecolor='black', alpha=0.8
)
sns.regplot(
    data=df, x=var1, y=var2, scatter=False, 
    color='gray', line_kws={'linestyle':'--'}
)
plt.title(f'Dispersão com Reta Ajustada: {var1} x {var2}', weight='bold')

plt.tight_layout()
plt.show()