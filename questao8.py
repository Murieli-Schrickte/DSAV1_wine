import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, kruskal

# 1. Carregando os dados e separando os grupos (Letra A)
df = pd.read_csv('wine_dataset.csv')
df_bc = df[df['cultivar'].isin(['Cultivar B', 'Cultivar C'])] 
magnesium_B = df[df['cultivar'] == 'Cultivar B']['magnesium']
magnesium_C = df[df['cultivar'] == 'Cultivar C']['magnesium']

print("--- LETRA B: PRESSUPOSTOS ---")
# 2. Teste de Levene
stat_lev, p_levene = levene(magnesium_B, magnesium_C)
print(f"Levene (Variâncias): p-valor = {p_levene:.6f}")

print("\n--- LETRA C: TESTE DE HIPÓTESE ---")
# 3. Teste de Kruskal-Wallis (já que Shapiro falhou)
stat_kruskal, p_kruskal = kruskal(magnesium_B, magnesium_C)
print(f"Kruskal-Wallis: p-valor = {p_kruskal:.6f}")

if p_kruskal < 0.05:
    print("-> Resultado: Rejeitamos H0. EXISTE diferença significativa no magnésio.")
else:
    print("-> Resultado: Não rejeitamos H0. NÃO existe diferença significativa.")

# 4. Boxplot (Letra D)
plt.figure(figsize=(8, 6))
sns.boxplot(
    data=df_bc, 
    x='cultivar', 
    y='magnesium', 
    palette=['#F8BBD0', '#B3E5FC']
)
plt.title('Comparação de Magnésio: Cultivar B vs Cultivar C', weight='bold')
plt.xlabel('Cultivar')
plt.ylabel('Teor de Magnésio')
plt.show()