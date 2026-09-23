import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, levene, kruskal, mannwhitneyu
import itertools

# Carregando os dados
df = pd.read_csv('wine_dataset.csv')

# Separando o malic_acid para os três cultivares
malic_A = df[df['cultivar'] == 'Cultivar A']['malic_acid']
malic_B = df[df['cultivar'] == 'Cultivar B']['malic_acid']
malic_C = df[df['cultivar'] == 'Cultivar C']['malic_acid']

alfa = 0.05

print("--- LETRA A: TESTES DE NORMALIDADE (SHAPIRO-WILK) ---")
grupos = {'Cultivar A': malic_A, 'Cultivar B': malic_B, 'Cultivar C': malic_C}
todos_normais = True

for nome, dados in grupos.items():
    stat, p_val = shapiro(dados)
    if p_val >= alfa:
        print(f"{nome}: p-valor = {p_val:.6f} -> NORMAL (p >= 0.05)")
    else:
        print(f"{nome}: p-valor = {p_val:.6f} -> NÃO NORMAL (p < 0.05)")
        todos_normais = False

print("\n--- LETRA A: HOMOGENEIDADE (LEVENE) ---")
stat_lev, p_levene = levene(malic_A, malic_B, malic_C)
if p_levene >= alfa:
    print(f"Levene: p-valor = {p_levene:.6f} -> VARIÂNCIAS HOMOGÊNEAS (p >= 0.05)")
else:
    print(f"Levene: p-valor = {p_levene:.6f} -> VARIÂNCIAS NÃO HOMOGÊNEAS (p < 0.05)")

print("\n--- LETRA B: DECISÃO E TESTE GLOBAL ---")
if todos_normais and p_levene >= alfa:
    print("Veredito: Todos os pressupostos atendidos. Escolha: ANOVA.")
else:
    print("Veredito: Pelo menos um pressuposto falhou. Escolha: KRUSKAL-WALLIS.")
    stat_k, p_kruskal = kruskal(malic_A, malic_B, malic_C)
    print(f"Resultado Kruskal-Wallis: p-valor = {p_kruskal:.2e}")

print("\n--- LETRA C: TESTE POST-HOC (Mann-Whitney ajustado) ---")
comparacoes = [('Cultivar A', malic_A), ('Cultivar B', malic_B), ('Cultivar C', malic_C)]
pares = list(itertools.combinations(comparacoes, 2))
multiplicador = len(pares) 

for (nome1, dados1), (nome2, dados2) in pares:
    stat, p_bruto = mannwhitneyu(dados1, dados2, alternative='two-sided')
    p_ajustado = min(p_bruto * multiplicador, 1.0) # Correção de Bonferroni
    significativo = "SIM" if p_ajustado < alfa else "NÃO"
    print(f"{nome1} vs {nome2} | p-ajustado: {p_ajustado:.4f} | Diferença Significativa? {significativo}")

print("\n--- LETRA D: GERANDO BOXPLOT ---")
print("Feche a janela do gráfico para encerrar o programa.")
plt.figure(figsize=(8, 5))
sns.boxplot(
    x='cultivar', 
    y='malic_acid', 
    data=df, 
    palette=['#D1C4E9', '#F8BBD0', '#B3E5FC']
)
plt.title('Comparação de Ácido Málico entre Cultivares', weight='bold')
plt.ylabel('Ácido Málico')
plt.xlabel('Cultivar')
plt.show()