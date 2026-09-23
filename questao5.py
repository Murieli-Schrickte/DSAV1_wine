import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados
df = pd.read_csv('wine_dataset.csv')
colunas_quimicas = df.columns.drop('cultivar')

# ==========================================
# ITEM (a): Medidas, CV e Top 3 Variáveis
# ==========================================
media = df.groupby('cultivar')[colunas_quimicas].mean()
desvio = df.groupby('cultivar')[colunas_quimicas].std()
mediana = df.groupby('cultivar')[colunas_quimicas].median()

# Coeficiente de Variação (%)
cv = (desvio / media) * 100

# Pegando as 3 variáveis com maior média de CV
cv_medio_por_variavel = cv.mean(axis=0)
top_3_vars = cv_medio_por_variavel.nlargest(3).index.tolist()

print('--- ITEM (A): TOP 3 VARIÁVEIS COM MAIOR CV MÉDIO ---')
for i, var in enumerate(top_3_vars, 1):
    print(f'{i}. {var}')
print('\n')

# ==========================================
# ITEM (b): Contagem de Outliers (1.5 x IQR)
# ==========================================
resultados_outliers = {}

for var in top_3_vars:
    resultados_outliers[var] = {}
    for cultivar, group in df.groupby('cultivar'):
        Q1 = group[var].quantile(0.25)
        Q3 = group[var].quantile(0.75)
        IQR = Q3 - Q1
        
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        
        # Filtrando e contando os outliers
        outliers = group[(group[var] < limite_inferior) | (group[var] > limite_superior)]
        resultados_outliers[var][cultivar] = len(outliers)

print('--- ITEM (B): CONTAGEM DE OUTLIERS POR CULTIVAR ---')
print(pd.DataFrame(resultados_outliers))


# ==========================================
# ITEM (c): Boxplots Comparativos
# ==========================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, var in enumerate(top_3_vars):
    sns.boxplot(
        x='cultivar',
        y=var,
        data=df,
        ax=axes[i],
        palette=['#D1C4E9', '#F8BBD0', '#B3E5FC'],
        hue='cultivar',
        legend=False,
    )
    axes[i].set_title(f'Boxplot: {var}', fontsize=12, weight='bold')
    axes[i].set_xlabel('Cultivar', fontsize=10)
    axes[i].set_ylabel(var, fontsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()