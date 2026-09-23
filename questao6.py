import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro

# 1. Carregando os dados e definindo as variáveis
df = pd.read_csv('wine_dataset.csv')
vars_q6 = ['magnesium', 'malic_acid', 'proanthocyanins', 'hue']

# 2. Criando uma grade 2x2 para colocar os 4 gráficos juntos
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

# Paleta com as cores anteriores + 1 tom menta pastel para o quarto gráfico
paleta = ['#D1C4E9', '#F8BBD0', '#B3E5FC', '#B2DFDB']

# 3. Loop para desenhar cada histograma + KDE no seu respectivo espaço
for i, var in enumerate(vars_q6):
    sns.histplot(
        data=df, 
        x=var, 
        kde=True,          
        ax=axes[i],        
        color=paleta[i],   
        edgecolor='black'  
    )
    
    # Deixando o gráfico alinhado
    axes[i].set_title(f'Histograma e KDE: {var}', weight='bold')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Frequência')

# 4. Ajusta o espaçamento para os gráficos não ficarem grudados
plt.tight_layout()

# Exibe tudo na tela
plt.show()

print("--- TESTE DE SHAPIRO-WILK ---")

variaveis_teste = ['hue', 'malic_acid']

for var in variaveis_teste:
    stat, p_valor = shapiro(df[var])
    
    print(f"\nVariável: {var}")
    print(f"Estatística W: {stat:.4f}")
    print(f"P-valor: {p_valor:.6f}")
    
    # Regrinha do 0.05
    if p_valor > 0.05:
        print("-> Resultado: Distribuição Normal (p > 0.05)")
    else:
        print("-> Resultado: Distribuição NÃO Normal (p < 0.05)")