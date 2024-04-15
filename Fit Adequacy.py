import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import weibull_min, norm, uniform
from scipy.stats import chisquare, ks_2samp

# Gerando os dados normais
nor = pd.DataFrame({'dados': np.random.normal(size=50)})
nor['pad'] = 'normal'

# Gerando os dados uniformes
uni = pd.DataFrame({'dados': np.random.uniform(low=-2, high=2, size=50)})
uni['pad'] = 'uniforme'

# Combining the data
xx = pd.concat([nor, uni], ignore_index=True)

# Creating the plot
sns.set(style="whitegrid", rc={"figure.figsize": (8, 4)})
sns.histplot(data=xx, x="dados", hue="pad", multiple="dodge", bins=40, palette=["blue", "orange"])
sns.kdeplot(data=xx, x="dados", hue="pad", fill=False, linewidth=1.4, color=["blue", "orange"])
plt.title("Comparação")
plt.show()

# Gerando os dados normais
nor = pd.DataFrame({'dados': np.random.normal(size=50)})

# Gerando os dados uniformes
uni = pd.DataFrame({'dados': np.random.uniform(low=-2, high=2, size=50)})

# Defining the breaks
brk = np.arange(-2, 2.5, 0.5)

# Creating the table for the normal distribution
tnor = pd.cut(nor['dados'], bins=brk, labels=False).value_counts().sort_index()

# Creating the table for the uniform distribution
tuni = pd.cut(uni['dados'], bins=brk, labels=False).value_counts().sort_index()

# Printing the tables
print("Table for normal distribution:")
print(tnor)

print("\nTable for uniform distribution:")
print(tuni)

# Defining the observed frequencies
c_tnor = np.array([3, 4, 8, 12, 9, 6, 6, 3])
c_tuni = np.array([5/50, 4/50, 11/50, 7/50, 6/50, 5/50, 10/50, 2/50])

# Normalize frequencies
c_tnor_normalized = c_tnor / np.sum(c_tnor)
c_tuni_normalized = c_tuni / np.sum(c_tuni)

# Creating the table
tab = np.vstack((c_tnor_normalized, c_tuni_normalized))

# Performing the chi-square test
test_QQ = chisquare(c_tnor_normalized, f_exp=c_tuni_normalized)
print("\nChi-square test result:")
print(test_QQ)

# Calculating KS test
ks_result = ks_2samp(nor['dados'], uni['dados'])
print("\nKS test result:")
print(ks_result)

# Plotting ECDF
plt.figure(figsize=(8, 5))
plt.plot(np.sort(nor['dados']), np.arange(1, len(nor) + 1) / len(nor), label='Normal')
plt.plot(np.sort(uni['dados']), np.arange(1, len(uni) + 1) / len(uni), linestyle='dashed', color='red', label='Uniforme')
plt.title("Comparação de ECDF")
plt.legend()
plt.show()

# Ajustando a distribuição Weibull
x = np.random.weibull(a=5, size=10000)  # Gerando dados fictícios Weibull
params = weibull_min.fit(x)  # Ajuste da distribuição Weibull aos dados
print("Parâmetros do ajuste da distribuição Weibull:", params)

# Plotando o histograma com a densidade de probabilidade
plt.figure(figsize=(8, 5))
plt.hist(x, bins=30, density=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.plot(np.linspace(0, max(x), 100), weibull_min.pdf(np.linspace(0, max(x), 100), *params), 'r-', lw=2)
plt.title("Distribuição Weibull Ajustada")
plt.xlabel("Valores")
plt.ylabel("Densidade de Probabilidade")
plt.show()
