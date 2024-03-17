import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t, f, ttest_ind, ttest_rel, ttest_1samp

# Samples
Control = pd.Series([21, 28, 24, 23, 23, 19, 28, 20, 22, 20, 26, 26])
experimental = pd.Series([26, 27, 23, 25, 25, 29, 30, 31, 36, 23, 32, 22])

# Histogram and Boxplot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.boxplot([Control, experimental])
plt.xticks([1, 2], ['Control', 'Experimental'])
plt.title('Boxplot')

plt.subplot(1, 2, 2)
plt.hist(Control, color='blue', alpha=0.5, label='Control')
plt.hist(experimental, color='red', alpha=0.5, label='Experimental')
plt.title('Histograma')
plt.legend()

plt.show()

# T-test (Independent)
t_stat, p_value = ttest_ind(Control, experimental)
print("Teste t (Independente):")
print("t =", t_stat)
print("p-value =", p_value)

# Confidence Interval, Mean and Median
diff_mean = Control.mean() - experimental.mean()
n_Control = len(Control)
n_experimental = len(experimental)
s_Control = Control.std()  # Standard deviation (Control)
s_experimental = experimental.std()  # Standard deviation (Experimental)
pool_var = ((n_Control - 1) * s_Control ** 2 + (n_experimental - 1) * s_experimental ** 2) / (n_Control + n_experimental - 2)
se = (pool_var * (1 / n_Control + 1 / n_experimental)) ** 0.5
t_critical = t.ppf(0.975, df=n_Control + n_experimental - 2)  # T critical for a 95% confidence interval
margin_of_error = t_critical * se
conf_int = (diff_mean - margin_of_error, diff_mean + margin_of_error)

print("Intervalo de Confiança (95%):", conf_int)
print("Média Control:", Control.mean())
print("Média Experimental:", experimental.mean())
print("Mediana Control:", Control.median())
print("Mediana Experimental:", experimental.median())

# F Test
f_stat = Control.var() / experimental.var()
p_value_f = f.cdf(f_stat, n_Control - 1, n_experimental - 1)
p_value_f = 1 - p_value_f if f_stat > 1 else p_value_f
p_value_f *= 2

print("Teste F:")
print("F =", f_stat)
print("p-value =", p_value_f)

# Calculation of Cohen's d
cohens_d = diff_mean / pool_var
cohens_d_ci = (cohens_d - 1.96 * ((1 / n_Control) + (1 / n_experimental)) ** 0.5,
               cohens_d + 1.96 * ((1 / n_Control) + (1 / n_experimental)) ** 0.5)

print("Cohen d:")
print("Cohens_d =", cohens_d)
print("CI =", cohens_d_ci)

# Calculation of Hedges' g
hedges_g = cohens_d * (1 - (3 / (4 * (n_Control + n_experimental) - 9)))
hedges_g_ci = (hedges_g - 1.96 * ((1 / n_Control) + (1 / n_experimental)) ** 0.5,
               hedges_g + 1.96 * ((1 / n_Control) + (1 / n_experimental)) ** 0.5)

print("Hedges g:")
print("Hedges_g =", hedges_g)
print("CI =", hedges_g_ci)

# Glass Delta Calculation
glass_delta = cohens_d * ((n_Control + n_experimental) / (n_Control * n_experimental)) ** 0.5
glass_delta_ci = (glass_delta - 1.96 * ((1 / n_Control) + (1 / n_experimental)) ** 0.5,
                  glass_delta + 1.96 * ((1 / n_Control) + (1 / n_experimental)) ** 0.5)

print("Glass delta:")
print("Glass_delta =", glass_delta)
print("CI =", glass_delta_ci)

# T-test (Paired)
t_stat, p_value = ttest_rel(Control, experimental)
print("Teste t (Pareado):")
print("t =", t_stat)
print("p-value =", p_value)

# Confidence Interval, Mean and Median of differences
diff_mean = Control.mean() - experimental.mean()
diff_std = (Control.sub(experimental).std(ddof=1) ** 2 / len(Control)) ** 0.5

# Calculating Confidence Interval Using Critical t
t_critical = t.ppf(0.975, df=len(Control) - 1)  # T critical for a 95% confidence interval
margin_of_error = t_critical * diff_std
conf_int = (diff_mean - margin_of_error, diff_mean + margin_of_error)

print("Intervalo de Confiança (95%):", conf_int)
print("Média das diferenças:", diff_mean)

# Calculation of Cohen's d
cohens_d = diff_mean / Control.sub(experimental).std()
cohens_d_ci = (cohens_d - 1.96 * (1 / len(Control) + 1 / len(experimental)) ** 0.5,
               cohens_d + 1.96 * (1 / len(Control) + 1 / len(experimental)) ** 0.5)

print("Cohen's d:")
print("Cohens_d =", cohens_d)
print("CI =", cohens_d_ci)

# Calculation of Hedges' g
hedges_g = cohens_d * (1 - (3 / (4 * (len(Control) + len(experimental)) - 9)))
hedges_g_ci = (hedges_g - 1.96 * (1 / len(Control) + 1 / len(experimental)) ** 0.5,
               hedges_g + 1.96 * (1 / len(Control) + 1 / len(experimental)) ** 0.5)

print("Hedges' g:")
print("Hedges_g =", hedges_g)
print("CI =", hedges_g_ci)

# Student's t-test for one sample
# Women's height (cm)
altura = [152.0, 153.1, 154.6, 157.8,
          158.8, 159.6, 161.1, 161.6,
          162.7, 163.7, 164.1, 165.5,
          165.8, 168.4, 168.4, 169.1,
          169.1, 170.2, 172.4, 172.9,
          173.1, 173.3, 175.6, 176.9,
          179.0]

# T-test
t_stat, p_value = ttest_1samp(altura, popmean=160)
df = len(altura) - 1  # Degrees of freedom
conf_int = t.interval(0.95, df=df, loc=pd.Series(altura).mean(), scale=pd.Series(altura).std(ddof=1) / len(altura)**0.5)

print("teste-t de Student para uma amostra:")
print(f"t = {t_stat:.4f}, df = {df}, p-value = {p_value:.6f}")
print("Hipótese alternativa: a média verdadeira não é igual a 160")
print(f"Intervalo de Confiança (95%):\n {conf_int[0]:.4f} {conf_int[1]:.4f}")
print("Estimativa da Amostra:")
print(f"Média de x\n  {pd.Series(altura).mean():.4f}")
