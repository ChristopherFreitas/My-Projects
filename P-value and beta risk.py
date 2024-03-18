import pandas as pd
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest

# Sample
strength = pd.Series([14.95, 19.5, 14.46, 13.54, 14.31, 15.36, 12.08, 16.62, 12.1, 8.57, 12.01, 13.49, 13.7, 13.2, 10.76, 15.3, 9.05, 9.58, 14.41, 12.57])
value = 12.5
alpha = 0.05
n = len(strength)

# Test Statistic and P-value
mean_sample = strength.mean()
std_dev = strength.std()
std_error = std_dev / (n ** 0.5)

print("sample mean =", round(mean_sample, 3), "; standard deviation =", round(std_dev, 3), "; standard error =", round(std_error, 3))

Zcrit = norm.ppf(1 - alpha)
print("critical value =", round(Zcrit, 3))

z = (mean_sample - value) / std_error
print("test statistic =", round(z, 3))

pval = norm.cdf(-z)
print("p-value =", round(pval, 3))

# If the true population mean is 14 kgf/cm², what is the beta risk?
# Beta Risk
critical_strength = Zcrit * std_error + value
print("If the true population mean is 14 kgf/cm², what is the beta risk?")
print("strength value corresponding to critical value =", round(critical_strength, 3))

zBeta = (critical_strength - 14) / std_error
beta_risk = norm.cdf(zBeta)
print("z for Beta =", round(zBeta, 3), "beta risk =", round(beta_risk, 3))

# If the true population mean is 11 kgf/cm², what is the beta risk?
# Z-test
z_stat, p_value = ztest(strength, value=value, alternative='larger', ddof=1)

# Sample mean
mean_x = pd.Series(strength).mean()

# Confidence Interval (95%)
ci = (norm.interval(0.95, loc=mean_x, scale=std_dev / len(strength) ** 0.5)[0], None)

print("If the true population mean is 11 kgf/cm², what is the beta risk?")
print("Z-test Statistics:")
print("z =", round(z_stat, 4))
print("p-value =", round(p_value, 4))
print("alternative hypothesis: true mean is greater than", value)
print("Confidence Interval (95%):", ci)
print("Mean of x:", mean_x)

# A type of steel must have strength LESS THAN 12.5 kgf/cm². Test this hypothesis with a significance level of 5% with the sample below.
# Z-test
z_stat, p_value = ztest(strength, value=value, alternative='smaller', ddof=1)

# Sample mean
mean_x = pd.Series(strength).mean()

# Confidence Interval (95%)
ci = (None, norm.interval(0.95, loc=mean_x, scale=std_dev / len(strength) ** 0.5)[1])

print("A type of steel must have strength LESS THAN 12.5 kgf/cm². Test this hypothesis with a significance level of 5% with the sample below.")
print("Z-test Statistics:")
print("z =", round(z_stat, 4))
print("p-value =", round(p_value, 4))
print("alternative hypothesis: true mean is less than", value)
print("Confidence Interval (95%):", ci)
print("Mean of x:", mean_x)
