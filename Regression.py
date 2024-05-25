# Code to try out regression analyses on the data set that we have
# This can be added to a bigger data set


import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as smtools
from statsmodels.tsa.arima_model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_table("ncl.dat", delim_whitespace=True, header=None, names=["Year", "Claim_Count", "Claim_Amount"])

# Adjust the year by adding 1990
data["Year"] = data["Year"] + 1990

# Divide the third column by 1e+09
data["Claim_Amount"] = data["Claim_Amount"] / 1e+09

# Time Series Analysis
print("\n--- Time Series Analysis ---")
# Convert to time series data
claims_ts = data.set_index("Year")["Claim_Amount"]

# Plot the time series
claims_ts.plot(figsize=(12, 6))
plt.title("Adjusted PCS Catastrophe Claims Time Series", fontsize=16)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Claim Amount (USD billion)", fontsize=14)
plt.show()

# Calculate rolling statistics
roll_mean = claims_ts.rolling(window=5).mean()
roll_std = claims_ts.rolling(window=5).std()

# Plot rolling statistics
plt.figure(figsize=(12, 6))
plt.subplot(211)
claims_ts.plot()
roll_mean.plot(color="red", label="Rolling Mean")
plt.legend(loc="best")
plt.title("Adjusted PCS Catastrophe Claims Time Series with Rolling Mean", fontsize=16)
plt.subplot(212)
claims_ts.plot()
roll_std.plot(color="green", label="Rolling Std")
plt.legend(loc="best")
plt.title("Adjusted PCS Catastrophe Claims Time Series with Rolling Std", fontsize=16)
plt.tight_layout()
plt.show()

# Check for stationarity
print("Augmented Dickey-Fuller Test:")
print(smtools.adfuller(claims_ts))

# ARIMA model
model = ARIMA(claims_ts, order=(1, 1, 1))
model_fit = model.fit()
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Forecast future claims
forecast = model_fit.forecast(steps=5)[0]
print("\nForecast for the next 5 years:")
print(forecast)

# Regression Analysis
print("\n--- Regression Analysis ---")
# Prepare data for regression
X = data[["Year", "Claim_Count"]]
y = data["Claim_Amount"]

# Fit linear regression model
regr = LinearRegression()
regr.fit(X, y)

# Print model coefficients
print("\nRegression Coefficients:")
print("Intercept:", regr.intercept_)
print("Year Coefficient:", regr.coef_[0])
print("Claim Count Coefficient:", regr.coef_[1])

# Clustering Analysis
print("\n--- Clustering Analysis ---")
# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[["Claim_Count", "Claim_Amount"]])

# Perform k-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original data
data["Cluster"] = labels

# Print the cluster centers
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# Plot the clusters
plt.figure(figsize=(8, 6))
plt.scatter(data["Claim_Count"], data["Claim_Amount"], c=data["Cluster"], cmap="viridis")
plt.xlabel("Claim Count", fontsize=14)
plt.ylabel("Claim Amount (USD billion)", fontsize=14)
plt.title("Clusters of Claims", fontsize=16)
plt.show()
