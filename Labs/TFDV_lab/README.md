Network Traffic Anomaly Detection - Feature Engineering

Feature engineering pipeline for predicting bandwidth anomalies 30 minutes in advance using multivariate time series data.

Goal:

Engineer features from network metrics to predict bandwidth anomalies 30 minutes ahead using LSTM models.
Dataset Metrics

Synthetic network data with 11 key metrics collected at 5-minute intervals:

Bandwidth In/Out (Mbps)
Packet Loss (%)
CPU/Memory Usage (%)
Active Connections (thousands)
DNS Queries & HTTP Requests (req/sec)
Failed Connections (per 5 min)
Latency (ms)
Temperature (°C)

Engineered Features

  Domain Features:

  Traffic_Ratio, Network_Stress, Request_Efficiency, Connection_Health

  Temporal Features:

  Cyclic encoding (Hour, Day, Month)
  Binary indicators (Business Hours, Weekend)

  Statistical Features:

  Moving averages (30min, 1hr, 2hr)
  Rate of change (CPU, Memory, Bandwidth)
