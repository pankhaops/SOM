import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("column_3C.dat", sep=r"\s+", header=None)
print("Dataset Shape:", data.shape)
print(data.head())


X = data.iloc[:, :6].values
y = data.iloc[:, 6].values


X = MinMaxScaler().fit_transform(X)

som = MiniSom(3, 2, 6, sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X)


som.train_random(X, 1000)
print("SOM Training Completed")


print("Quantization Error:", som.quantization_error(X))


plt.pcolor(som.distance_map().T)
plt.colorbar()
plt.title("SOM U-Matrix")
plt.show()


markers = {"DH": "o", "SL": "s", "NO": "^"}
colors = {"DH": "red", "SL": "blue", "NO": "green"}

plt.pcolor(som.distance_map().T)
plt.colorbar()

for i, sample in enumerate(X):
    winner = som.winner(sample)
    plt.plot(winner[0]+0.5, winner[1]+0.5,
             markers[y[i]], markerfacecolor="None",
             markeredgecolor=colors[y[i]], markersize=10)

plt.title("SOM Clustering")
plt.show()


#python -m pip install minisom
#python -m pip install scikit-learn
