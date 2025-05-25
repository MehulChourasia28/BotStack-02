import pandas as pd
import numpy as np
df = pd.read_csv(r"battleship_dataset.csv")
data = df.values.astype('float32')

boards = np.array(data[:, :100]).reshape(50000, 10, 10)    # First 100 columns = board (hits)
ships = np.array(data[:, 100:]).reshape(50000, 10, 10)     # Last 100 columns = ships

print(boards[0])
print(ships.shape[0])
