import pandas as pd
import random

data = []

for _ in range(1000):
    l1 = random.randint(0, 50)
    l2 = random.randint(0, 50)
    l3 = random.randint(0, 50)
    l4 = random.randint(0, 50)

    max_lane = max(l1, l2, l3, l4)

    green_time = 10 + (max_lane * 1.5)

    data.append([l1, l2, l3, l4, green_time])

df = pd.DataFrame(data, columns=["lane1","lane2","lane3","lane4","green_time"])
df.to_csv("traffic_data.csv", index=False)

print("Dataset created")