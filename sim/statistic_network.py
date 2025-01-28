import pandas as pd
import matplotlib.pyplot as plt

# Data parsing
data = """ Initial probability = 0.00010891, GC2 ER network Size = 3, Avg Degree = 0.12
Initial probability = 0.00011078, GC2 ER network Size = 3, Avg Degree = 0.09
Initial probability = 0.00011861, GC2 ER network Size = 3, Avg Degree = 0.11
Initial probability = 0.00013141, GC2 ER network Size = 3, Avg Degree = 0.15
Initial probability = 0.00017566, GC2 ER network Size = 4, Avg Degree = 0.16
Initial probability = 0.00016129, GC2 ER network Size = 4, Avg Degree = 0.18
Initial probability = 0.00019796, GC2 ER network Size = 5, Avg Degree = 0.20
Initial probability = 0.00024298, GC2 ER network Size = 5, Avg Degree = 0.24
Initial probability = 0.00031930, GC2 ER network Size = 5, Avg Degree = 0.33
Initial probability = 0.00041250, GC2 ER network Size = 5, Avg Degree = 0.39
Initial probability = 0.00051501, GC2 ER network Size = 7, Avg Degree = 0.46
Initial probability = 0.00055142, GC2 ER network Size = 12, Avg Degree = 0.59
Initial probability = 0.0006214, GC2 ER network Size = 17, Avg Degree = 0.62
Initial probability = 0.00073711, GC2 ER network Size = 26, Avg Degree = 0.71
Initial probability = 0.00078921, GC2 ER network Size = 34, Avg Degree = 0.84
Initial probability = 0.00092029, GC2 ER network Size = 28, Avg Degree = 0.91
Initial probability = 0.00096866, GC2 ER network Size = 44, Avg Degree = 0.99
Initial probability = 0.00105498, GC2 ER network Size = 108, Avg Degree = 1.02
Initial probability = 0.00103712, GC2 ER network Size = 223, Avg Degree = 1.10
Initial probability = 0.00125139, GC2 ER network Size = 342, Avg Degree = 1.19
Initial probability = 0.00136291, GC2 ER network Size = 420, Avg Degree = 1.31
Initial probability = 0.00153593, GC2 ER network Size = 593, Avg Degree = 1.48
Initial probability = 0.00182188, GC2 ER network Size = 751, Avg Degree = 1.81
Initial probability = 0.00185325, GC2 ER network Size = 807, Avg Degree = 2.05
Initial probability = 0.00223614, GC2 ER network Size = 847, Avg Degree = 2.22
Initial probability = 0.00256340, GC2 ER network Size = 911, Avg Degree = 2.67
Initial probability = 0.00279184, GC2 ER network Size = 928, Avg Degree = 2.83
Initial probability = 0.00314626, GC2 ER network Size = 959, Avg Degree = 3.15
Initial probability = 0.00366884, GC2 ER network Size = 970, Avg Degree = 3.66
Initial probability = 0.00406460, GC2 ER network Size = 984, Avg Degree = 4.10
"""


rows = []
for line in data.splitlines():
    print(line)
    parts = line.split(", ")
    prob = float(parts[0].split(" = ")[1])
    gc_size = int(parts[1].split(" = ")[1])
    avg_degree = float(parts[2].split(" = ")[1])
    rows.append((prob, gc_size, avg_degree))

df = pd.DataFrame(rows, columns=["Initial Probability", "GC Size", "Avg Degree"])
df["GC Proportion"] = df["GC Size"] / 1000  # Network size fixed at 1000

# Plotting with Average Degree on the x-axis and GC Proportion on the y-axis
plt.figure(figsize=(10, 6))
plt.plot(df["Avg Degree"], df["GC Proportion"], label="GC Proportion", color="purple", marker="o")
plt.xlabel("Average Degree (K)")
plt.ylabel("GC Proportion (S / 1000)")
plt.title("Relationship Between Average Degree (K) and GC Proportion (S)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# # GC Proportion vs. Initial Probability
# plt.subplot(2, 1, 2)
# plt.plot(df["Initial Probability"], df["GC Proportion"], label="GC Proportion", color="red")
# plt.xscale("log")
# plt.xlabel("Initial Probability (log scale)")
# plt.ylabel("GC Proportion")
# plt.title("Effect of Link Probability on GC Proportion")
# plt.grid(True)
# plt.legend()

# plt.tight_layout()
# plt.show()
