## Visualizing the data set


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the file
data = pd.read_table("ncl.dat", delim_whitespace=True, header=None)

# Use this if you rename the table to a csv
# Found no difference in dat and csv
# In case we get a bigger data set we will be replacing it with this line of code
#data = pd.read_table("ncl.csv", delim_whitespace=True, header=None)


# Adjust the year by adding 1990
data[1] = data[1] + 1990

# Divide the third column by 1e+09
data[2] = data[2] / 1e+09

# Create a line plot
plt.figure()
plt.plot(data[1], data[2], color="blue", linewidth=2)
plt.xlabel("Years", fontsize=14)
plt.ylabel("Adjusted PCS catastrophe claims (USD billion)", fontsize=14)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)̨
plt.show()
