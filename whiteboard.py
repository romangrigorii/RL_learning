import matplotlib.pyplot as plt 
import numpy as np
x = []
y = []
for i in range(100):
    x.append(i)
    y.append(np.sin(i))
    plt.plot(x,y)
    plt.pause(.01)
