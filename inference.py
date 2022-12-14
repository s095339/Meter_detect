a = [
    [1.,2.,3.,4.],
    [2.,3.,4.,5.],
    [3.,4.,5.,6.]
]
import numpy as np
a = np.array(a)

n = np.mean(a,axis = 0)
print(n)

import matplotlib.pyplot as plt
xaxis = [int(i+1) for i in range(len(n))]
plt.plot(xaxis,n)

plt.show()