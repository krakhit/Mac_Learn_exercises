import matplotlib.pyplot as plt
import numpy as np
import math
r1 = np.arange(0,1,0.01)
r2 = np.sin(2 * math.pi *r1) 
r3 = np.sin(2 * math.pi *r1*r1)
plt.plot(r2,'r--')
plt.plot(r3,'g-')
plt.xlabel('x value')
plt.ylabel('frequency')
plt.title('Frequency plots')
plt.show()