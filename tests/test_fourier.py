import sabs_pkpd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

# Number of samplepoints
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y = 15 + np.sin(5 * 2.0*np.pi*x) + 5*np.sin(8 * 2.0*np.pi*x)
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

plt.subplot(3,1,1)
plt.plot(x,y)
plt.subplot(3, 1, 2)
plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.subplot(3, 1, 3)
plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
