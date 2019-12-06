import sabs_pkpd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import myokit

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack


# Define protocol example
p = myokit.Protocol()
for i in range(10):
    p.schedule(-80,1000*i,1000*i + 50)
    p.schedule(-80 + 10 * (i+1), 1000*i +  50, 250)
    p.schedule(-80, 1000*i + 300, 700)
for i in range(10, 20):
    p.schedule(-80,1000*i,1000*i + 50)
    p.schedule(-80 + 10 * (i-9), 1000*i +  50, 250)
    p.schedule(-80, 1000*i + 300, 700)

time_max = 19999

# sample spacing
T = 1
n_time_points = int(time_max/T) +1

times = p.log_for_times(np.linspace(0, time_max, n_time_points))
time_series = times['time']
clamp_series = times['pace']

# Number of sample points
N = 400

x = time_series
y = clamp_series
# We truncate xf to the first half of the frequencies, due to symmetry in the FFT
yf = scipy.fftpack.fft(y)
xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))

yf_reduced = np.zeros(len(yf), dtype=np.complex_)
mean_range = int(n_time_points / N)
for i in range(1, N-1):
    real_sum = np.sum(np.abs(np.real(yf[mean_range*i-int(mean_range/2):mean_range*i+int(mean_range/2)])))
    imag_sum = np.sum(np.abs(np.imag(yf[mean_range*i-int(mean_range/2):mean_range*i+int(mean_range/2)])))
    yf_reduced[i*mean_range] = real_sum + 1.0j * imag_sum
    yf_reduced[(i-1)*mean_range+1:i*mean_range-1] = np.zeros(mean_range-2, dtype=np.complex_)
yf_reduced[0] = yf[0]

yi = scipy.fftpack.ifft(yf)
xi = np.linspace(0, time_max, len(yi))
yif = scipy.fftpack.ifft(yf[0:int(N/2)])
xif = np.linspace(0, time_max, len(yif))

plt.figure(0)
plt.subplot(3,1,1)
plt.plot(x,y)
plt.subplot(3, 1, 2)
plt.plot(xf, np.abs(yf[0:int(N/2)]))
plt.plot(xf, np.abs(yf_reduced[0:int(N/2)]))
plt.subplot(3, 1, 3)
plt.plot(xi, yi)
plt.plot(xif, yif)

