# ASK AND FSK
# Aim
Write a simple Python program for the modulation and demodulation of ASK and FSK.
# Tools required
# Program
```
1.ASK
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
def lp_filter(x, cutoff, fs):
    b, a = butter(5, cutoff/(0.5*fs), btype='low')
    return lfilter(b, a, x)
fs, fc, br, T = 1000, 50, 10, 1
t = np.linspace(0, T, fs*T, endpoint=False)
bits = np.random.randint(0, 2, br)
samples_per_bit = fs // br
msg = np.repeat(bits, samples_per_bit)
carrier = np.sin(2*np.pi*fc*t)
ask = msg * carrier
demod = lp_filter(ask * carrier, fc, fs)
decoded = (demod[::samples_per_bit] > 0.25).astype(int)
plt.figure(figsize=(10,7))
plt.subplot(411); plt.plot(t, msg); plt.title("Message"); plt.grid()
plt.subplot(412); plt.plot(t, carrier); plt.title("Carrier"); plt.grid()
plt.subplot(413); plt.plot(t, ask); plt.title("ASK Signal"); plt.grid()
plt.subplot(414); plt.step(range(len(decoded)), decoded); plt.title("Decoded Bits")
plt.tight_layout(); plt.show()

2.FSK
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

fs,f1,f2,br,T = 1000,30,70,10,1
t = np.linspace(0,T,fs*T,endpoint=False)
bits = np.random.randint(0,2,br)
spb = fs//br
msg = np.repeat(bits,spb)

# FSK
fsk = np.zeros_like(t)
for i,b in enumerate(bits):
    fsk[i*spb:(i+1)*spb] = np.sin(2*np.pi*(f2 if b else f1)*t[i*spb:(i+1)*spb])
b,a = butter(5,f1/(0.5*fs),'low')
c1 = lfilter(b,a,fsk*np.sin(2*np.pi*f1*t))
b,a = butter(5,f2/(0.5*fs),'low')
c2 = lfilter(b,a,fsk*np.sin(2*np.pi*f2*t))
dec = [(np.sum(c2[i*spb:(i+1)*spb]**2) >
        np.sum(c1[i*spb:(i+1)*spb]**2))*1 for i in range(br)]
demod = np.repeat(dec,spb)
plt.figure(figsize=(9,9))
plt.subplot(511); plt.plot(t,msg); plt.title("Message"); plt.grid()
plt.subplot(512); plt.plot(t,np.sin(2*np.pi*f1*t)); plt.title("Carrier f1"); plt.grid()
plt.subplot(513); plt.plot(t,np.sin(2*np.pi*f2*t)); plt.title("Carrier f2"); plt.grid()
plt.subplot(514); plt.plot(t,fsk); plt.title("FSK"); plt.grid()
plt.subplot(515); plt.plot(t,demod); plt.title("Demodulated"); plt.grid()
plt.tight_layout(); plt.show()
```

# Output Waveform
ASK
<img width="1190" height="790" alt="508106456-bde1ea11-d732-461e-90bb-b9cb4f6bb9d8" src="https://github.com/user-attachments/assets/3076f6da-805a-4d40-adc5-c35e3d864d67" />

FSK
<img width="1201" height="1012" alt="508108306-8adfb7d2-54f7-4ce4-840b-8a693c6dae0b" src="https://github.com/user-attachments/assets/209bc4b8-719d-4954-87c2-acb51322fb0f" />

# Results
THUS, THE ASK (Amplitude Shift Keying) AND FSK (Frequency Shift Keying) ARE PERFORMED USING PYTHON

