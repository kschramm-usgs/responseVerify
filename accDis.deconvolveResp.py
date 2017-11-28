#!/bin/env python
from obspy import UTCDateTime
from obspy.core import *
from obspy import read
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import glob

def ReadTwoColumnFile(file_name):
# from stackoverflow
   with open(file_name, 'r') as mydat:
      x=[]
      y=[]
      for line in mydat:
         p = line.split()
         x.append(float(p[0]))
         y.append(float(p[1]))
   return x,y

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

stime=UTCDateTime('2017-323T22:5600.0Z')
#etime=UTCDateTime('2017-321T20:1500.0Z')
etime=stime+10000

print(stime.julday)
print(etime.julday)


# information about the sensors.
network = "XX"
station = ["TST1","MOFO3"]
#station = ["TST1"]
channel = ["00","10"]
#component = ["EH0","EH0"]
component = ["BH0","BHZ"]

dataloc = '/msd/'

fileName=[]
#respFile=['RESP.XX.TST1.00.EH0','STS-1_Q330HR_BH_20']
# should be able to build resp file name
#respFile=['responses/RESP.XX.TST1.00.EH0','responses/RESP.XX.TST1.10.EH0']
respFile=['responses/RESP.XX.TST1.10.BH0','responses/RESP.XX.MOFO3.00.BHZ']
fileName=['/msd/XX_TST1/2017/323/10_BH0.512.seed','/msd/XX_MOFO3/2017/323/00_BHZ.512.seed']
#for year in range(stime.year, etime.year+1):
#   for day in range(stime.julday, etime.julday+1):
#      for i in range(len(station)):
#         string = dataloc + network + '_' + station[i] + '/' + str(year) + '/*' \
#                 + str(day).zfill(3)+\
#                '/*' + channel[i] + '*' + component[i] + '*.seed'
#         print(glob.glob(string))
#         fileName += glob.glob(string)
#         print(fileName)

# read in data streams
st = Stream()

st=read(fileName[0],starttime=stime, endtime=etime)
stRef=st.copy()
stRefAcc=stRef.copy()
stRefRaw=stRef.copy()

st=read(fileName[1],starttime=stime, endtime=etime)
stNom=st.copy()
stNomAcc=stNom.copy()
stNomRaw=stNom.copy()

# remove responses, creating both displacement and acceleration traces
prefilt=(0.003, 0.004, 100.0, 150.0)
respf = respFile[0]
unit='DIS'
refResp = {'filename':respf, 'units':'DIS'}
stRef.simulate(paz_remove=None,pre_filt=prefilt,seedresp=refResp,pitsasim=False,sacsim=True)
trRef=stRef[0]

unit='ACC'
refResp = {'filename':respf, 'units':'ACC'}
stRefAcc.simulate(paz_remove=None,pre_filt=prefilt,seedresp=refResp,pitsasim=False,sacsim=True)
trRefAcc=stRefAcc[0]
print('removed ref response')

respf = respFile[1]
unit='DIS'
nomResp = {'filename':respf, 'units':'DIS'}
stNom.simulate(paz_remove=None,pre_filt=prefilt,seedresp=nomResp,pitsasim=False,sacsim=True)
trNom=stNom[0]

unit='ACC'
refResp = {'filename':respf, 'units':'ACC'}
stNomAcc.simulate(paz_remove=None,pre_filt=prefilt,seedresp=nomResp,pitsasim=False,sacsim=True)
trNomAcc=stNomAcc[0]
print('removed nom response')

# define a few things for the spectral calculations
samprate=40.
trLength=trRef.data.size
print('trace length:i '+str(trLength))
po2=trLength.bit_length()
print('power of 2: '+str(po2))
pad=np.power(2,int(np.ceil(po2)+1))
print('padding length: '+str(pad))
ivl=1/samprate
#
# need the fft to look at the amplitude and phase going into the PSD
fftRef = np.fft.fft(trRefAcc.data,n=pad)
fftRefFreq=np.fft.fftfreq(pad,d=ivl)
fftRefMag=np.absolute(fftRef)
fftRefPha=np.unwrap(np.degrees(np.arctan2(fftRef.imag,fftRef.real)))

fftNom = np.fft.fft(trNomAcc.data,n=pad)
fftNomFreq=np.fft.fftfreq(pad,d=ivl)
fftNomMag=np.absolute(fftNom)
fftNomPha=np.unwrap(np.degrees(np.arctan2(fftNom.imag,fftNom.real)))

# want to look at the differences in the amplitude and phase
deltaMag = fftRefMag-fftNomMag
deltaPha = fftRefPha-fftNomPha

#need to plot the results from the FFT
plt.figure(figsize=(11,8.5))
plt.suptitle('FFT amplitude')
plt.subplot(211)
plt.loglog(1/fftNomFreq,fftNomMag,'b',label= 'test sensor')
plt.loglog(1/fftRefFreq,fftRefMag,'r',label= 'reference sensor')
plt.grid(True, which='both')
plt.legend()
plt.xlabel('Period [seconds]')
plt.ylabel('Magnitude (acceleration)')
plt.subplot(212)
plt.grid(True, which='both')
plt.semilogx(1/fftNomFreq,deltaMag)
plt.xlabel('Period [seconds]')
plt.ylabel('Magnitude difference')

plt.figure(figsize=(11,8.5))
plt.suptitle('FFT phase')
plt.subplot(211)
plt.semilogx(1/fftNomFreq,fftNomPha,'b',label= 'test sensor')
plt.semilogx(1/fftRefFreq,fftRefPha,'r',label= 'reference sensor')
plt.grid(True, which='both')
plt.xlabel('Period [seconds]')
plt.ylabel('Phase [degrees]')
plt.subplot(212)
plt.grid(True, which='both')
plt.semilogx(1/fftNomFreq,deltaPha)
plt.xlabel('Period [seconds]')
plt.ylabel('Phase difference')

# define a few things for the spectral calculations
samprate=40.
nsegments=4.
trLength=trRef.data.size
print(trLength)
po2=np.log(trLength/nsegments)
print('power of 2: '+str(po2))
print(trLength/nsegments)
pad=np.power(2,int(np.ceil(po2)))
print(pad)
nfft=int(pad)
print(nfft)
overlap=int(3*nfft//4)
print(overlap)
ivl=1/samprate

# calculate the PSD

PSDfreqs, PSDRef = signal.welch(trRefAcc.data, return_onesided=True, fs=samprate, nperseg=nfft, noverlap=overlap, scaling='density')
PSDfreqs, PSDNom = signal.welch(trNomAcc.data, return_onesided=True, fs=samprate, nperseg=nfft, noverlap=overlap, scaling='density')
deltaPSD = PSDRef-PSDNom

#plot it up
plt.figure(figsize=(11,8.5))
plt.title('PSD in displacement')
plt.subplot(211)
plt.semilogx(1/PSDfreqs,10*np.log10(PSDNom),'b',label='test sensor')
plt.semilogx(1/PSDfreqs,10*np.log10(PSDRef),'r',label='reference sensor')
print(len(PSDNom))
period, IniAmp = ReadTwoColumnFile('data/PSD_XX_TST1_00_EH0') 
print(len(IniAmp))
plt.semilogx(period,IniAmp,'g',label='reference PSD from xmax')
plt.legend()
plt.grid(True, which='both')
plt.xlabel('Period [seconds]')
plt.ylabel('Power spectral density [dB relative to 1 m/s^2')
plt.title('PSD response validation')
plt.subplot(212)
plt.semilogx(1/PSDfreqs,deltaPSD,'r')
plt.xlabel('Period [seconds]')
plt.ylabel('Power spectral density [dB relative to 1 m/s^2')
#

#plot the response removed waveforms
plt.figure(figsize=(11,8.5))
plt.suptitle('Data comparison')
t1=(np.linspace(0,trNom.data.size,num=trNom.data.size))/samprate
#plt.figure(figsize=(11,8.5))
plt.subplot(311)
plt.plot(t1,stRefRaw[0],'r',label='reference sensor')
plt.plot(t1,stNomRaw[0],'b',label='test sensor')
plt.title('Raw data')
plt.ylabel('Counts')
plt.xlabel('Time [s]')
plt.subplot(312)
plt.plot(t1,trNom.data,'b',label='test sensor')
plt.plot(t1,trRef.data,'r',label='reference sensor')
plt.ylabel('Displacement')
plt.xlabel('Time [s]')
plt.title('response removed data')
plt.subplot(313)
plt.plot(t1,trNomAcc.data,'b',label='test sensor')
plt.plot(t1,trRefAcc.data,'r',label='reference sensor')
plt.xlim(50,50.5)
plt.title('response removed data')
plt.ylabel('Acceleration [m/s^2]')
plt.xlabel('Time [s]')
plt.legend()
#plt.title(station[1] + ' r
#
plt.show()
