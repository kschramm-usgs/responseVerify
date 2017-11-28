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

stime=UTCDateTime('2017-319T21:3900.0Z')
etime=UTCDateTime('2017-319T21:5000.0Z')
#etime=stime+10000
doy='319'

print(stime.julday)
print(etime.julday)


# information about the sensors. edit before running
samprate=200.
network = "XX"
station = ["TST1","GSN2"]
channel = ["00","00"]
component = ["EH0","EHZ"]
sensor = ["STS-2HG","T-360"]

labelRef="reference sensor: "+ sensor[0] + ' on ' + station[0]
labelNom="test sensor: "+ sensor[1] + ' on ' + station[1]
respFile=["responses/RESP."+network+"."+station[0]+"."+channel[0]+"."+component[0],
"responses/RESP."+network+"."+station[1]+"."+channel[1]+"."+component[1]]
print(respFile)
fileName=['/msd/'+network+'_'+station[0]+'/2017/'+doy+'/'+channel[0]+'_'+component[0]+'.512.seed',
'/msd/'+network+'_'+station[1]+'/2017/'+doy+'/'+channel[1]+'_'+component[1]+'.512.seed']
print(fileName[0])

#respFile=['RESP.XX.TST1.00.EH0','STS-1_Q330HR_BH_20']
# should be able to build resp file name
#respFile=['responses/RESP.XX.TST1.00.EH0','responses/RESP.XX.TST1.10.EH0']
#respFile=['responses/RESP.XX.TST1.00.BH0','responses/RESP.XX.MOFO3.00.BHZ']
#fileName=['/msd/XX_TST1/2017/323/00_BH0.512.seed','/msd/XX_MOFO3/2017/323/00_BHZ.512.seed']

# read in data streams
st = Stream()

st=read(fileName[0],starttime=stime, endtime=etime)
stRef=st.copy()
#st.plot()
stRefAcc=stRef.copy()
stRefRaw=stRef.copy()

st=read(fileName[1],starttime=stime, endtime=etime)
stNom=st.copy()
stNomAcc=stNom.copy()
stNomRaw=stNom.copy()

# remove responses, creating both displacement and acceleration traces
prefilt=(0.003, 0.004, 100.0, 150.0)
respf = respFile[0]
refResp = {'filename':respf, 'units':'DIS'}
stRef.simulate(paz_remove=None,pre_filt=prefilt,seedresp=refResp,pitsasim=False,sacsim=True)
trRef=stRef[0]

refResp = {'filename':respf, 'units':'ACC'}
stRefAcc.simulate(paz_remove=None,pre_filt=prefilt,seedresp=refResp,pitsasim=False,sacsim=True)
trRefAcc=stRefAcc[0]

#plt.figure()
#plt.plot(trRef.data)
#plt.plot(trRefAcc.data)
print('removed ref response')

respf = respFile[1]
nomResp = {'filename':respf, 'units':'DIS'}
stNom.simulate(paz_remove=None,pre_filt=prefilt,seedresp=nomResp,pitsasim=False,sacsim=True)
trNom=stNom[0]

nomResp = {'filename':respf, 'units':'ACC'}
stNomAcc.simulate(paz_remove=None,pre_filt=prefilt,seedresp=nomResp,pitsasim=False,sacsim=True)
trNomAcc=stNomAcc[0]

print('removed nom response')

# define a few things for the spectral calculations 
trLength=trRefAcc.data.size
print('trace length:i '+str(trLength))
po2=trLength.bit_length()
print('power of 2: '+str(po2))
pad=np.power(2,int(np.ceil(po2)+1))
print('padding length: '+str(pad))
ivl=1/samprate
#
# need the fft to look at the amplitude and phase going into the PSD
# use acceleration
fftRef = np.fft.fft(trRefAcc.data,n=pad)
fftRefFreq=np.fft.fftfreq(pad,d=ivl)
fftRefMag=np.absolute(fftRef)
fftRefPha=np.unwrap(np.degrees(np.arctan2(fftRef.imag,fftRef.real)))

fftNom = np.fft.fft(trNomAcc.data,n=pad)
fftNomFreq=np.fft.fftfreq(pad,d=ivl)
fftNomMag=np.absolute(fftNom)
fftNomPha=np.unwrap(np.degrees(np.arctan2(fftNom.imag,fftNom.real)))

# want to look at the differences in the amplitude and phase
deltaMag = np.log10(fftRefMag)-np.log10(fftNomMag)
deltaPha = fftRefPha-fftNomPha

#need to plot the results from the FFT
plt.figure(figsize=(8.5,5))
plt.suptitle('FFT amplitude')
plt.subplot(211)
plt.loglog(1/fftNomFreq,fftNomMag,'b',label=labelRef )
plt.loglog(1/fftRefFreq,fftRefMag,'r',label=labelNom)
plt.grid(True, which='both')
plt.legend()
plt.xlabel('Period [seconds]')
plt.ylabel('Magnitude \n (acceleration)')
plt.subplot(212)
plt.grid(True, which='both')
#plt.loglog(1/fftNomFreq,deltaMag,'g')
plt.semilogx(1/fftNomFreq,(deltaMag),'g')
plt.xlabel('Period [seconds]')
plt.ylabel('Magnitude difference \n of log values')
#plt.subplot(313)
#plt.grid(True, which='both')
##plt.loglog(1/fftNomFreq,deltaMag,'g')
#plt.semilogx(1/fftNomFreq,np.log10(np.abs(deltaMag)),'g')
#plt.xlabel('Period [seconds]')
#plt.ylabel('Magnitude difference')
string='Magnitude_'+network+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')

#plt.figure(figsize=(11,8.5))
plt.figure(figsize=(8.5,5))
plt.suptitle('FFT phase')
plt.subplot(211)
plt.semilogx(1/fftNomFreq,fftNomPha,'b',label= labelNom)
plt.semilogx(1/fftRefFreq,fftRefPha,'r',label= labelRef)
plt.grid(True, which='both')
plt.xlabel('Period [seconds]')
plt.ylabel('Phase [degrees]')
plt.legend()
plt.subplot(212)
plt.grid(True, which='both')
plt.semilogx(1/fftNomFreq,deltaPha,'g')
plt.xlabel('Period [seconds]')
plt.ylabel('Phase difference')
string='Phase_'+network+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')

# define a few things for the spectral calculations
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
#make sure to subtract decibel values
deltaPSD = 10*np.log10(PSDRef)-10*np.log10(PSDNom)

#plot it up

#plt.figure(figsize=(11,8.5))
plt.figure(figsize=(8.5,5))
plt.title('PSD in displacement')
plt.subplot(211)
#plt.semilogx(1/PSDfreqs,(PSDNom),'b',label=labelNom)
#plt.semilogx(1/PSDfreqs,(PSDRef),'r',label=labelRef)
plt.semilogx(1/PSDfreqs,10*np.log10(PSDNom),'b',label=labelNom)
plt.semilogx(1/PSDfreqs,10*np.log10(PSDRef),'r',label=labelRef)
print(len(PSDNom))
#period, IniAmp = ReadTwoColumnFile('data/PSD_XX_TST1_00_EH0') 
#print(len(IniAmp))
#plt.semilogx(period,IniAmp,'g',label='reference PSD from xmax')
plt.legend()
plt.grid(True, which='both')
plt.xlabel('Period [seconds]')
#plt.ylabel('Power spectral density \n [dB relative to 1 m]')
plt.ylabel('Power spectral density \n [dB relative to 1 m/s^2]')
plt.title('PSD response validation')
plt.subplot(212)
plt.semilogx(1/PSDfreqs,deltaPSD,'g')
#plt.semilogx(1/PSDfreqs,10*np.log10(deltaPSD),'g')
plt.xlabel('Period [seconds]')
#plt.ylabel('Power spectral density \n [dB relative to 1 m]')
plt.ylabel('Power spectral density \n [dB relative to 1 m/s^2]')
plt.title('PSD difference')
plt.grid(True, which='both')
string='PSD_'+network+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')
#plt.ylabel('Power spectral density \n [dB relative to 1 m/s^2]')
#

#plot the response removed waveforms
#plt.figure(figsize=(11,8.5))
plt.figure(figsize=(8.5,5))
plt.suptitle('Data comparison')
print(trNom.data.size/samprate)
t1=(np.linspace(0,(trNom.data.size/samprate),num=trNom.data.size))
#plt.figure(figsize=(11,8.5))
plt.subplot(311)
plt.plot(t1,stRefRaw[0],'r',label=labelRef)
plt.plot(t1,stNomRaw[0],'b',label=labelNom)
plt.legend()
plt.title('Raw data')
plt.ylabel('Counts')
plt.xlabel('Time [s]')
plt.subplot(312)
plt.plot(t1,trNom.data,'b',label=labelNom)
plt.plot(t1,trRef.data,'r',label=labelRef)
plt.ylabel('Displacement')
plt.xlabel('Time [s]')
plt.title('response removed data')
plt.subplot(313)
#plt.plot(t1,trNom.data-trRef.data,'b',label='trace difference')
plt.plot(t1[76000:96000],trNom.data[76000:96000],'b',label=labelNom)
plt.plot(t1[76000:96000],trRef.data[76000:96000],'r',label=labelRef)
#plt.title('difference in response removed data')
plt.title('response removed data,zoomed')
plt.ylabel('Displacement')
#plt.ylabel('Acceleration [m/s^2]')
plt.xlabel('Time [s]')
string='Data_'+network+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')
#plt.title(station[1] + ' r
#
plt.show()
