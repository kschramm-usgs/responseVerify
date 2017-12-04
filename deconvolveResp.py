#!/bin/env python
from obspy import UTCDateTime
from obspy.core import *
from obspy import read
from scipy import signal
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import glob

#adding this to read ouput from xmax
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

stime=UTCDateTime('2017-335T17:1200.0Z')
etime=UTCDateTime('2017-335T17:1300.0Z')
doy='335'

print(stime.julday)
print(etime.julday)

# information about the sensors. edit before running
samprate=20.
network = ["XX","IU"] 
station = ["TST1","ANMO"]
channel = ["00","00"]
component = ["BH0","BHZ"]
sensor = ["STS-2HG","KS54000"]
# set x-axis limits, in period, for the magnitude and phase plots
xlimits=[0.01,10]

print(network[0])
labelRef="reference sensor: "+ sensor[0] + ' on ' + station[0]
labelNom="test sensor: "+ sensor[1] + ' on ' + station[1]
respFile=["responses/RESP."+network[0]+"."+station[0]+"."+channel[0]+"."+component[0],
"responses/RESP."+network[1]+"."+station[1]+"."+channel[1]+"."+component[1]]
print(respFile)
fileName=['/msd/'+network[0]+'_'+station[0]+'/2017/'+doy+'/'+channel[0]+'_'+component[0]+'.512.seed',
'/msd/'+network[1]+'_'+station[1]+'/2017/'+doy+'/'+channel[1]+'_'+component[1]+'.512.seed']
print(fileName[0])
print(fileName[1])

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
refResp = {'filename':respf, 'units':'DIS'}
stRef.simulate(paz_remove=None,pre_filt=prefilt,seedresp=refResp,pitsasim=False,sacsim=True)
trRef=stRef[0]

refResp = {'filename':respf, 'units':'ACC'}
stRefAcc.simulate(paz_remove=None,pre_filt=prefilt,seedresp=refResp,pitsasim=False,sacsim=True)
trRefAcc=stRefAcc[0]

print('removed ref response')

respf = respFile[1]
nomResp = {'filename':respf, 'units':'DIS'}
stNom.simulate(paz_remove=None,pre_filt=prefilt,seedresp=nomResp,pitsasim=False,sacsim=True)
trNom=stNom[0]
#if (trNom.stats['sampling_rate'] > samprate):
#   deciFactor=trNom.stats['sampling_rate']/samprate
#trNom.decimate(factor=2) 

nomResp = {'filename':respf, 'units':'ACC'}
stNomAcc.simulate(paz_remove=None,pre_filt=prefilt,seedresp=nomResp,pitsasim=False,sacsim=True)
trNomAcc=stNomAcc[0]

print('removed nom response')

# define a few things for the fft calculation
trLength=trRefAcc.data.size
po2=trLength.bit_length()
pad=np.power(2,int(np.ceil(po2)+1))
ivl=1/samprate
print('trace length: '+str(trLength))
print('FFT power of 2: '+str(po2))
print('FFT padding length: '+str(pad))

# need the fft to look at the amplitude and phase going into the PSD

fftRef=np.fft.fft(trRefAcc.data,n=pad)
fftRefFreq=np.fft.fftfreq(pad,d=ivl)
fftRefMag=np.absolute(fftRef)
fftRefPha=np.unwrap(np.degrees(np.arctan2(fftRef.imag,fftRef.real)))

fftNom=np.fft.fft(trNomAcc.data,n=pad)
fftNomFreq=np.fft.fftfreq(pad,d=ivl)
fftNomMag=np.absolute(fftNom)
fftNomPha=np.unwrap(np.degrees(np.arctan2(fftNom.imag,fftNom.real)))

# want to look at the differences in the amplitude and phase
deltaMag=np.log10(fftRefMag)-np.log10(fftNomMag)
deltaPha=fftRefPha-fftNomPha


# define a few things for the spectral calculations
#nsegments=4.
#trLength=trRefAcc.data.size
#po2=(trLength.bit_length()/nsegments)
#pad=np.power(2,int(np.ceil(po2)))
#nfft=int(pad)/nsegments
#overlap=int(3*nfft//4)
#ivl=1/samprate
nsegments=4.
trLength=trRefAcc.data.size
po2=np.log(trLength/nsegments)
pad=np.power(2,int(np.ceil(po2)))
nfft=int(pad)
overlap=int(3*nfft//4)
ivl=1/samprate

print('PSD length: '+str(trLength))
print('power of 2: '+str(po2))
print('length/nsegments: '+str(trLength/nsegments))
print('Padding length: '+str(pad))
print('Nfft: '+str(nfft))
print('overlap: '+str(overlap))

# calculate the PSD

PSDfreqs, PSDRef = signal.welch(trRefAcc.data, return_onesided=True, fs=samprate, nperseg=nfft, noverlap=overlap, scaling='density')
PSDfreqs, PSDNom = signal.welch(trNomAcc.data, return_onesided=True, fs=samprate, nperseg=nfft, noverlap=overlap, scaling='density')

#make sure to subtract decibel values
deltaPSD = 10*np.log10(PSDRef)-10*np.log10(PSDNom)

#plot it up
plt.figure(figsize=(8.5,5))
plt.title('PSD in displacement')
plt.subplot(211)
plt.semilogx(1/PSDfreqs,20*np.log10(PSDNom),'b',label=labelNom)
plt.semilogx(1/PSDfreqs,20*np.log10(PSDRef),'r',label=labelRef)
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
string='PSD_'+network[0]+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')
#plt.ylabel('Power spectral density \n [dB relative to 1 m/s^2]')
#

#need to plot the results from the FFT
#want to make sure that the x-axis limits match the PSD as the FFT is taken on
#the entire window
#should likely do something fancy, but for now will just xlim.

plt.figure(figsize=(8.5,5))
plt.suptitle('FFT amplitude')
plt.subplot(211)
plt.semilogx(1/fftNomFreq,10*np.log10(fftNomMag),'b',label=labelRef )
plt.semilogx(1/fftRefFreq,10*np.log10(fftRefMag),'r',label=labelNom)
plt.grid(True, which='both')
plt.legend()
plt.xlabel('Period [seconds]')
plt.ylabel('Magnitude \n (acceleration)')
plt.xlim(xlimits)
plt.subplot(212)
plt.grid(True, which='both')
plt.semilogx(1/fftNomFreq,(deltaMag),'g',label='difference')
plt.legend()
plt.xlabel('Period [seconds]')
plt.ylabel('Magnitude difference \n of log values')
plt.xlim(xlimits)
string='Magnitude_'+network[0]+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')

plt.figure(figsize=(8.5,5))
plt.suptitle('FFT phase')
plt.subplot(211)
plt.semilogx(1/fftNomFreq,fftNomPha,'b',label= labelNom)
plt.semilogx(1/fftRefFreq,fftRefPha,'r',label= labelRef)
plt.grid(True, which='both')
plt.xlabel('Period [seconds]')
plt.ylabel('Phase [degrees]')
plt.legend()
plt.xlim(xlimits)
plt.subplot(212)
plt.grid(True, which='both')
plt.semilogx(1/fftNomFreq,deltaPha,'g',label='difference')
plt.legend()
plt.xlabel('Period [seconds]')
plt.ylabel('Phase difference')
plt.xlim(xlimits)
string='Phase_'+network[0]+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.savefig('pngs/'+string+'.png',format='png')
plt.savefig('pdfs/'+string+'.pdf',format='pdf')

#filter data for plotting
#trNom.data=-1.*trNom.data # for the GS-13 which has reversed polarity
trNomfiltp1=trNom.copy()
trNomfiltp1.detrend('linear') #literally picking because SAC
trNomfiltp1.taper(0.1)
trNomfiltp1.filter("highpass",freq=0.1,zerophase='True')

trNomfilt1=trNom.copy()
trNomfilt1.detrend('linear') #literally picking because SAC
trNomfilt1.taper(0.1)
trNomfilt1.filter("highpass",freq=1.)

trNomfilt10=trNom.copy()
trNomfilt10.detrend('linear') #literally picking because SAC
trNomfilt10.taper(0.1)
trNomfilt10.filter("highpass",freq=10.)

#trNomfilt100=trNom.copy()
#trNomfilt100.detrend('linear') #literally picking because SAC
#trNomfilt100.taper(0.1)
#trNomfilt100.filter("highpass",freq=50.)

trReffiltp1=trRef.copy()
trReffiltp1.detrend('linear') #literally picking because SAC
trReffiltp1.taper(0.1)
trReffiltp1.filter("highpass",freq=0.1,zerophase='True')

trReffilt1=trRef.copy()
trReffilt1.detrend('linear') #literally picking because SAC
trReffilt1.taper(0.1)
trReffilt1.filter("highpass",freq=1.)

trReffilt10=trRef.copy()
trReffilt10.detrend('linear') #literally picking because SAC
trReffilt10.taper(0.1)
trReffilt10.filter("highpass",freq=10.)

#trReffilt100=trRef.copy()
#rReffilt100.detrend('linear') #literally picking because SAC
#rReffilt100.taper(0.1)
#rReffilt100.filter("highpass",freq=50.)

#get the x-axis in seconds
print(trNom.stats['sampling_rate'])
print(trRef.stats['sampling_rate'])
print(trNom.data.size)
print(trRef.data.size)
print(samprate)
t1=(np.linspace(0,(trNom.data.size/samprate),num=trNom.data.size))
# would like to actually get times one of these days... maybe Austin knows?
#t1=t1+stime

#plot the response removed waveforms
fig =plt.figure(figsize=(11,8.5))
#fig.suptitle('Data comparison')
fig.subplots_adjust(wspace=0.15,hspace=0.45)

# first plot up the raw data
print(trNomfilt1.stats)
print(trReffilt1.stats)
rdata=fig.add_subplot(621)
rdata.plot(t1,stRefRaw[0],'r',label=labelRef)
rdata.plot(t1,stNomRaw[0],'b',label=labelNom)
rdata.set_ylabel('Raw Data, \nCounts')
rdata.set_title('Full Time Series')

s1=1200
s2=3200
rdataZ=fig.add_subplot(622)
rdataZ.plot(t1[s1:s2],stRefRaw[0][s1:s2],'r',label=labelRef)
rdataZ.plot(t1[s1:s2],stNomRaw[0][s1:s2],'b',label=labelNom)
rdataZ.set_title('Zoomed Time Series')
rdataZ.tick_params(labelleft='off')

## now plot up resp removed data
noResp=fig.add_subplot(623)
noResp.plot(t1,trNom.data,'b',label=labelNom)
noResp.plot(t1,trRef.data,'r',label=labelRef)
noResp.set_ylabel('Resp removed, \nDisplacement')
noResp.yaxis.set_major_formatter(FormatStrFormatter('%g'))

noRespZ=fig.add_subplot(624)
noRespZ.yaxis.set_label_position("right")
noRespZ.plot(t1[s1:s2],trNom.data[s1:s2],'b',label=labelNom)
noRespZ.plot(t1[s1:s2],trRef.data[s1:s2],'r',label=labelRef)
noRespZ.yaxis.set_major_formatter(FormatStrFormatter('%g'))

## now the .1 hz filtered data
filtp1=fig.add_subplot(625)
filtp1.plot(t1,trNomfiltp1.data,'b',label=labelNom)
filtp1.plot(t1,trReffiltp1.data,'r',label=labelRef)
filtp1.set_ylabel('Filtered,0.1 Hz, \nDisplacement')

s1=1000
s2=2000
filtp1Z=fig.add_subplot(626)
filtp1Z.plot(t1[s1:s2],trNomfiltp1.data[s1:s2],'b',label=labelNom)
filtp1Z.plot(t1[s1:s2],trReffiltp1.data[s1:s2],'r',label=labelRef)
#filtp1Z.tick_params(labelleft='off')

s1=2200
s2=2400
## now the 1 hz filtered data
filt1=fig.add_subplot(627)
filt1.plot(t1,trNomfilt1.data,'b',label=labelNom)
filt1.plot(t1,trReffilt1.data,'r',label=labelRef)
filt1.set_ylabel('Filtered,1 Hz, \nDisplacement')

filt1Z=fig.add_subplot(628)
filt1Z.plot(t1[s1:s2],trNomfilt1.data[s1:s2],'b',label=labelNom)
filt1Z.plot(t1[s1:s2],trReffilt1.data[s1:s2],'r',label=labelRef)

s1=2200
s2=2400
### now the 10 hz filtered data
filt10=fig.add_subplot(629)
filt10.plot(t1,trNomfilt10.data,'b',label=labelNom)
filt10.plot(t1,trReffilt10.data,'r',label=labelRef)
filt10.set_ylabel('Filtered,10 Hz, \nDisplacement')
#
filt10Z=fig.add_subplot(6,2,10)
filt10Z.plot(t1[s1:s2],trNomfilt10.data[s1:s2],'b',label=labelNom)
filt10Z.plot(t1[s1:s2],trReffilt10.data[s1:s2],'r',label=labelRef)

# now the 100 hz filtered data
filt100=fig.add_subplot(6,2,11)
filt100.plot(t1,trNomfilt100.data,'b',label=labelNom)
filt100.plot(t1,trReffilt100.data,'r',label=labelRef)
filt100.set_ylabel('Filtered,50 Hz, \nDisplacement')
#
filt100Z=fig.add_subplot(6,2,12)
filt100Z.plot(t1[s1:s2],trNomfilt100.data[s1:s2],'b',label=labelNom)
filt100Z.plot(t1[s1:s2],trReffilt100.data[s1:s2],'r',label=labelRef)
filt100.set_xlabel('Time [s]')
filt100Z.set_xlabel('Time [s]')

handles,labels = filt100.get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=2)

# save the figure
string='Data_'+network[0]+'_'+station[0]+'_'+channel[0]+'_'+station[1]+'_'+channel[1]+'_'+sensor[1]
plt.subplot_tool()
fig.savefig('pngs/'+string+'.png',format='png')
fig.savefig('pdfs/'+string+'.pdf',format='pdf')
#
plt.show()
