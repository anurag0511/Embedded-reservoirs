
##import numpy
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pylab as plt
from pylab import *
import thread
import scipy.linalg
ion()
import time
import random

with open('data_file_circle.txt','r') as f:
	data = f.read().split()
data1 =loadtxt('data_file.txt')
data1 =data1/(max(abs(data1)))
var = 0
for i in data:	
	print "this is the angle value", data1[var]
	var=var+1
 
x = len(data1)
print "the size is ", x

#y = len(data_y)
plt.ion()   
plt.plot(data1)
#plt.plot(data_y[10:y-1])

plt.show
plt.savefig('plot1.png',format='png')
plt.close
# intitialise the training andd testing size parameters
inputsize = 1
resSize = 100
outsize =1 
trainLen = 99
initLen = 0
testLen = 100
maxsamples = 100
##start the counter
start_time = time.time()
## initialise the weights
random.seed()

W = (np.random.rand(resSize,resSize)*2)-1
Wprob = np.random.rand(resSize,resSize)
W[Wprob<0.7]= 0

Wfb = (np.random.rand(resSize,outsize)-0.5)*2
Wfbprob = np.random.rand(resSize,outsize)
Wfb[Wfbprob<0.7] = 0

Wout = (np.random.rand(outsize,resSize)-0.5)*2

Win = (np.random.rand(resSize,inputsize)-0.5)*2
Winprob =np.random.rand(resSize,inputsize)
Win[Winprob<0.7] = 0

Wdel = zeros((outsize,resSize))

#print "the feedback weight matrix is ", Wfb
  
rhoW = max(abs(linalg.eig(W)[0]))
print "this is done3", rhoW
k=0;
W *= 1.25/rhoW

#print "the output matrix is ", Win

Yexp = np.append(0,data1)
print "the y traget is ", Yexp
a = 0.5
x= zeros((resSize,1))

for j in range(1,maxsamples):
	for k in range(1,len(data1)-1):
		u = data1[k]
		x = (1-a)*x + a*tanh( dot (Win , u ) + dot( W, x) + dot(Wfb, Yexp[k]) )
		ypre = dot(Wout,x)
		Wdel = 0.01*(Yexp[k+2] - ypre)*(x.T)
		Wout = Wout + Wdel

print "the x value is ", x
###print Yt.shape


Y = zeros((outsize,testLen))
#x=data1

print " the popoutput weights are " , Wout
for t in range(1,testLen):
	u = data1[t]
	if t<21:
		x = (1-a)*x + a*tanh( dot (Win , u ) + dot( W, x) + dot(Wfb, Yexp[t]) )
	else:
		x = (1-a)*x + a*tanh( dot (Win , u ) + dot( W, x) + dot(Wfb,y) ) 
	y = dot(Wout,x)
	Y[:,t]= y
	#print "the values are ", x
print "the output matrix ius", Y
f = open('sum_file.txt','a+')

for i in range(testLen):  
	f.write(str(Y[0].item(i)))
	f.write(" ")

f.write("\n")
plt.ion()  
plt.plot(Y.T,"r")	 	
plt.show
plt.savefig('plot2.png',format='png')
#plt.close
