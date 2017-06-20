
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
import tensorflow
with open('data_file_circle.txt','r') as f:
	data = f.read().split()
data1 =loadtxt('data_file.txt')
data1 =data1/(max(abs(data1)))
var = 0
#data_x=[]  
#data_y=[]
for i in data:
#	
#	if var%2 == 0:
#		data_x.append(data1[var])
#	else : 
#		data_y.append(data1[var])
#		
	print "this is the angle value", data1[var]
	var=var+1
 
x = len(data1)
print "the size is ", x
#y = len(data_y)
plt.ion()   
plt.plot(data1[9:x])
#plt.plot(data_y[10:y-1])
plt.show
plt.savefig('plot1.png',format='png')
plt.close

inputsize = 1
resSize = 100
outsize =1 
trainLen = 99
initLen = 0
testLen = 99

start_time = time.time()
random.seed()
W = (np.random.rand(resSize,resSize)*2)-1
Wfb = (np.random.rand(resSize,outsize)-0.5)*2
Wout = (np.random.rand(outsize,resSize)-0.5)*2
Wdel = zeros((outsize,resSize))
#print "the feedback weight matrix is ", Wfb
Win = (np.random.rand(resSize,1+inputsize)-0.5)*2
print "this is done"
print "this is done2"
rhoW = max(abs(linalg.eig(W)[0]))
print "this is done3", rhoW
k=0;
W *= 1.25/rhoW   
Wfb=Wfb/rhoW
Wout=Wout/rhoW
X = zeros((1+inputsize+resSize,trainLen-initLen))
for j in range(len(data1)):
#	if j%2== 0:
	Yt = data1[None,initLen+1:trainLen+1]
a = 0.18
print Yt.shape
x= zeros((resSize,1))
y=zeros((outsize,outsize))
Y = zeros((outsize,testLen))
#x=data1
for t in range(len(data1)):
	x[t][0]= data1[t]
temp =0
print "the x matrix is is ", x
for t in range(1,trainLen):
	u = data1[t]
	x = (1-a)*x + a*tanh( dot (Win , vstack((1,u)) ) + dot( W, x) + dot(Wfb, Yt [0][t-1]) )
	##Wout[0][t]=Yt[0][t]/(x[t][0])
	y = dot(Wout,x)
	temp = y[0]
	E   = 0.5*(square(Yt[0][t]-y))
	Wdel = -(Yt[0][t] - y)*(x.T)
	Wout = Wout - (0.1)*Wdel
	print "the E value is ", E
	
	#temp = dot((1-square(tanh( dot (Win , vstack((1,u)) ) + dot( W, x) + dot(Wfb, Yt [0][t-1]) ))),Yt[0][t-1])
	#Wfb = dot(dot(-(Yt[0][t]-y),Wout),temp)
	Y[:,t]= y
##	if (t>= initLen): 
##		X[:,t-initLen] = vstack((1,u,x))[:,0]  


print "the  wdelta are ", Wdel  
reg = .5e-4
X_t = X.T

Wout = dot(dot(Yt,X_t), linalg.inv( dot(X,X_t) + reg*eye(1+inputsize+resSize)))
print "the Output weights are ",Wout[0]

     
u = data1[trainLen]
'''
z=0
for t in range (testLen):
	
	x = (1-a)*x + a*tanh(dot ( Win,vstack((1,u)) ) + dot( W, x))
	#print "the dot product is ", Wout
	y = dot( Wout, vstack((1,u,x)) )
	z=u-y
	Y[:,t]= y
	u=y	

	'''	
mse = sum( square( data1[61:96]-Y[0,0:35]))/35
print "the program ended with mse", mse, "and time in seconds", (time.time()-start_time)

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
