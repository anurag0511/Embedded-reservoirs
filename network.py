
##import numpy
from numpy import *
import numpy as np
from matplotlib.pyplot import *
import matplotlib.pylab as plt
from pylab import *
import thread
import scipy.linalg
ion()


with open('data_file_2.txt','r') as f:
	data = f.read().split()
data1 =loadtxt('data_file_2.txt')
var = 0
data_x=[]  
data_y=[]
for i in data:
	
	if var%2 == 0:
		data_x.append(data1[var])
	else :
		data_y.append(data1[var])
		
	print "this is the stuff", data1[var]
	var=var+1
 
x = len(data1)
print "the size is ", x
y = len(data_y)
plt.ion()   
plt.plot(data_x[9:x-1])
plt.plot(data_y[10:y-1])
plt.show
plt.savefig('plot1.png',format='png')
plt.close

inputsize = 1
resSize = 100

trainLen = 99
initLen = 10
testLen = 99


W = np.random.rand(resSize,resSize)-0.5
 
Win = (np.random.rand(resSize,1+inputsize)-0.5)*1
print "this is done"
print "this is done2"
rhoW = max(abs(linalg.eig(W)[0]))
print "this is done3", rhoW
k=0;
W *= 1.25/rhoW   
X = zeros((1+inputsize+resSize,trainLen-initLen))
for j in range(len(data1)):
	if j%2== 0:
		Yt = data1[None,initLen+1:trainLen+1]
a = 0.20
print Yt.shape
x= zeros((resSize,1))
for t in range(trainLen):
	u = data1[t]
	x = (1-a)*x + a*tanh( dot (Win , vstack((1,u)) ) + dot( W, x))
	if (t>= initLen): 
		X[:,t-initLen] = vstack((1,u,x))[:,0]  

reg = 1e-8
X_t = X.T
Wout = dot(dot(Yt,X_t), linalg.inv( dot(X,X_t) + reg*eye(1+inputsize+resSize)))
outsize =1 
Y = zeros((outsize,testLen))
u = data1[trainLen]

for t in range (testLen):
	x = (1-a)*x + a*tanh(dot ( Win,vstack((1,u)) ) + dot( W, x))
	y = dot( Wout, vstack((1,u,x)) )
	Y[:,t]= y
	u=y	
		
mse = sum( square( data1[0:35]-Y[0,0:35]))/35
print "the program ended with mse", mse


plt.ion()  
plt.plot(Y.T,"r")	 	
plt.show
plt.savefig('plot2.png',format='png')
plt.close
