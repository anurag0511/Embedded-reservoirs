import smbus
import math
import time
import datetime
import sys
import numpy

pi = 3.57

Q_angle = 0.02
Q_gyro = 0.0015
R_angle = 0.005
y_bias = 0.0
x_bias = 0.0

XP_00 = 0.0
XP_01 = 0.0
XP_10 = 0.0
XP_11 = 0.0
YP_00 = 0.0
YP_01 = 0.0
YP_10 = 0.0
YP_11 = 0.0
kalman_x_list = []
Kalman_xangle = 0.0
Kalman_yangle = 0.0

def kalmanFilterY( accelangle, gyrorate, DT):
	y=0.0
	s=0.0
	global Kalman_yangle
	global Q_angle
	global Q_gyro
	global y_bias
	global YP_00
	global YP_01
	global YP_10
	global YP_11
	
	Kalman_yangle += DT*(gyrorate - y_bias)
	YP_00 += (-DT*(YP_10+YP_01)+Q_angle*DT)
	YP_01 += -DT*YP_11
	YP_10 += -DT*YP_11 
	YP_11 += Q_gyro*DT
	
	y = accelangle - Kalman_yangle 
	s = YP_00 + R_angle
	K_0 = YP_00/s
	K_1 = YP_10/s
	
	Kalman_yangle += (K_0*y)
	y_bias += K_1*y
	YP_00 -= K_0*YP_00
	YP_01 -= K_0*YP_01
	YP_10 -= K_1*YP_00
	YP_11 -= K_1*YP_01
	return Kalman_yangle
	

def kalmanFilterX( accelangle, gyrorate, DT):
	x=0.0
	s=0.0

	global Kalman_xangle
	global Q_angle
	global Q_gyro
	global x_bias
	global XP_00
	global XP_01
	global XP_10
	global XP_11
	
	Kalman_xangle += DT*(gyrorate - x_bias)
	XP_00 += (-DT*(XP_10+XP_01)+Q_angle*DT)
	XP_01 += -DT*XP_11
	XP_10 += -DT*XP_11 
	XP_11 += Q_gyro*DT
	
	x= accelangle - Kalman_xangle
	s = XP_00 + R_angle
	K_0 = XP_00/s
	K_1 = XP_10/s
	
	Kalman_xangle += (K_0*x)
	x_bias += K_1*x
	XP_00 -= K_0*XP_00
	XP_01 -= K_0*XP_01
	XP_10 -= K_1*XP_00
	XP_11 -= K_1*XP_01
	return Kalman_xangle	
	
def read_byte(address):
	return bus.read_byte_data(i2c_adr,address)

def read_word( address,flag):
	if flag==1:
		msb = bus.read_byte_data(i2c_adr,address)
		lsb = bus.read_byte_data(i2c_adr,address+1)
	else:
		msb = bus.read_byte_data(i2c_adr,address+1)
		lsb = bus.read_byte_data(i2c_adr,address)
	
	value = (msb<<8)+lsb
	return value

def read_word_comp(address,flag):
	value=read_word(address,flag)
	if(value>=0x8000):
		return -((65535 - value)+1)
	else:
		return value


bus=smbus.SMBus(1)
i2c_adr = 0x68
M_PI = 3.14159265358979323846
Rad_To_Deg =  57.29578
bus.write_byte_data(i2c_adr,0x6b,0)

gyro_x_angle = 0.0
gyro_y_angle = 0.0
gyro_z_angle = 0.0
kalmanY = 0.0
kalmanX = 0.0
time1 = datetime.datetime.now() 
var=0

#f = open('data_file_rotate.txt','w')

while var<100:
	gyro_x = read_word_comp(0x43,1)
	gyro_y = read_word_comp(0x45,1)
	gyro_z = read_word_comp(0x47,1)
	
	accel_x = read_word_comp(0x3b,1)
	accel_y = read_word_comp(0x3d,1)
	accel_z = read_word_comp(0x3f,1)
	
	mag_x = read_word_comp(0x03,0)
	mag_y = read_word_comp(0x05,0)
	mag_z = read_word_comp(0x07,0)
	
	time2=datetime.datetime.now() - time1
	time1 = datetime.datetime.now()
	TP = time2.microseconds/(1000000*1.0)
	
	gyro_x_rate = gyro_x *(0.07)
	gyro_y_rate = gyro_y*(0.07)
	gyro_z_rate = gyro_z*(0.07)
	
	gyro_x_angle += gyro_x_rate*TP 
	gyro_y_angle += gyro_y_rate*TP
	gyro_z_angle += gyro_z_rate*TP
	
	accel_x_angle = (math.atan2(accel_y,accel_z) + M_PI)*Rad_To_Deg
	accel_y_angle = (math.atan2(accel_z,accel_x) + M_PI)*Rad_To_Deg
		
	kalmanY = kalmanFilterY(accel_y_angle,gyro_y_rate,TP) 
	kalmanX = kalmanFilterX(accel_x_angle,gyro_x_rate,TP)
	kalman_x_list.append(kalmanX)
	size = len(kalman_x_list)
	print "the gyro values are"
	print "x axis  ",gyro_x, "y_axis  ", gyro_y, "z-axis  ", gyro_z 
	print "the accel values are"
	print "x axis  ",accel_x, "y_axis  ", accel_y, "z-axis  ", accel_z 
	print "the mag values are"    
	print "x axis  ",mag_x, "y_axis  ", mag_y, "z-axis  ", mag_z 
	print " "
	print "the angles are"
	print "y angle filtered ", kalmanY,"y_angle  ",accel_y_angle
	print "the angles are"
	print "x angle filtered ", kalmanX,"x_angle  ",accel_x_angle
	print " size is    ", size
	time.sleep(0.03)
	var+=1
	#f.write(str(kalmanX))
	#f.write(" ")
	##f.write(str(kalmanY))
	##f.write(" ")
