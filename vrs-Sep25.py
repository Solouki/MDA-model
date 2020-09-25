# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 16:06:50 2020

@author: Amin
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:24:05 2020

@author: Amin
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

#************************* ODE solver funcction
def ode(z,t):
    
    DEN1=1
    DEN2=1
    for i in range(0,4):
        DEN1= DEN1 + Kad1[i]*z[i]
        
    for i in range(0,7):
        DEN2= DEN2 + Kad2[i]*z[i]
        
    if t>0.2: #H2 removal membrane 
        z[0]=0

    r1 = (float( kj[0] * ( (z[1])**2 - float(z[2]*z[0]**2)/Kj[0] ) )/DEN1  ) * alpha1
    r2=(float( kj[1] * ( z[2]**4 - float(z[4]*z[3]*z[0]**2)/Kj[1] ) )/DEN2  ) * alpha2
    r3=(float( kj[2] * ( z[2]**3 - float(z[4]*z[0]**3)/Kj[2] ) )/DEN2  ) * alpha2
    r4=(float( kj[3] * ( z[4]*z[1] - float(z[5]*z[0])/Kj[3] ) )/DEN2  ) * alpha2
    r5=(float( kj[4] * ( z[4]*z[2]**2 - float(z[6]*z[0]**3)/Kj[4] ) )/DEN2  ) * alpha2
    r6=(float( kj[5] * ( z[2]*z[0] - float(z[3])/Kj[5] ) )/DEN1  ) * alpha1
    
    
    dFAdw=float((2*r1+2*r2+3*r3+r4+3*r5-r6))*W#*(R*T))/V
    dFBdw=float((-2*r1-r4))*W#*(R*T))/V
    dFCdw=float((r1-4*r2-3*r3-2*r5-r6))*W#*R*T)/V
    dFDdw=float((r2+r6))*W#*R*T)/V
    dFEdw=float((r2+r3-r4-r5))*W#*R*T)/V
    dFFdw=float(r4)*W#*R*T)/V
    dFGdw=float(r5)*W#*R*T)/V
    V = float(t*0.001)/(rho_cat * (1 - eps)) #m3
    L = float(V) /At  #m
    

    return [dFAdw,dFBdw,dFCdw,dFDdw,dFEdw,dFFdw,dFGdw,L]

global kj,Kj,alpha1,alpha2,R,T,V,W,Kad1,Kad2,DEN1,nu0,rho_cat,eps,At


#**************** Operating parameters and constants
T= 973.15 #K
Tref=973.15 #K
R = 8.314 #J/mol.K
g=9.81 #m/s2
Pbar=1 #bar
P=Pbar*101325 #Pa
eps = 0.3

#**************Geometry & Physical properties
Dia = 15 #m
At = np.pi*0.25 * Dia**2 #m2
rho_cat = 2500 #Kg/m3
d_cat=4e-4   # Cat particle size [m]
rho_fd= 0.198 #kg/cum
mu_fd=2.53 #Pa.s


#************** Fluidization parameters
epsmf=0.5
# Ut= 1.75 * np.sqrt(float(g*d_cat*(rho_cat-rho_fd))/rho_cat) 
#Ut= np.sqrt(float(4*g*d_cat*(rho_cat-rho_fd))/(3*rho_fd)) 


Umf= np.sqrt(g*d_cat*(rho_cat-rho_fd)*np.power(epsmf,3) *0.67/    (1.75*rho_fd)         )
# Umf=float(Ut*np.power(eps,0.5))/2.32
#Umf= 0.0075 * float(g*d_cat**2*(rho_cat-rho_fd))/mu_fd

#****************
uo= 0.6 #m/s assumption
nu0=At*uo #m3/s
FAo=float(P*nu0)/(R*T) #mol/s 
mdot_fd=FAo* 16.04 #g/s
whsv=1000 #1/s
W= float(mdot_fd)/whsv #g





#*********************** EQul. & Kinetic coeff. estimation
alpha1=1
alpha2=1
NoC=6 
Aj =[16.36	,-16.18,-10.69,	5.494,1.297,7.658]
Bj =[-26030,	17330	,23530,	6201,	-5990,	1451]
Kj=[]
for i in range(0,NoC):
    Kj.append(np.exp(Aj[i]+float(Bj[i])/T))

kj=[]
A0=[0.01831,1.231e3,1.134e7,1.697,7.063e5,2.917e3]
Ea=[92710,34440,31350,501400,39250,71300]

for i in range(0,NoC):
    kj.append(    ( A0[i]* np.exp(  ((float(-Ea[i]) )/R) * ( (1.0/T) - 1.0/Tref)      ) ))


 

Kad1=[]
A01=[0.07581,0.09644,0.07944,0.08095]
E01=[21410,94610,40020,9725]

for i in range(0,4):
    Kad1.append(A01[i]*np.exp(    (float(-E01[i])/R) *  ((1.0/T )- 1.0/Tref)      ) )

Kad2=[]
A02=[0.09396,0.1189,0.2269,0.05123,0.04881,0.05219]
E02=[21080,44070,128700,23220,17400,34850]

for i in range(0,6):
    Kad2.append(A02[i]*np.exp(    (float(-E02[i])/R) *  ((1.0/T) - 1.0/Tref)      ) )
    
Kad2.append(0.1175) 


#***************************Solving ODEs of the kinetic model & result generation
P0=[0,Pbar,0,0,0,0,0,0]

t=np.arange(0,1,0.001)
conc=odeint(ode,P0,t)

CA=(float(nu0)/(R*T))*(conc[:,0])*101325
CB=(float(nu0)/(R*T))*(conc[:,1])*101325
CC=(float(nu0)/(R*T))*(conc[:,2])*101325
CD=(float(nu0)/(R*T))*(conc[:,3])*101325
CE=(float(nu0)/(R*T))*(conc[:,4])*101325
CF=(float(nu0)/(R*T))*(conc[:,5])*101325
CG=(float(nu0)/(R*T))*(conc[:,6])*101325
L=conc[:,7]
#**************************** Yields & conversion rates
x=100*(FAo-CB )/FAo
yc=100*(CC*2  )/FAo
yd=100*(CD*2  )/FAo
ye=100*(CE* 6 )/FAo
yf=100*(CF*7  )/FAo
yg=100*(CG*10 )/FAo
ci=FAo
Co=CB+2*CC+2*CD+6*CE+7*CF+10*CG
hin=4*FAo
hout=2*CA+4*CB+4*CC+6*CD+6*CE+8*CF+8*CG
errh=hout-hin
err=Co-ci

plt.figure(0)
plt.xlabel('Catalyst Consumption (g)')
plt.ylabel('F (mol/s)')
plt.plot(t,x,label='CH4')

plt.legend()
plt.title('T='+str(T)+ ' K,  P= '+str(Pbar)+' bar, W=' + str(round(W,2))+ ' g,  whsv= '+str(round(whsv,2))+ ' 1/s')









