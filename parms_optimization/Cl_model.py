import numpy as np
from scipy.integrate import odeint
from os import path
# import scipy; from scipy import *; from pylab import *; import math; from scipy.optimize import curve_fit; from scipy.optimize import minimize; from random import *; import os.path; import time

elec=1.602176565e-7#Elementarladung in pC
R=8.3144621#Gaskonstante J/(mol*1Kelvin):
T=273.15+22#Temperatur (22C in Kelvin)
F=96485.3356#Faradaykonstante

savedir=""

def modelselect(start,model=None):
    if type(model)==str:
        IND_SIZE=0
    else:
        IND_SIZE=len(start)
    flux=None;Cldep=[];closingstates=[]

    if IND_SIZE==80 or model=="Cl":model="Cl";Cldep=[28,36,40,48,56,64];Hdep=[0,8,16,24,32,52,44,60];sig0=[1,2, 9,10, 17,18, 33,34, 45,46, 61,62, 69,70, 77,78]+[3,11,31,39,43,51,55,59,63];noVdep=[];closingstates=[[0,3],[1,2],[4,5],[6,9],[7,10],[8,11]];flux=[["apo cH", (1, 0), "k01*Hex-k10"], ["cH oH", (2, 1), "k02-k20"], ["o oH", (2, 3), "k03*Hex-k30"], ["apo o", (3, 0), "k04-k40"], ["cH cH2", (4, 1), "k05*Hex-k50"], ["cH2 oH2", (5, 4), "k06-k60"], ["oH oH2", (5, 2), "k07*Hex-k70"], ["apo cCl", (6, 0), "k08*Clex-k80"], ["cH cClH", (7, 1), "k0a*Clex-ka0"], ["cCl cClH", (7, 6), "k09*Hex-k90"], ["cH2 cClH2", (8, 4), "k0b*Clex-kb0"], ["cClH cClH2", (8, 7), "k0c*Hex-kc0"], ["o oCl", (9, 3), "k0d*Clex-kd0"], ["cCl oCl", (9, 6), "k0i-ki0"], ["oH oClH", (10, 2), "k0f*Clex-kf0"], ["cClH oClH", (10, 7), "k0j-kj0"], ["oCl oClH", (10, 9), "k0e*Hex-ke0"], ["oH2 oClH2", (11, 5), "k0h*Clex-kh0"], ["cClH2 oClH2", (11, 8), "k0k-kk0"], ["oClH oClH2", (11, 10), "k0g*Hex-kg0"]]#[["apo closedH",(1,0),"k01*Hex-k10",],["closedH openH",(2,1),"k02-k20"],["openH open",(3,2),"k30-k03*Hex"],["open apo",(0,3),"k40-k04"],["closedH closedH2",(1,4),"k50-k05*Hex"],["closedH2 openH2",(4,5),"k60-k06"],["openH2 openH",(5,2),"k07*Hex-k70"],["apo closedCl",(0,6),"k80-k08*Clex"],["closedH closedClH",(1,7),"ka0-k0a*Clex"],["closedH2 closedClH2",(4,8),"kb0-k0b*Clex"],["closedCl closedClH",(6,7),"k90-k09*Hex"],["closedClH closedClH2",(7,8),"kc0-k0c*Hex"],["open openCl",(3,9),"kd0-k0d*Clex"],["openH openClH",(2,10),"kf0-k0f*Clex"],["openH2 openClH2",(5,11),"kh0-k0h*Clex"],["openCl openClH",(9,10),"ke0-k0e*Hex"],["openClH openClH2",(10,11),"kg0-k0g*Hex"],["closedCl openCl",(6,9),"ki0-k0i"],["closedClH openClH",(7,10),"kj0-k0j"],["closedClH2 openClH2",(8,11),"kk0-k0k"]]
    elif IND_SIZE==80 or model=="altCl":model="altCl";Cldep=[28,36,40,48,56,64];Hdep=[0,8,16,24,32,52,44,60];sig0=[5,6, 21,22, 29,30, 37,38, 49,50, 53,54, 57,58, 73,74]+[3,11,31,39,43,51,55,59,63];noVdep=[];closingstates=[[0,3],[1,2],[4,5],[6,9],[7,10],[8,11]];flux=[["apo cH", (1, 0), "k01*Hex-k10"], ["cH oH", (2, 1), "k02-k20"], ["o oH", (2, 3), "k03*Hex-k30"], ["apo o", (3, 0), "k04-k40"], ["cH cH2", (4, 1), "k05*Hex-k50"], ["cH2 oH2", (5, 4), "k06-k60"], ["oH oH2", (5, 2), "k07*Hex-k70"], ["apo cCl", (6, 0), "k08*Clex-k80"], ["cH cClH", (7, 1), "k0a*Clex-ka0"], ["cCl cClH", (7, 6), "k09*Hex-k90"], ["cH2 cClH2", (8, 4), "k0b*Clex-kb0"], ["cClH cClH2", (8, 7), "k0c*Hex-kc0"], ["o oCl", (9, 3), "k0d*Clex-kd0"], ["cCl oCl", (9, 6), "k0i-ki0"], ["oH oClH", (10, 2), "k0f*Clex-kf0"], ["cClH oClH", (10, 7), "k0j-kj0"], ["oCl oClH", (10, 9), "k0e*Hex-ke0"], ["oH2 oClH2", (11, 5), "k0h*Clex-kh0"], ["cClH2 oClH2", (11, 8), "k0k-kk0"], ["oClH oClH2", (11, 10), "k0g*Hex-kg0"]]#[["apo closedH",(1,0),"k01*Hex-k10",],["closedH openH",(2,1),"k02-k20"],["openH open",(3,2),"k30-k03*Hex"],["open apo",(0,3),"k40-k04"],["closedH closedH2",(1,4),"k50-k05*Hex"],["closedH2 openH2",(4,5),"k60-k06"],["openH2 openH",(5,2),"k07*Hex-k70"],["apo closedCl",(0,6),"k80-k08*Clex"],["closedH closedClH",(1,7),"ka0-k0a*Clex"],["closedH2 closedClH2",(4,8),"kb0-k0b*Clex"],["closedCl closedClH",(6,7),"k90-k09*Hex"],["closedClH closedClH2",(7,8),"kc0-k0c*Hex"],["open openCl",(3,9),"kd0-k0d*Clex"],["openH openClH",(2,10),"kf0-k0f*Clex"],["openH2 openClH2",(5,11),"kh0-k0h*Clex"],["openCl openClH",(9,10),"ke0-k0e*Hex"],["openClH openClH2",(10,11),"kg0-k0g*Hex"],["closedCl openCl",(6,9),"ki0-k0i"],["closedClH openClH",(7,10),"kj0-k0j"],["closedClH2 openClH2",(8,11),"kk0-k0k"]]

    else: print("IND_SIZE and model number not in trimmed version of modelselect!")
    # fastsets=[i//4 for i in Cldep+Hdep]
    sig0+=noVdep
    return model,Cldep,Hdep,sig0,noVdep,flux,closingstates

def startcalc(start,model):#,variables    #for i in range(len(start)):exec(variables[i]=start[i])
    if model=="Cl":
        k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk=start
        k32=(k31*kf1*ke2*kd2)/(kf2*ke1*kd1) #
        z3=-(zf-ze-zd)#
        k12=(k11*k21*k32*k42)/(k22*k31*k41)
        z1=-(z2-z3-z4)
        k52=(k51*k61*k72*k22)/(k62*k71*k21)
        z5=-(z6-z7-z2)
        k92=(k91*k81*ka2*k12)/(k82*ka1*k11)
        z9=-(z8-za-z1)
        kc2=(kc1*ka1*kb2*k52)/(ka2*kb1*k51)
        zc=-(za-zb-z5)
        kg2=(kg1*kf1*kh2*k72)/(kf2*kh1*k71)
        zg=-(zf-zh-z7)
        ki2=(ki1*ke1*kj2*k92)/(ke2*kj1*k91)
        zi=-(ze-zj-z9)
        kk2=(kk1*kc1*kg2*kj2)/(kc2*kg1*kj1)
        zk=-(zc-zg-zj)
        start=k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk
    elif model=="altCl":
        k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk=start
        k22=(k11*k21*k32*k42)/(k12*k31*k41)
        z2=-(z1-z3-z4)
        k62=(k51*k61*k72*k22)/(k52*k71*k21)
        z6=-(z5-z7-z2)
        k82=(k91*k81*ka2*k12)/(k92*ka1*k11)
        z8=-(z9-za-z1)
        ka2=(kc1*ka1*kb2*k52)/(kc2*kb1*k51)
        za=-(zc-zb-z5)
        kd2=(ke1*kd1*kf2*k32)/(ke2*kf1*k31)
        zd=-(ze-zf-z3)
        kf2=(kg1*kf1*kh2*k72)/(kg2*kh1*k71)
        zf=-(zg-zh-z7)
        kj2=(kg1*kj1*kk2*kc2)/(kg2*kk1*kc1)
        zj=-(zg-zk-zc)
        ke2=(ki1*ke1*kj2*k92)/(ki2*kj1*k91)
        ze=-(zi-zj-z9)
        start=k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk

    else:
        return f'Warning, model {model} not in trimmed version of startcalc!'
    return list(start)

def simulate(t,A,y0):
    def func(x,t):
        return np.array(A.dot(x))[0]
    y=odeint(func,y0,t,full_output=True)
    return y

def initialvalue(states,X):
    temp=np.matrix([[0.0]*len(states)]*len(states))#7
    temp=temp+X
    for i in range(0,len(states)):#7
        temp[len(states)-1,i]=1#6
    temp=np.linalg.solve(temp,[0.]*(len(states)-1)+[1.])
    return temp

def normalized_anioncurrent(openstates,y):
    opentraces=np.array([y[:,i] for i in openstates])#p_open=sum([y[:,i] for i in openstates]) suffices in jupyter
    p_open=[sum(opentraces[:,j]) for j in range(len(opentraces[0]))]#/sum([np.mean(y[-100:,i]) for i in openstates])
    return(np.array(p_open))


##################################################################################################################################################################################################################################################################################################
def Cltransitionmatrix(k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,phex,phint,Clex,Clint,V,
                       states=["apo","cH","oH","o","cH2","oH2","cCl","cClH","cClH2","oCl","oClH","oClH2"]):
    Hex=10.0**-phex
    # Hint=10.0**-phint

    k32=(k31*kf1*ke2*kd2)/(kf2*ke1*kd1)
    z3=-(zf-ze-zd)
    k12=(k11*k21*k32*k42)/(k22*k31*k41)
    z1=-(z2-z3-z4)
    k52=(k51*k61*k72*k22)/(k62*k71*k21)
    z5=-(z6-z7-z2)
    k92=(k91*k81*ka2*k12)/(k82*ka1*k11)
    z9=-(z8-za-z1)
    kc2=(kc1*ka1*kb2*k52)/(ka2*kb1*k51)
    zc=-(za-zb-z5)
    kg2=(kg1*kf1*kh2*k72)/(kf2*kh1*k71)
    zg=-(zf-zh-z7)
    ki2=(ki1*ke1*kj2*k92)/(ke2*kj1*k91)
    zi=-(ze-zj-z9)
    kk2=(kk1*kc1*kg2*kj2)/(kc2*kg1*kj1)
    zk=-(zc-zg-zj)

    k01=k11*np.exp(z1*d1*F*V/(R*T)) # z charge inversed for more convenient convention
    k10=k12*np.exp(-z1*(1-d1)*F*V/(R*T))
    k02=k21*np.exp(z2*d2*F*V/(R*T))
    k20=k22*np.exp(-z2*(1-d2)*F*V/(R*T))
    k03=k31*np.exp(z3*d3*F*V/(R*T))
    k30=k32*np.exp(-z3*(1-d3)*F*V/(R*T))
    k04=k41*np.exp(z4*d4*F*V/(R*T))
    k40=k42*np.exp(-z4*(1-d4)*F*V/(R*T))
    k05=k51*np.exp(z5*d5*F*V/(R*T))
    k50=k52*np.exp(-z5*(1-d5)*F*V/(R*T))
    k06=k61*np.exp(z6*d6*F*V/(R*T))
    k60=k62*np.exp(-z6*(1-d6)*F*V/(R*T))
    k07=k71*np.exp(z7*d7*F*V/(R*T))
    k70=k72*np.exp(-z7*(1-d7)*F*V/(R*T))
    k08=k81*np.exp(z8*d8*F*V/(R*T))
    k80=k82*np.exp(-z8*(1-d8)*F*V/(R*T))
    k09=k91*np.exp(z9*d9*F*V/(R*T))
    k90=k92*np.exp(-z9*(1-d9)*F*V/(R*T))
    k0a=ka1*np.exp(za*da*F*V/(R*T))
    ka0=ka2*np.exp(-za*(1-da)*F*V/(R*T))
    k0b=kb1*np.exp(zb*db*F*V/(R*T))
    kb0=kb2*np.exp(-zb*(1-db)*F*V/(R*T))
    k0c=kc1*np.exp(zc*dc*F*V/(R*T))
    kc0=kc2*np.exp(-zc*(1-dc)*F*V/(R*T))
    k0d=kd1*np.exp(zd*dd*F*V/(R*T))
    kd0=kd2*np.exp(-zd*(1-dd)*F*V/(R*T))
    k0e=ke1*np.exp(ze*de*F*V/(R*T))
    ke0=ke2*np.exp(-ze*(1-de)*F*V/(R*T))
    k0f=kf1*np.exp(zf*df*F*V/(R*T))
    kf0=kf2*np.exp(-zf*(1-df)*F*V/(R*T))
    k0g=kg1*np.exp(zg*dg*F*V/(R*T))
    kg0=kg2*np.exp(-zg*(1-dg)*F*V/(R*T))
    k0h=kh1*np.exp(zh*dh*F*V/(R*T))
    kh0=kh2*np.exp(-zh*(1-dh)*F*V/(R*T))
    k0i=ki1*np.exp(zi*di*F*V/(R*T))
    ki0=ki2*np.exp(-zi*(1-di)*F*V/(R*T))
    k0j=kj1*np.exp(zj*dj*F*V/(R*T))
    kj0=kj2*np.exp(-zj*(1-dj)*F*V/(R*T))
    k0k=kk1*np.exp(zk*dk*F*V/(R*T))
    kk0=kk2*np.exp(-zk*(1-dk)*F*V/(R*T))

    A=np.matrix([[0.0]*len(states)]*len(states))

    A[0,0]=-k01*Hex-k04-k08*Clex
    A[0,1]=k10
    A[0,3]=k40
    A[0,6]=k80

    A[1,0]=k01*Hex
    A[1,1]=-k10-k02-k05*Hex-k0a*Clex
    A[1,2]=k20
    A[1,4]=k50
    A[1,7]=ka0

    A[2,1]=k02
    A[2,2]=-k20-k30-k07*Hex-k0f*Clex
    A[2,3]=k03*Hex
    A[2,5]=k70
    A[2,10]=kf0

    A[3,0]=k04
    A[3,2]=k30
    A[3,3]=-k40-k03*Hex-k0d*Clex
    A[3,9]=kd0

    A[4,1]=k05*Hex
    A[4,4]=-k50-k06-k0b*Clex
    A[4,5]=k60
    A[4,8]=kb0

    A[5,4]=k06
    A[5,5]=-k60-k70-k0h*Clex
    A[5,2]=k07*Hex
    A[5,11]=kh0

    A[6,0]=k08*Clex
    A[6,6]=-k80-k09*Hex-k0i
    A[6,7]=k90
    A[6,9]=ki0

    A[7,1]=k0a*Clex
    A[7,6]=k09*Hex
    A[7,7]=-ka0-k90-k0c*Hex-k0j
    A[7,8]=kc0
    A[7,10]=kj0

    A[8,4]=k0b*Clex
    A[8,7]=k0c*Hex
    A[8,8]=-kb0-kc0-k0k
    A[8,11]=kk0

    A[9,3]=k0d*Clex
    A[9,6]=k0i
    A[9,9]=-kd0-ki0-k0e*Hex
    A[9,10]=ke0

    A[10,2]=k0f*Clex
    A[10,7]=k0j
    A[10,9]=k0e*Hex
    A[10,10]=-kf0-kj0-ke0-k0g*Hex
    A[10,11]=kg0

    A[11,5]=k0h*Clex
    A[11,8]=k0k
    A[11,10]=k0g*Hex
    A[11,11]=-kh0-kk0-kg0

    return A
##################################################################################################################################################################################################################################################################################################
def altCltransitionmatrix(k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,phex,phint,Clex,Clint,V,
                          states=["apo","closedH","openH","open","closedH2","openH2","closedCl","closedClH","closedClH2","openCl","openClH","openClH2"]):
    Hex=10.0**-phex
    # Hint=10.0**-phint

    k22=(k11*k21*k32*k42)/(k12*k31*k41)
    z2=-(z1-z3-z4)
    k62=(k51*k61*k72*k22)/(k52*k71*k21)
    z6=-(z5-z7-z2)
    k82=(k91*k81*ka2*k12)/(k92*ka1*k11)
    z8=-(z9-za-z1)
    ka2=(kc1*ka1*kb2*k52)/(kc2*kb1*k51)
    za=-(zc-zb-z5)
    kd2=(ke1*kd1*kf2*k32)/(ke2*kf1*k31)
    zd=-(ze-zf-z3)
    kf2=(kg1*kf1*kh2*k72)/(kg2*kh1*k71)
    zf=-(zg-zh-z7)
    kj2=(kg1*kj1*kk2*kc2)/(kg2*kk1*kc1)
    zj=-(zg-zk-zc)
    ke2=(ki1*ke1*kj2*k92)/(ki2*kj1*k91)
    ze=-(zi-zj-z9)

    k01=k11*np.exp(z1*d1*F*V/(R*T)) # z charge inversed for more convenient convention
    k10=k12*np.exp(-z1*(1-d1)*F*V/(R*T))
    k02=k21*np.exp(z2*d2*F*V/(R*T))
    k20=k22*np.exp(-z2*(1-d2)*F*V/(R*T))
    k03=k31*np.exp(z3*d3*F*V/(R*T))
    k30=k32*np.exp(-z3*(1-d3)*F*V/(R*T))
    k04=k41*np.exp(z4*d4*F*V/(R*T))
    k40=k42*np.exp(-z4*(1-d4)*F*V/(R*T))
    k05=k51*np.exp(z5*d5*F*V/(R*T))
    k50=k52*np.exp(-z5*(1-d5)*F*V/(R*T))
    k06=k61*np.exp(z6*d6*F*V/(R*T))
    k60=k62*np.exp(-z6*(1-d6)*F*V/(R*T))
    k07=k71*np.exp(z7*d7*F*V/(R*T))
    k70=k72*np.exp(-z7*(1-d7)*F*V/(R*T))
    k08=k81*np.exp(z8*d8*F*V/(R*T))
    k80=k82*np.exp(-z8*(1-d8)*F*V/(R*T))
    k09=k91*np.exp(z9*d9*F*V/(R*T))
    k90=k92*np.exp(-z9*(1-d9)*F*V/(R*T))
    k0a=ka1*np.exp(za*da*F*V/(R*T))
    ka0=ka2*np.exp(-za*(1-da)*F*V/(R*T))
    k0b=kb1*np.exp(zb*db*F*V/(R*T))
    kb0=kb2*np.exp(-zb*(1-db)*F*V/(R*T))
    k0c=kc1*np.exp(zc*dc*F*V/(R*T))
    kc0=kc2*np.exp(-zc*(1-dc)*F*V/(R*T))
    k0d=kd1*np.exp(zd*dd*F*V/(R*T))
    kd0=kd2*np.exp(-zd*(1-dd)*F*V/(R*T))
    k0e=ke1*np.exp(ze*de*F*V/(R*T))
    ke0=ke2*np.exp(-ze*(1-de)*F*V/(R*T))
    k0f=kf1*np.exp(zf*df*F*V/(R*T))
    kf0=kf2*np.exp(-zf*(1-df)*F*V/(R*T))
    k0g=kg1*np.exp(zg*dg*F*V/(R*T))
    kg0=kg2*np.exp(-zg*(1-dg)*F*V/(R*T))
    k0h=kh1*np.exp(zh*dh*F*V/(R*T))
    kh0=kh2*np.exp(-zh*(1-dh)*F*V/(R*T))
    k0i=ki1*np.exp(zi*di*F*V/(R*T))
    ki0=ki2*np.exp(-zi*(1-di)*F*V/(R*T))
    k0j=kj1*np.exp(zj*dj*F*V/(R*T))
    kj0=kj2*np.exp(-zj*(1-dj)*F*V/(R*T))
    k0k=kk1*np.exp(zk*dk*F*V/(R*T))
    kk0=kk2*np.exp(-zk*(1-dk)*F*V/(R*T))

    A=np.matrix([[0.0]*len(states)]*len(states))

    A[0,0]=-k01*Hex-k04-k08*Clex
    A[0,1]=k10
    A[0,3]=k40
    A[0,6]=k80

    A[1,0]=k01*Hex
    A[1,1]=-k10-k02-k05*Hex-k0a*Clex
    A[1,2]=k20
    A[1,4]=k50
    A[1,7]=ka0

    A[2,1]=k02
    A[2,2]=-k20-k30-k07*Hex-k0f*Clex
    A[2,3]=k03*Hex
    A[2,5]=k70
    A[2,10]=kf0

    A[3,0]=k04
    A[3,2]=k30
    A[3,3]=-k40-k03*Hex-k0d*Clex
    A[3,9]=kd0

    A[4,1]=k05*Hex
    A[4,4]=-k50-k06-k0b*Clex
    A[4,5]=k60
    A[4,8]=kb0

    A[5,4]=k06
    A[5,5]=-k60-k70-k0h*Clex
    A[5,2]=k07*Hex
    A[5,11]=kh0

    A[6,0]=k08*Clex
    A[6,6]=-k80-k09*Hex-k0i
    A[6,7]=k90
    A[6,9]=ki0

    A[7,1]=k0a*Clex
    A[7,6]=k09*Hex
    A[7,7]=-ka0-k90-k0c*Hex-k0j
    A[7,8]=kc0
    A[7,10]=kj0

    A[8,4]=k0b*Clex
    A[8,7]=k0c*Hex
    A[8,8]=-kb0-kc0-k0k
    A[8,11]=kk0

    A[9,3]=k0d*Clex
    A[9,6]=k0i
    A[9,9]=-kd0-ki0-k0e*Hex
    A[9,10]=ke0

    A[10,2]=k0f*Clex
    A[10,7]=k0j
    A[10,9]=k0e*Hex
    A[10,10]=-kf0-kj0-ke0-k0g*Hex
    A[10,11]=kg0

    A[11,5]=k0h*Clex
    A[11,8]=k0k
    A[11,10]=k0g*Hex
    A[11,11]=-kh0-kk0-kg0

    return A
##################################################################################################################################################################################################################################################################################################

def formatstart(start, model, variables=[]):
    start=startcalc(start,model)
    row=""
    for i in range(len(start)):
        if i%4==0 and variables:
            print(variables[i][1], end=" ")
        value=[]
        if start[i]>=1e8:                             value="{:.2e}".format(start[i])
        if start[i]>=1e6 and start[i]<1e8:            value=int(round(start[i]))
        if start[i]<1e6 and abs(start[i])>1:          value=round(start[i],6-len(str(start[i]).split(".")[0]))
        if abs(start[i])<=1 and abs(start[i])>=.0001: value=round(start[i],5)
        if abs(start[i])<.0001:                       value="{:.1e}".format(start[i])
        value=str(value)
        if len(value)<8:
            val=value
            value=" "*(8-len(val))
            value=str(val)+value
        row+=value+" "
        if i in np.arange(0,len(start),4)+3:print (row);row=""
    print(row)


def loaddata(protein="WT"):
    if protein=="WT":
        datasets=[
        ["WTintCl140Cl_pH55leaksubtract",[5.5,7.4,.14,.14,0],None,[[-.16+.01*x for x in range(6)]],[0,0,2980,2980],100000.,[2880,2980]],
        ["WTintCl140Cl_pH55Vdeact",      [5.5,7.4,.14,.14,0],[5.5,7.4,.14,.14],[[-.16],[-.14+.02*x for x in range(5)]],[500,515,3502,3525,5500,5500],100000.,[5400,5500]],
        ["WTintCl_180Cl_pH50",           [5.,7.4,.18,.14,0],None,[[-0.16+.01*i for i in range(21)][:6]],[1000,1012,5800,5800],100000.,[5700,5800]],#4000
        ["WTintCl_180Cl_pH65",           [6.5,7.4,.18,.14,0],None,[[-0.16+.01*i for i in range(21)][:6]],[1000,1060,5800,5800],100000.,[5700,5800]],#6000

        ["WTintClpH5_40ClApp",           [5.,7.4,.0,.14,0],[5.,7.4,.04,.14],[np.array([-.16-.02*i for i in range(5)][:4])],[252,310,1340, 1340, 3140, 3140, 4940, 4940],20000.,[3090,3140]],

        ["WTintClpH5_140ClApp",          [5.,7.4,.0,.14,0],[5.,7.4,.14,.14],[[-.16+.02*x for x in range(4)]],[252,265,2000, 2000, 3200, 3200, 4400, 4400],20000.,[3150,3200]],#1200

        ["WTintCl140Cl_pH55App",         [7.4,7.4,.14,.14,0],[5.5,7.4,.14,.14],[[-0.16,-0.14,-0.12,-0.1,-0.08][:4]],[252,265,2120, 2120, 3570, 3570, 4450, 4450],20000.,[3520,3570]],

        ["WTintCl0Cl_pH5App",            [7.4,7.4,.0,.14,0],[5.,7.4,.0,.14],[(np.array([-0.16,-0.14,-0.12,-0.1,-0.08])-.026)[:4]],[252,310,1640, 1640, 4640, 4640, 7320, 7320],20000.,[4590,4640]],

        ["WTintCl0Cl_pHdep_50",          [5.,7.4,.0,.14,0],None,[np.array([-.16+.01*x for x in range(21)][2:8])-.026],[1000,1000,16000,17000],100000.,[-1100,-1000]],
        ["WTintCl0Cl_pHdep_55",          [5.,7.4,.0,.14,0],None,[np.array([-.16+.01*x for x in range(21)][2:8])-.026],[0,0,14960,14960],100000.,[14860,14960]],
        ["WTintCl_0Cl_pH55Vdeact_short", [5.5,7.4,.0,.14,0],[5.5,7.4,.0,.14],[[-.16-.026],np.array([-.14+.02*x for x in range(21)][:5])-.026],[500,515,3502,3525,5500,5500],100000.,[5400,5500]],
        ]

        deps=dict(
            Cldep=dict(data=[0.230512060999056, 0.4632702244021336, 0.5360004647704542, 0.6438465185022781, 0.6909082511612975, 0.871415608059366, 0.9669177877217507, 1.0],
            CIs=[0.0771174007610825, 0.10498241188784507, 0.10807446386587627, 0.10874705675132923, 0.11023132600033425, 0.07379093571071449, 0.054871813797390034, 0.0],
            pHs=[5], Cls=[0, 5, 10, 20, 40, 80, 100, 140], Vs=[-160]),

            Cl0pHdep=dict(data=[1.4180734881370678, 1.4049153657558218, 1.0, 0.9646921369890465, 0.5808510105565777, 0.1817644034323761, 0.037335846740401336, 0.0][1:],
            CIs=[0.13493324892739356, 0.10953678442514692, 0, 0.11930700965156327, 0.0707801530565223, 0.05921518165868464, 0.040587136707005644, 0][1:],
            pHs=[4.6, 4.85, 5.1, 5.21, 5.6, 6.1, 6.6, 7.1][1:], Cls=[0], Vs=[-160]),

            Cl140pHdep=dict(data=[1.0, 0.7720531798679182, 0.570735285258431, 0.3770345381682631, 0.2275435307288326, 0.06154518911535596, 0.009268963709768359, 0.0],
            CIs=[0, 0.055329428966907734, 0.05116068504425997, 0.046662019614465866, 0.03489580652818877, 0.015531915242480016, 0.0038733579885103977, 0],
            pHs=[5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 7.5], Cls=[140], Vs=[-160]),)

        return datasets,deps#noClpHdep,ClpHdep,pH5Cldep

    elif protein=="H120A":
        datasets=[#experiment,conds0,conds2,Vs,tsteps,freq,normrange
        ["H120AintCl140Cl_pHdep_55",        [5.5,7.4,.14,.14,0],None,[[-.16+.01*x for x in range(8)][:6]],[0,50,14960,14960],100000.,[14860,14960]],
        ["H120AintCl140Cl_pH55Vdeact",      [5.5,7.4,.14,.14,0],[5.5,7.4,.14,.14],[[-.16],[-.14+.02*x for x in range(13)][:5]],[500,550,15500,15550,17500,17500],100000.,[17400,17500]],
        ["H120AintCl140ClpH50",             [5.,7.5,.14,.14,0],None,[[-.16+.01*x for x in range(20)][:6]],[1000,1060,31000,32000],100000.,[-1100,-1000]],#17500
        ["H120AintCl140ClpH60",             [6.,7.5,.14,.14,0],None,[[-.16+.01*x for x in range(20)][:6]],[1000,1060,31000,32000],100000.,[-1100,-1000]],#13000
        ["H120AintClpH5_140ClApp",          [5,7.4,0,.14,0],[5,7.4,.14,.14],[[-.16+i*.02 for i in range(9)][:4]],[252,310,2000, 2000, 3460, 3460, 5060, 5060],20000.,[3410,3460]],

        ["H120AintCl140Cl_pH55App",          [7.5,7.5,.140,.14,0],[5.5,7.5,.140,.14],[[-.16+i*.02 for i in range(9)][:4]],[325,325,2120, 2120, 3520, 3520, 5070, 5070],20000.,[3470,3520]],

        ["H120AintCl0Cl_pH5App",            [7.4,7.4,.0,.14,0],[5.,7.4,.0,.14],[[-.16+.02*x for x in range(8)][:4]],[252,310,2120, 2120, 3780, 3780, 5440, 5440],20000.,[3730,3780]],

        ["H120AintCl_0Cl_pH55leaksubtract",  [5.5,7.4,.0,.14,0],None,[np.array([-.16+.01*x for x in range(8)][2:8])-.026],[0,50,14960,14960],100000.,[-1100,-1000]],
        ["H120AintCl_0Cl_pH55Vdeact",       [5.5,7.5,.0,.14,0],[5.5,7.5,.0,.14],[[-.16-.026],np.array([-.14+.02*x for x in range(13)][:5])-.026],[500,550,15500,15550,17500,17500],100000.,[17400,17500]],
        ]

        deps=dict(
            Cldep=dict(data=[0.12699910441980986, 0.31236034077652347, 0.4896231972410552, 0.6439241098901448, 0.8142125944567957, 1.0126330107414263, 1.0],
            CIs=[0.05038229856688842, 0.06814665786605478, 0.07800695760306087, 0.09395375974482933, 0.09189461152308342, 0.08964472876456586, 0.0],
            pHs=[5.5], Cls=[0, 5, 10, 20, 40, 90, 140], Vs=[-160]),

            Cl0pHdep=dict(data=[3.555275767381606, 3.247859946163989, 1.9114238415117968, 1.4206417883676052, 1., 0.33678503130665854, 0.1565149713347818, 0.0][1:],
            CIs=[0, 0.33057996455707794, 0.3018932146843958, 0.3097424080327438, 0.2699106684373307, 0.11387470241017461, 0.08736877334180382, 0][1:],
            pHs=[4.6, 4.85, 5.1, 5.35, 5.6, 6.1, 6.6, 7.1][1:], Cls=[0], Vs=[-160]),

            Cl140pHdep=dict(data=[1.0, 0.7702877449712613, 0.5392586171211641, 0.33795876594550855, 0.18332444017656366, 0.041262105659181106, 0.009510864566443024, 0.0],
            CIs=[0, 0.06774422467393137, 0.06196499690175028, 0.05146828081194993, 0.03479181075042587, 0.008955503987905691, 0.0037285245920102347, 0],
            pHs=[5.0, 5.25, 5.5, 5.75, 6.0, 6.5, 7.0, 7.5], Cls=[140], Vs=[-160]),)

        return datasets,deps

    else:
        print("Can only load proteins WT and H120A")

