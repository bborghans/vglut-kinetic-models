#!/usr/bin/env python
import pickle
from scipy.integrate import odeint
from scipy.constants import physical_constants
import numpy as np
import sys
import argparse

F = physical_constants['Faraday constant'][0]
R = physical_constants['molar gas constant'][0]
T = 295.15 # Temperature (in Kelvin)
states=["apo", "cH", "oH", "o", "cH2", "oH2", "cCl", "cClH", "cClH2", "oCl", "oClH", "oClH2"]

def transitionmatrix(k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,
                     k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,
                     zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,
                     kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,phex,phint,Clex,
                     Clint,V,states=["apo","cH","oH","o","cH2","oH2","cCl","cClH","cClH2","oCl",
                                     "oClH","oClH2"]):
    """
    Detailed balance
    To ensure detailed balance, it suffices to calculate two constants from all others, for each 
    cycle in the model.  One rate constants parameter: 
    (product of rate constants in the opposite cycle direction) / (product of the remaining rate constants in the same cycle direction)
    One charge movement parameter: sum(all other z parameters in the cycle, directionally) * -1
    Since the simulation of channel function causes no net charge movement in steady state, 
    the calculated charge movement adds to 0 for each cycle. As long as all states that are part 
    of a cycles are interconnected through these calculations, detailed balance holds.  These 
    calculations can be applied to replace values after the constants are passed to the 
    transitionmatrix function.
    """
    Hex=10.0**-phex
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
def normalized_anioncurrent(openstates,y):
    opentraces=np.array([y[:,i] for i in openstates])
    p_open=[sum(opentraces[:,j]) for j in range(len(opentraces[0]))]
    return np.array(p_open)
def get_parms():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',type=str,default='./Cl_WT_parms.pkl',
                        help='Path for the input parameters. (Default ./Cl_WT_parms.pkl)')
    parser.add_argument('-pHext0',type=float,default=5.5,
                        help='External pH of the initial condition. (Default 5.5)') 
    parser.add_argument('-pHext1',type=float,default=5.5,
                        help='External pH of the target condition. (Default 5.5)') 
    parser.add_argument('-pHint0',type=float,default=7.4,
                        help='Internal pH of the initial condition. (Default 7.4)') 
    parser.add_argument('-pHint1',type=float,default=7.4,
                        help='Internal pH of the target condition. (Default 7.4)') 
    parser.add_argument('-Clext0',type=float,default=0.140,
                        help='External [Cl-] in M of the initial condition. (Default 0.140)')
    parser.add_argument('-Clext1',type=float,default=0.140,
                        help='External [Cl-] in M of the target condition. (Default 0.140)')
    parser.add_argument('-Clint0',type=float,default=0.140,
                        help='Internal [Cl-] in M of the initial condition. (Default 0.140)')
    parser.add_argument('-Clint1',type=float,default=0.140,
                        help='Internal [Cl-] in M of the target condition. (Default 0.140)')
    parser.add_argument('-V0',type=float,default=0.,
                        help='Membrane voltage in volts of the initial condition. (Default 0)')
    parser.add_argument('-V1',type=float,default=-.160,
                        help='Membrane voltage in volts of the target condition. (Default -0.160)')
    parser.add_argument('-nsteps',type=int,default=3000,
                        help='Number of time steps. The total time, in s, is nsteps/freq. (Default 3000)')
    parser.add_argument('-freq',type=int,default=100000,
                        help='Frequency of datapoints per second in Hz. (Default 100000)')
    parser.add_argument('-out',default='sim.txt',type=str,
                        help='Output file in txt format. The file contains two columns, the '
                        'first representing the time in seconds and the second the open '
                        'probability. (Default sim.txt)')
    parser.add_argument('-n',default=0,type=int,
                        help='Specify which set of parameters to select in the input file. (Default 0)')
    args = parser.parse_args()
    return args
def initialvalue(states,X):
    temp=np.matrix([[0.0]*len(states)]*len(states))
    temp=temp+X
    for i in range(0,len(states)):
        temp[len(states)-1,i]=1
    temp=np.linalg.solve(temp,[0.]*(len(states)-1)+[1.])
    return temp
def simulate(t,A,y0):
    def func(x,t):
        return np.array(A.dot(x))[0]
    y=odeint(func,y0,t,full_output=True)
    return y
def Cl_open_p(start,args):
    """
    Simulates Cl current based on ext(ernal) and int(ernal) pH, [Cl], and V.
    Plots current over time from 0 to <timestep> ms at frequency <freq> and prints steady state 
    p_open
    """
    openstates = [states.index(i) for i in [i for i in states if i[0]=="o"]]
    t0     = np.arange(args.nsteps)/args.freq # total timespan
    t      = t0[0: args.nsteps]
    tm0 = transitionmatrix(*start,
                           args.pHext0,args.pHint0,
                           args.Clext0,args.Clint0,
                           args.V0)
    tm1 = transitionmatrix(*start,
                           args.pHext1,args.pHint1,
                           args.Clext1,args.Clint1,
                           args.V1)
    step_groundstate = initialvalue(states, tm0) # state distribution at the end of step_0
    sim0             = simulate(t, tm1, step_groundstate) # first simulation of state & flux change over time
    if sim0[-1]["message"]!="Integration successful.":
        raise "Integration unsuccessful."
    sim=sim0[0]
    p_open_as_current = normalized_anioncurrent(openstates, sim) # simulation converted to p_open as current
    #eventual_current = sum([initialvalue(states, step0)[x] for x in openstates])
    return np.array([t,p_open_as_current]).T
def read_pkl(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == '__main__':
    args = get_parms()
    start = read_pkl(args.f)[args.n]
    open_p = Cl_open_p(start,args)
    np.savetxt(args.out,open_p,header='Time (s)\tOpen probability')
