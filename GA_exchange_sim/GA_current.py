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
samelen = 52 # nr of shared parameters
g_len   = 108 # nr of glutamate-bound parameters
a_len   = 108 # nr of aspartate-bound parameters
states      = ["iapo", "iH", "iH2", "oH2", "oH", "oapo", "iCl", "iClH", "iClH2", "oClH2", 
               "oClH", "oCl", "iH2S", "iHS", "oHS", "oH2S", "iClH2S", "iClHS", "oClHS", "oClH2S"]
connections = [['1', 1, 0], ['2', 2, 1], ['3', 3, 2], ['4', 3, 4], ['5', 4, 5], 
               ['6', 7, 6], ['7', 8, 7], ['b', 9, 3], ['8', 9, 8], ['9', 9, 10], 
               ['c', 10, 4], ['a', 10, 11], ['d', 11, 5], ['e', 12, 2], ['f', 12, 13], 
               ['g', 14, 13], ['i', 15, 3], ['q', 15, 12], ['h', 15, 14], ['j', 16, 8], 
               ['k', 16, 17], ['o', 18, 14], ['l', 18, 17], ['n', 19, 9], ['p', 19, 15], 
               ['r', 19, 16], ['m', 19, 18]]

def transitionmatrix(k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,
                     k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,
                     zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,
                     dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,
                     km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,
                     zr,dr,phex,phint,Clex,Clint,V,states):
    """
    Detailed balance
    To ensure detailed balance, it suffices to calculate two constants from all others, for each cycle in the model.
    One rate constants parameter: 
    (product of rate constants in the opposite cycle direction) / (product of the remaining rate constants in the same cycle direction)
    One charge movement parameter:
          sum(all other z parameters in the cycle, directionally) * -1
    Since the simulation of channel function causes no net charge movement in steady state, the calculated charge movement adds to 0 for each cycle.
    As long as all states that are part of a cycles are interconnected through these calculations, detailed balance holds.
    These calculations can be applied to replace values after the constants are passed to the transitionmatrix function.
    """
    Hex=10**-phex
    Hint=10**-phint
    Sint=.14
    Sex=0

    kc2=(kc1*ka2*kd2*k51)/(ka1*kd1*k52)
    zc=-(-za-zd+z5)
    kb2=(kb1*k92*kc2*k41)/(k91*kc1*k42)
    zb=-(-z9-zc+z4)
    kp2=(kp1*kn2*kb2*ki1)/(kn1*kb1*ki2)
    zp=-(-zn-zb+zi)
    ko2=(ko1*km1*kp2*kh2)/(km2*kp1*kh1)
    zo=-(+zm-zp-zh)
    kg2=(kg1*kh1*ki2*k32*ke1*kf2)/(kh2*ki1*k31*ke2*kf1)
    zg=-(+zh-zi-z3+ze-zf+2)
    kl2=(kl1*km1*kn2*k82*kj1*kk2)/(km2*kn1*k81*kj2*kk1)
    zl=-(+zm-zn-z8+zj-zk+2)
    kq2=(kq1*ki2*k32*ke1)/(ki1*k31*ke2)
    zq=-(-zi-z3+ze+1)
    kr2=(kr1*kn2*k82*kj1)/(kn1*k81*kj2)
    zr=-(-zn-z8+zj+1)

    k01=k11*np.exp(z1*d1*F*V/(R*T))
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
    k0l=kl1*np.exp(zl*dl*F*V/(R*T))
    kl0=kl2*np.exp(-zl*(1-dl)*F*V/(R*T))
    k0m=km1*np.exp(zm*dm*F*V/(R*T))
    km0=km2*np.exp(-zm*(1-dm)*F*V/(R*T))
    k0n=kn1*np.exp(zn*dn*F*V/(R*T))
    kn0=kn2*np.exp(-zn*(1-dn)*F*V/(R*T))
    k0o=ko1*np.exp(zo*do*F*V/(R*T))
    ko0=ko2*np.exp(-zo*(1-do)*F*V/(R*T))
    k0p=kp1*np.exp(zp*dp*F*V/(R*T))
    kp0=kp2*np.exp(-zp*(1-dp)*F*V/(R*T))
    k0q=kq1*np.exp(zq*dq*F*V/(R*T))
    kq0=kq2*np.exp(-zq*(1-dq)*F*V/(R*T))
    k0r=kr1*np.exp(zr*dr*F*V/(R*T))
    kr0=kr2*np.exp(-zr*(1-dr)*F*V/(R*T))

    A=np.matrix([[0.0]*len(states)]*len(states))

    A[0,0]=-k01*Hex
    A[0,1]=k10
    A[1,0]=k01*Hex
    A[1,1]=-k10 -k02*Hex
    A[1,2]=k20
    A[2,1]=k02*Hex
    A[2,2]=-k20 -k03 -k0e*Sint
    A[2,3]=k30
    A[2,12]=ke0
    A[3,2]=k03
    A[3,3]=-k30 -k40 -k0b*Clex -k0i*Sex
    A[3,4]=k04*Hex
    A[3,9]=kb0
    A[3,15]=ki0
    A[4,3]=k40
    A[4,4]=-k04*Hex -k50 -k0c*Clex
    A[4,5]=k05*Hex
    A[4,10]=kc0
    A[5,4]=k50
    A[5,5]=-k05*Hex -k0d*Clex
    A[5,11]=kd0
    A[6,6]=-k06*Hex
    A[6,7]=k60
    A[7,6]=k06*Hex
    A[7,7]=-k60 -k07*Hex
    A[7,8]=k70
    A[8,7]=k07*Hex
    A[8,8]=-k70 -k08 -k0j*Sint
    A[8,9]=k80
    A[8,16]=kj0
    A[9,3]=k0b*Clex
    A[9,8]=k08
    A[9,9]=-k80 -k90 -kb0 -k0n*Sex
    A[9,10]=k09*Hex
    A[9,19]=kn0
    A[10,4]=k0c*Clex
    A[10,9]=k90
    A[10,10]=-k09*Hex -ka0 -kc0
    A[10,11]=k0a*Hex
    A[11,5]=k0d*Clex
    A[11,10]=ka0
    A[11,11]=-k0a*Hex -kd0
    A[12,2]=k0e*Sint
    A[12,12]=-ke0 -kf0 -k0q
    A[12,13]=k0f*Hint
    A[12,15]=kq0
    A[13,12]=kf0
    A[13,13]=-k0f*Hint -k0g
    A[13,14]=kg0
    A[14,13]=k0g
    A[14,14]=-kg0 -k0h*Hex -k0o*Clex
    A[14,15]=kh0
    A[14,18]=ko0
    A[15,3]=k0i*Sex
    A[15,12]=k0q
    A[15,14]=k0h*Hex
    A[15,15]=-kh0 -ki0 -k0p*Clex -kq0
    A[15,19]=kp0
    A[16,8]=k0j*Sint
    A[16,16]=-kj0 -kk0 -k0r
    A[16,17]=k0k*Hint
    A[16,19]=kr0
    A[17,16]=kk0
    A[17,17]=-k0k*Hint -k0l
    A[17,18]=kl0
    A[18,14]=k0o*Clex
    A[18,17]=k0l
    A[18,18]=-kl0 -k0m*Hex -ko0
    A[18,19]=km0
    A[19,9]=k0n*Sex
    A[19,15]=k0p*Clex
    A[19,16]=k0r
    A[19,18]=k0m*Hex
    A[19,19]=-km0 -kn0 -kp0 -kr0

    return A
def normalized_anioncurrent(openstates,y):
    opentraces=np.array([y[:,i] for i in openstates])
    p_open=[sum(opentraces[:,j]) for j in range(len(opentraces[0]))]
    return np.array(p_open)
def get_parms():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',type=str,default='./Glut_Asp_parms.pkl',
                        help='Path for the input parameters')
    parser.add_argument('-sub',choices=['G','A'],type=str,default='A',
                        help='Select the substrate to consider (Glutamate or Aspartate)')
    parser.add_argument('-pHext0',type=float,default=5.5,
                        help='External pH of the initial condition. (Default 5.5)') 
    parser.add_argument('-pHext1',type=float,default=5.5,
                        help='External pH of the target condition. (Default 5.5)') 
    parser.add_argument('-pHint0',type=float,default=7.4,
                        help='Internal pH of the initial condition. (Default 7.4)') 
    parser.add_argument('-pHint1',type=float,default=7.4,
                        help='Internal pH of the target condition. (Default 7.4)') 
    parser.add_argument('-Clext0',type=float,default=0.0,
                        help='External [Cl-] in M of the initial condition. (Default 0.0)')
    parser.add_argument('-Clext1',type=float,default=0.04,
                        help='External [Cl-] in M of the target condition. (Default 0.04)')
    parser.add_argument('-Sint0',type=float,default=0.140,
                        help='Internal substrate concentration (in M) of the initial condition. (Default 0.140)')
    parser.add_argument('-Sint1',type=float,default=0.140,
                        help='Internal substrate concentration (in M) of the target condition. (Default 0.140)')
    parser.add_argument('-V0',type=float,default=-0.160,
                        help='Membrane voltage in volts of the initial condition. (Default -0.160)')
    parser.add_argument('-V1',type=float,default=-0.160,
                        help='Membrane voltage in volts of the target condition. (Default -0.160)')
    parser.add_argument('-nsteps',type=int,default=2000,
                        help='Number of time steps. The total time, in s, is nsteps/freq. (Default 2000)')
    parser.add_argument('-freq',default=5000,type=int,
                        help='Frequency of datapoints per second in Hz. (Default 5000)')
    parser.add_argument('-out',default='sim.txt',type=str,
                        help='Output file in txt format. The file contains two columns, the '
                        'first representing the time in seconds and the second the total current'
                        'in terms of unitary charge. (Default sim.txt)')
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
def gatingcurrent(A, y, start, connections, reference="123456789abcdefghijklmnopqrstuvwxyz"):
    """for each z calculates I active transport from timecourse odeint in simulate()
    as sum of charge_k01 * [ (TO,FROM)*FROM (i.e. A[apo_H,apo]=k01*Hex) - (FROM,TO)*TO ]"""
    current=0
    charges=start[2::4]
    for n,(z,TO,FROM) in enumerate(connections):
        z_ind=reference.index(z)
        calculation=charges[z_ind]*(A[TO,FROM]*y[:,FROM]-A[FROM,TO]*y[:,TO])
        current+=calculation
    return current#*elec#*scale
def stationarycurrent(A, y, start, connections, reference="123456789abcdefghijklmnopqrstuvwxyz"):
    """
    For each z rate set calculates I active transport from steady state value from initialvalue()
    as sum of charge_k01 * [ (TO,FROM)*FROM (i.e. A[apo_H,apo]=k01*Hex) - (FROM,TO)*TO ]
    """
    current=0
    charges=start[2::4]
    for n,(z,TO,FROM) in enumerate(connections):
        z_ind=reference.index(z)
        calculation=charges[z_ind]*(A[TO,FROM]*y[FROM]-A[FROM,TO]*y[TO])
        current+=calculation
    return current#*elec#*scale
def GA_current(start,args):
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
                           args.Clext0,args.Sint0,
                           args.V0,states)
    tm1 = transitionmatrix(*start,
                           args.pHext1,args.pHint1,
                           args.Clext1,args.Sint1,
                           args.V1,states)
    step_groundstate = initialvalue(states, tm0) # state distribution at the end of step_0
    sim0             = simulate(t, tm1, step_groundstate) # first simulation of state & flux change over time
    if sim0[-1]["message"]!="Integration successful.":
        raise "Integration unsuccessful."
    sim=sim0[0]
    charge_flux_as_current = gatingcurrent(tm0, sim, start, connections) # simulation converted to charge flux as current
    return np.array([t,charge_flux_as_current]).T

def read_pkl(filename):
    with open(filename,'rb') as f:
        data = pickle.load(f)
    return data
if __name__ == '__main__':
    args = get_parms()
    start = read_pkl(args.f)[args.n]
    Gstart  = start[:g_len] # all parameters relevant to glutamate
    Astart  = start[:samelen] + start[g_len:] # all parameters relevant to aspartate
    if args.sub=="G":
        start = Gstart # using glutamate
    if args.sub=="A":
        start = Astart # using aspartate
    current = GA_current(start,args)
    np.savetxt(args.out,current,header='Time (s)\tCurrent (e)')
