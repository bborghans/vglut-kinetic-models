import numpy as np
from scipy import linalg
from scipy.integrate import odeint

elec = 1.602176565e-7 # Elementary charge in pC
R = 8.3144621 # Gas constant J/(mol*1Kelvin):
T = 273.15+22 # Temperature (22C in Kelvin)
F = 96485.33 # Faraday constant

def simulate(t, A, y0):
    """Integrates ordinary differential equations of transition matrix.
    t is the time range in ms
    A is the transitionmatrix()
    y0 is the steady state output of initialvalue()"""
    def func(x,t):
        return np.array(A.dot(x))[0]
    y=odeint(func, y0, t, full_output=True)
    return y

def initialvalue(states, X):
    """Generates an initial model state for subsequent simulation.
    states is any iterable with the number of states to get the correct matrix size
    A is the transitionmatrix"""
    temp=np.matrix([[0.0]*len(states)]*len(states))
    temp=temp+X
    for i in range(0,len(states)):
        temp[len(states)-1, i]=1
    temp=linalg.solve(temp, [0.]*(len(states)-1)+[1.])
    return temp

def gatingcurrent(A, y, start, connections, reference="123456789abcdefghijklmnopqrstuvwxyz"):#*scale,
    """for each z calculates I active transport from timecourse odeint in simulate()
    as sum of charge_k01 * [ (TO,FROM)*FROM (i.e. A[apo_H,apo]=k01*Hex) - (FROM,TO)*TO ].
    A is the transitionmatrix()
    y is the time course output of simulate()
    start is the parameter set, only z values are used
    connections is a list as follows: [['1', 1, 0], ['2', 2, 1], ...];
        including a sublist for each k0x "forward" state-state connection in the transitionmatrix;
        connections are named as 1-characters str in the given reference argument (1-9 then a-z);
        ints indicate target & origin state, ['1', 1, 0]: connection 1 is = state 0 -> state 1"""
    current=0
    charges=start[2::4]
    for n,(z,TO,FROM) in enumerate(connections):
        z_ind=reference.index(z)
        calculation=charges[z_ind]*(A[TO,FROM]*y[:,FROM]-A[FROM,TO]*y[:,TO])
        current+=calculation
    return current

def stationarycurrent(A, y, start, connections, reference="123456789abcdefghijklmnopqrstuvwxyz"):#*scale,
    """for each z rate set calculates I active transport from steady state value in initialvalue()
    as sum of charge_k01 * [ (TO,FROM)*FROM (i.e. A[apo_H,apo]=k01*Hex) - (FROM,TO)*TO ]
    A is the transitionmatrix()
    y is the time course output of simulate()
    start is the parameter set, only z values are used
    connections is a list as follows: [['1', 1, 0], ['2', 2, 1], ...];
        including a sublist for each k0x "forward" state-state connection in the transitionmatrix;
        connections are named as 1-characters str in the given reference argument (1-9 then a-z);
        ints indicate target & origin state, ['1', 1, 0]: connection 1 is = state 0 -> state 1"""
    current=0
    charges=start[2::4]
    for n,(z,TO,FROM) in enumerate(connections):
        z_ind=reference.index(z)
        calculation=charges[z_ind]*(A[TO,FROM]*y[FROM]-A[FROM,TO]*y[TO])
        current+=calculation
    return current

def modelselect(start, model=None, whatsfast=3):
    if type(model)==str:
        IND_SIZE=0
    else:
        IND_SIZE=len(start)
    connections=Cldep=closingstates=[]

    if "0" in model:
        samelen,g_len,a_len=52,108,108#164
        #start0=[1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1, 1, 0, 0.5, 1, 1, 0, 0.5]+[1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 0, 0.5, 1000, 1000, 0, 0.5]
        G={"model":"GlutWT",
        "Cldep":[40, 44, 48, 92, 96],
        "Hdep":[0, 4, 12, 16, 20, 24, 32, 36, 56, 64, 76, 84],
        "Sdep":[52, 68, 72, 88],
        "sig0":[45, 46, 41, 42, 97, 98, 93, 94, 61, 62, 81, 82, 101, 102, 105, 106], "noVdep":[],
        "connections":[['1', 1, 0], ['2', 2, 1], ['3', 3, 2], ['4', 3, 4], ['5', 4, 5], ['6', 7, 6], ['7', 8, 7], ['b', 9, 3], ['8', 9, 8], ['9', 9, 10], ['c', 10, 4], ['a', 10, 11], ['d', 11, 5], ['e', 12, 2], ['f', 12, 13], ['g', 14, 13], ['i', 15, 3], ['q', 15, 12], ['h', 15, 14], ['j', 16, 8], ['k', 16, 17], ['o', 18, 14], ['l', 18, 17], ['n', 19, 9], ['p', 19, 15], ['r', 19, 16], ['m', 19, 18]],
        }
        A={"model":"AspWT",
        "Cldep":[40, 44, 48, 92, 96],
        "Hdep":[0, 4, 12, 16, 20, 24, 32, 36, 56, 64, 76, 84],
        "Sdep":[52, 68, 72, 88],
        "sig0":[45, 46, 41, 42, 97, 98, 93, 94, 61, 62, 81, 82, 101, 102, 105, 106], "noVdep":[],
        "connections":[['1', 1, 0], ['2', 2, 1], ['3', 3, 2], ['4', 3, 4], ['5', 4, 5], ['6', 7, 6], ['7', 8, 7], ['b', 9, 3], ['8', 9, 8], ['9', 9, 10], ['c', 10, 4], ['a', 10, 11], ['d', 11, 5], ['e', 12, 2], ['f', 12, 13], ['g', 14, 13], ['i', 15, 3], ['q', 15, 12], ['h', 15, 14], ['j', 16, 8], ['k', 16, 17], ['o', 18, 14], ['l', 18, 17], ['n', 19, 9], ['p', 19, 15], ['r', 19, 16], ['m', 19, 18]],
        }
        if IND_SIZE==g_len or "GlutWT0" in model:
            model,Cldep,Hdep,Sdep,sig0,noVdep,connections=[G[key] for key in ["model","Cldep","Hdep","Sdep","sig0","noVdep","connections"]]
        elif IND_SIZE==a_len or "AspWT0" in model:
            model,Cldep,Hdep,Sdep,sig0,noVdep,connections=[A[key] for key in ["model","Cldep","Hdep","Sdep","sig0","noVdep","connections"]]
        elif IND_SIZE==g_len+(a_len-samelen) or model=="SYM0":
            model="SYM"; connections=[]
            Cldep,Hdep,Sdep,sig0,noVdep=[G[key]+[x+(g_len-samelen) for x in A[key] if x>=samelen] for key in ["Cldep","Hdep","Sdep","sig0","noVdep"]]
        else: raise Exception(f'No 0-model fitting given start length ({len(start)}) and model ({model}) can be found. Model name overrides start.')

    else: raise Exception(f'No model fitting given start length ({len(start)}) and model ({model}) can be found. Model name overrides start.')
    sig0+=noVdep

    return model, Hdep, Cldep, Sdep, sig0, noVdep, connections, closingstates


def dicts(data, full=0, indent=["    ", "\t"][1]):
    """Returns name, type, and shape or value for the contents of each key in a dictionary.
    Shows all objects in dictionary when full!=0. Indent is tab by default, can be changed."""
    if not full:
        for key, value in data.items():
            TYPE=type(value).__name__
            print(f"{indent}{key}: {TYPE if TYPE!='dict' else ''}", end='')
            if isinstance(value, dict):
                print()
                dicts(value, full=0, indent=indent+"    ")
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], dict):
                    print()
                    dicts(value[0], full=0, indent=indent+"    ")
                else:
                    try:
                        print(f" {np.shape(value)}")
                    except ValueError:
                        print(f" {len(value)}")
            elif isinstance(value, (np.ndarray)):
                print(f" {np.shape(value)}")
            elif isinstance(value, (str,int,float)):
                print(f" ({value})")
            else:
                print()
    else: # full
        for key, value in data.items():
            print(f"{indent}{key}:\t", end='')
            if isinstance(value, dict):
                print()
                dicts(value, full=1, indent=indent+"    ")
            elif isinstance(value, (list, tuple)):
                if len(value) > 0 and isinstance(value[0], dict):
                    print()
                    dicts(value[0], full=1, indent=indent+"    ")
                else:
                    try:
                        print(f" {value}")
                    except ValueError:
                        print(f" {value}")
            else:
                print(f" {value}")
def startcalc(start, model):#,variables    #for i in range(len(start)):exec(variables[i]=start[i])
    """Calculates variables in start, following microscopic reversibility."""
    if model=="GlutWT0":
        k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr=start
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
        start=k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr

    elif model=="AspWT0":
        k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr=start
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
        start=k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr

    ###########################################################################
    elif model=="altGlutWT0":
        k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr=start
        k52=(k51*kc1*ka2*kd2)/(kc2*ka1*kd1)
        z5=-(+zc-za-zd)
        k42=(k41*kb1*k92*kc2)/(kb2*k91*kc1)
        z4=-(+zb-z9-zc)
        ki2=(ki1*kp1*kn2*kb2)/(kp2*kn1*kb1)
        zi=-(+zp-zn-zb)
        ke2=(ke1*kq1*ki2*k32)/(kq2*ki1*k31)
        ze=-(+zq-zi-z3+1)
        kj2=(kj1*kr1*kn2*k82)/(kr2*kn1*k81)
        zj=-(+zr-zn-z8+1)
        kh2=(kh1*ki2*k32*ke1*kf2*kg1)/(ki1*k31*ke2*kf1*kg2)
        zh=-(-zi-z3+ze-zf+zg+2)
        km2=(km1*kp2*kh2*ko1)/(kp1*kh1*ko2)
        zm=-(-zp-zh+zo)
        kk2=(kk1*kj2*k81*kn1*km2*kl2)/(kj1*k82*kn2*km1*kl1)
        zk=-(-zj+z8+zn-zm-zl-2)#defined against the cycle!
        start=k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr

    elif model=="altAspWT0":
        k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr=start
        k52=(k51*kc1*ka2*kd2)/(kc2*ka1*kd1)
        z5=-(+zc-za-zd)
        k42=(k41*kb1*k92*kc2)/(kb2*k91*kc1)
        z4=-(+zb-z9-zc)
        ki2=(ki1*kp1*kn2*kb2)/(kp2*kn1*kb1)
        zi=-(+zp-zn-zb)
        ke2=(ke1*kq1*ki2*k32)/(kq2*ki1*k31)
        ze=-(+zq-zi-z3+1)
        kj2=(kj1*kr1*kn2*k82)/(kr2*kn1*k81)
        zj=-(+zr-zn-z8+1)
        kh2=(kh1*ki2*k32*ke1*kf2*kg1)/(ki1*k31*ke2*kf1*kg2)
        zh=-(-zi-z3+ze-zf+zg+2)
        km2=(km1*kp2*kh2*ko1)/(kp1*kh1*ko2)
        zm=-(-zp-zh+zo)
        kk2=(kk1*kj2*k81*kn1*km2*kl2)/(kj1*k82*kn2*km1*kl1)
        zk=-(-zj+z8+zn-zm-zl-2)#defined against the cycle!
        start=k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr

    else:
        raise Exception(f'Warning, model {model} not in trimmed version of startcalc!')
    return list(start)
def loaddata(protein="GlutWT"):
    if protein=="GlutWT":
        datasets=[
        ["WTintGlut40Cl_pH55",      [5.5,7.4, .04,.14, -.05], None, [[-.16+.02*x for x in range(12)][:4]], [1005,1045,6000,6000], 100000, [5950,6000]],#[2500,2645,7500,10000], 100000, [7400,7500]
        ["WTintGlut40Cl_pH5",       [5,7.4, .04,.14, -.05], None, [[-.16+.02*x for x in range(12)][:4]], [252,400,1300,1300], 20000, [1250,1300]],

        ["WTintGlut40Cl_pH5App",    [7.4,7.4, .04,.14, -.05], [5.,7.4, .04,.14], [[-.16+.02*x for x in range(5)][:4]], [252,550,1300, 1300, 2050, 2050, 2850, 2850], 20000, [2000,2050]],

        ["WTintGlutpH5_40ClApp",    [5.,7.4, 0,.14, -.05], [5.,7.4, .04,.14], [[-.16+.02*x for x in range(5)][:4]], [252,400,620, 620, 1420, 1420, 2220, 2220], 20000, [1370,1420]],
        ["WTintGlutpH55_140ClApp",  [5.5,7.4, 0,.14, -.05], [5.5,7.4, .14,.14], [[-.134+.02*x for x in range(5)][:4]], [252,350,600, 600, 1350, 1350, 1830, 1830][:6], 20000, [1300,1350]],#
        ["WTintGlutpH55_140ClApp2", [5.5,7.4, .14,.14, -.05], [5.5,7.4, 0,.14], [[-.186+.02*x for x in range(5)][:4]], [600, 600, 1350, 1350, 1830, 1830], 20000, [1300,1350]],
        ["WTintGlutpH55_140ClApp_leaksubtract",   [5.5,7.4, 0,.14, -.05], [5.5,7.4, .14,.14], [[-.134+.02*x for x in range(5)][:4]], [252,350,600, 600, 1350, 1350, 1830, 1830][:6], 20000, [1300,1350]],#
        ["WTintGlutpH55_140ClApp2_leaksubtract",  [5.5,7.4, .14,.14, -.05], [5.5,7.4, 0,.14], [[-.186+.02*x for x in range(5)][:4]], [600, 600, 1350, 1350, 1830, 1830], 20000, [1300,1350]],
        ]

        deps=dict(
            pH55Cldep=dict(data=[0.227480292149936, 0.5068402643771823, 0.6848164991404779, 0.8580232073527664, 1.0, 0.9566242807419659, 0.984444992342673, 0.6758232675961837][:-1],
            CIs=[0.0666644626493865, 0.08871264976699234, 0.07780951913042589, 0.036979889425549306, 0, 0.32505606863556213, 0.37641489096142067, 0.22512240900162117][:-1],
            pHs=[5.5], Cls=[0, .005, .01, .02, .04, .08, .1, .14][:-1], Vs=[-.14]), # Ns=[10, 12, 12, 14, 31, 11, 11, 8]

            Cl40pHdep=dict(data=[0.0, 0.022181215844110966, 0.08544916120334525, 0.24354221930936537, 0.39315587631907895, 0.6062964816675012, 0.8395309269498162, 1.0],
            CIs=[0, 0.05086277473895897, 0.0421380112821047, 0.05133456151417837, 0.0724377614512783, 0.09774701833486273, 0.10578015202841252, 0.0],#pHglutcurrent BK.ods
            pHs=[7.5, 7, 6.5, 6, 5.75, 5.5, 5.25, 5], Cls=[.04], Vs=[-.16]),
            )

        return datasets,deps#noClpHdep,ClpHdep,pH5Cldep

    elif protein=="AspWT":
        datasets=[
        ["WTintAsp40Cl_pH5",     [5,7.4, .04,.14, -.05], None, [[-.16+.02*x for x in range(12)][:4]], [990, 1075, 6000, 7000], 100000, [5950,6000]],

        ["WTintAsp40Cl_pH5App",  [7.4,7.4, .04,.14, -.05], [5.,7.4, .04,.14], [[-.16+.02*x for x in range(5)][:4]], [252, 650, 1320, 1320, 2120, 2120, 2870, 2870], 20000, [2070,2120]],

        ["WTintAsppH55_40ClApp", [5.5, 7.4, 0, .14, -.05], [5.5, 7.4, .04, .14], [[-.12, -.10]], [0, 50, 199, 199, 2199, 2199, 4199, 4199], 4000, [2189,2199]],
        ["WTintAsppH55_40ClApp_leaksubtract",[5.5, 7.4, 0, .14, -.05], [5.5, 7.4, .04, .14], [[-.12, -.10]], [0, 50, 199, 199, 2199, 2199, 4199, 4199], 4000, [2189,2199]],
        ]

        deps=dict(
            pH55Cldep=dict(data=[0.21105707216745506, 0.49921929317432184, 0.630084074931286, 0.8174044955996139, 1.0, 0.9912233953625819, 0.9928767405605649, 0.7916678079211145][:-1],
            CIs=[0.03543362803383926, 0.07928579602477238, 0.07813046627745364, 0.10058996590201418, 0.11886634264298267, 0.11647626360787734, 0.1315035617745481, 0.0][:-1],
            pHs=[5.5], Cls=[0, .005, .01, .02, .04, .08, .1, .14][:-1], Vs=[-.14]),

            Cl140pHdep=dict(data=[0, 0.019941249731637575, 0.07913906265594323, 0.2506990359195078, 0.41821630298178336, 0.6045652160724451, 0.928349215998816, 1],
            CIs=[0, 0.01189351256959599, 0.023519855409070645, 0.07036079105317966, 0.058428196203882066, 0.06416352061787711, 0.0527744129295189, 0],
            pHs=[7.5, 7.0, 6.5, 6.0, 5.75, 5.5, 5.25, 5.0], Cls=[.04], Vs=[-.16]),
            )

        return datasets,deps#noClpHdep,ClpHdep,pH5Cldep

    else:
        print("Can only load proteins GlutWT and AspWT")
###############################################################################
GlutWT0_transitionmatrix_args = ['k11','k12','z1','d1','k21','k22','z2','d2','k31','k32','z3','d3','k41','k42','z4','d4','k51','k52','z5','d5','k61','k62','z6','d6','k71','k72','z7','d7','k81','k82','z8','d8','k91','k92','z9','d9','ka1','ka2','za','da','kb1','kb2','zb','db','kc1','kc2','zc','dc','kd1','kd2','zd','dd','ke1','ke2','ze','de','kf1','kf2','zf','df','kg1','kg2','zg','dg','kh1','kh2','zh','dh','ki1','ki2','zi','di','kj1','kj2','zj','dj','kk1','kk2','zk','dk','kl1','kl2','zl','dl','km1','km2','zm','dm','kn1','kn2','zn','dn','ko1','ko2','zo','do','kp1','kp2','zp','dp','kq1','kq2','zq','dq','kr1','kr2','zr','dr', 'phex','phint','Clex','Clint','V'] # external pH, internal pH, external [Cl-], internal [Cl-], membrane V
GlutWT0_states=["iapo", "iH", "iH2", "oH2", "oH", "oapo", "iCl", "iClH", "iClH2", "oClH2", "oClH", "oCl", "iH2S", "iHS", "oHS", "oH2S", "iClH2S", "iClHS", "oClHS", "oClH2S"]
 
def GlutWT0_transitionmatrix(k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr,
    phex,phint,Clex,Clint,V, # external pH, internal pH, external [Cl-], internal [Cl-], membrane V
    states=["iapo", "iH", "iH2", "oH2", "oH", "oapo", "iCl", "iClH", "iClH2", "oClH2", "oClH", "oCl", "iH2S", "iHS", "oHS", "oH2S", "iClH2S", "iClHS", "oClHS", "oClH2S"]):

    Hex=10**-phex # [H+] calculated from external pH
    Hint=10**-phint # [H+] calculated from internal pH
    Sint=.14 # internal [substrate], as a constant
    Sex=0 # external [substrate], as a constant

    # detailed balance calculation
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

    # voltage dependence calculation
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

    # state-to-state flux calculation. connections are named 1-9, then a-z
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
###############################################################################

AspWT0_transitionmatrix_args = ['k11','k12','z1','d1','k21','k22','z2','d2','k31','k32','z3','d3','k41','k42','z4','d4','k51','k52','z5','d5','k61','k62','z6','d6','k71','k72','z7','d7','k81','k82','z8','d8','k91','k92','z9','d9','ka1','ka2','za','da','kb1','kb2','zb','db','kc1','kc2','zc','dc','kd1','kd2','zd','dd','ke1','ke2','ze','de','kf1','kf2','zf','df','kg1','kg2','zg','dg','kh1','kh2','zh','dh','ki1','ki2','zi','di','kj1','kj2','zj','dj','kk1','kk2','zk','dk','kl1','kl2','zl','dl','km1','km2','zm','dm','kn1','kn2','zn','dn','ko1','ko2','zo','do','kp1','kp2','zp','dp','kq1','kq2','zq','dq','kr1','kr2','zr','dr', 'phex','phint','Clex','Clint','V']
AspWT0_states=["iapo", "iH", "iH2", "oH2", "oH", "oapo", "iCl", "iClH", "iClH2", "oClH2", "oClH", "oCl", "iH2S", "iHS", "oHS", "oH2S", "iClH2S", "iClHS", "oClHS", "oClH2S"]
def AspWT0_transitionmatrix(k11,k12,z1,d1,k21,k22,z2,d2,k31,k32,z3,d3,k41,k42,z4,d4,k51,k52,z5,d5,k61,k62,z6,d6,k71,k72,z7,d7,k81,k82,z8,d8,k91,k92,z9,d9,ka1,ka2,za,da,kb1,kb2,zb,db,kc1,kc2,zc,dc,kd1,kd2,zd,dd,ke1,ke2,ze,de,kf1,kf2,zf,df,kg1,kg2,zg,dg,kh1,kh2,zh,dh,ki1,ki2,zi,di,kj1,kj2,zj,dj,kk1,kk2,zk,dk,kl1,kl2,zl,dl,km1,km2,zm,dm,kn1,kn2,zn,dn,ko1,ko2,zo,do,kp1,kp2,zp,dp,kq1,kq2,zq,dq,kr1,kr2,zr,dr,
    phex,phint,Clex,Clint,V,states=["iapo", "iH", "iH2", "oH2", "oH", "oapo", "iCl", "iClH", "iClH2", "oClH2", "oClH", "oCl", "iH2S", "iHS", "oHS", "oH2S", "iClH2S", "iClHS", "oClHS", "oClH2S"]):

    Hex=10**-phex # [H+] calculated from external pH
    Hint=10**-phint # [H+] calculated from internal pH
    Sint=.14 # internal [substrate], as a constant
    Sex=0 # external [substrate], as a constant

    # detailed balance calculation
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

    # voltage dependence calculation
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

    # state-to-state flux calculation. connections are named 1-9, then a-z
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
