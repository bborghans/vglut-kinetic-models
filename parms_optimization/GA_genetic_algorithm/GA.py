modelfile='GA_model'
import importlib
import argparse
from GA_weights import weights
from GA_model import modelselect, loaddata, startcalc, initialvalue, simulate, gatingcurrent, stationarycurrent,GlutWT0_transitionmatrix_args,AspWT0_transitionmatrix_args,GlutWT0_states,AspWT0_states,GlutWT0_transitionmatrix,AspWT0_transitionmatrix
import sys
from copy import deepcopy
from deap import base, creator, tools
from random import random
import inspect
import numpy as np
from math import inf
import pickle as pkl
import multiprocessing
import warnings
from scipy.linalg import LinAlgWarning
from datetime import datetime
np.set_printoptions(legacy='1.25')

start0=[1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1, 1, 0, 0.5, 1, 1, 0, 0.5]+[1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 0, 0.5, 1000, 1000, 0, 0.5]
def selectmodel(start, MODELS): # selects internal model format
    model=[int(MODELS[x]["ID"]) for x in MODELS.keys() if len(start)==MODELS[x]["g_len"]+MODELS[x]["a_len"]-MODELS[x]["samelen"]]
    if model==[]:
        raise FileNotFoundError
    else:
        model=model[0]

    ID, samelen, g_len, a_len, slowones, clapp, start0=[MODELS[model][x] for x in ["ID","samelen","g_len","a_len","slowones","clapp","start0"]]
    return (model, ID, samelen, g_len, a_len, slowones, clapp, start0)
def evaluate_individual(ind):
    return toolbox.evaluate(
        ind, experimental_data, mode, autoH, chargelim, slowlim, fastlim,
        slowones, modelfile, ID, samelen, g_len, a_len
    )
def evaluate(START, experimental_data,mode, autoH, chargelim, slowlim, fastlim,
        slowones, modelfile, ID, samelen, g_len, a_len, alt=0, reference=[inf]*19):
    RETURN=[]
    OUTTEXT={"GlutWT":{}, "AspWT":{}}
    for SUB,(protein,model) in enumerate([["GlutWT"]*2, ["AspWT"]*2][:(1 if no_Asp else None)]):
        if not [g_len, a_len][SUB]:
            continue
        err=err2=0
        datasets, deps, start, model, Hdep, Cldep, Sdep, sig0, noVdep, flux, closingstates, states,\
            variables, limsmin, limsmax, errs, worsenfactor, transitionmatrix=loadin(START, protein, model, SUB, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, alt)
        datasets=[x for x in datasets if "_leaksubtract" not in x[0]]##########################################################

        peakweight,clweight,phweight,transportweight,pkaweight=[weights[protein][x] for x in ["peakweight","clweight","phweight","transportweight","pkaweight"]]
        def peakweigh(sweep, err, multiplier=1, low=-.5, hi=1.25):
            """Increases RSS error with peaks in + and - direction"""
            test=np.concatenate(sweep)
            MIN, MAX=test.min(), test.max()
            if MIN<low: err+=peakweight*multiplier*MIN**2
            if MAX>hi: err+=peakweight*multiplier*MAX**2
            return np.mean([MIN**2, MAX**2]), err

        startcheck=np.asarray(start)
        if min(np.concatenate([startcheck-limsmin,limsmax-startcheck]))<0:
            if mode == 0: return inf,
            else: print("START VALUE OUT OF BOUNDS:",list(np.where(np.concatenate([startcheck-limsmin,limsmax-startcheck])<0)[0]%len(start)))

        for n,dataset in enumerate(datasets):
            PEAKS=[]
            ERRS=[],[],[]
            experiment,conds0,conds1,Vs,tsteps,freq,normrange=dataset
            e0,e1,e2=[0]*3
            W0,W1,W2=[weights[experiment][x] for x in [0,1,2]]
            keyword = experiment+(clapp if "ClApp" in experiment else "")
            data=experimental_data[keyword]

            length=tsteps[-1]
            t=np.arange(length)/freq # total timespan
            errprint=err

            step1=sim1=sim2=[]
            for V in range(len(Vs[-1])): # =sweeps
                Imax=np.mean(data[V][normrange[0]:normrange[1]])

                sweep=np.asarray([0]*int((len(tsteps)-2)/2),dtype=object)
                step_0=transitionmatrix(*start, *conds0) # pre-V starting point

                step_groundstate=initialvalue(states,step_0)
                step0=transitionmatrix(*start, *conds0[:4], Vs[0][V]) # first condition
                sim0=simulate(t[0:tsteps[2]-tsteps[0]],step0,step_groundstate) # first simulation
                if sim0[-1]["message"]!="Integration successful.": return inf,
                sim0=sim0[0]
                sweep[0]=gatingcurrent(step0, sim0, start, flux)#normalized_anioncurrent(openstates,sim0)
                subnormer=sweep[0][-1]

                if len(tsteps)>4:
                    step1_0=sim0[-1]
                    step1=transitionmatrix(*start, *conds1[:4], Vs[-1][V]) # second condition
                    sim1=simulate(t[0:tsteps[4]-tsteps[3]], step1, step1_0) # second simulation
                    if sim1[-1]["message"]!="Integration successful.": return inf,
                    sim1=sim1[0]
                    sweep[1]=gatingcurrent(step1, sim1, start, flux)#normalized_anioncurrent(openstates,sim1)
                    if "forw" in experiment or ("App" in experiment and "App2" not in experiment):
                        subnormer=[sweep[1][-1]]

                if len(tsteps)>6:
                    step2_0=sim1[-1]
                    sim2=simulate(t[0:tsteps[6]-tsteps[5]], step0, step2_0) # simulated return to first condition
                    if sim2[-1]["message"]!="Integration successful.": return inf,
                    sim2=sim2[0]
                    sweep[2]=gatingcurrent(step0, sim2, start, flux)

                sweep*=(Imax/subnormer)

                if experiment=="WTintGlut40Cl_pH55":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])

                elif experiment=="WTintGlut40Cl_pH5":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])

                elif experiment=="WTintGlut40Cl_pH5App":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])
                    e1=(sweep[1]-data[V][tsteps[3]:tsteps[4]])
                    e2=(sweep[2]-data[V][tsteps[5]:tsteps[6]])

                elif experiment=="WTintGlutpH5_40ClApp":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])
                    e1=(sweep[1]-data[V][tsteps[3]:tsteps[4]])
                    e2=(sweep[2]-data[V][tsteps[5]:tsteps[6]])

                elif experiment=="WTintGlutpH55_140ClApp": # activation
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])
                    e1=(sweep[1]-data[V][tsteps[3]:tsteps[4]])

                elif experiment=="WTintGlutpH55_140ClApp2": # deactivation
                    e1=(sweep[1]-data[V][tsteps[3]:tsteps[4]])


                elif experiment=="WTintAsp40Cl_pH5":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])

                elif experiment=="WTintAsp40Cl_pH5App":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])
                    e1=(sweep[1]-data[V][tsteps[3]:tsteps[4]])
                    e2=(sweep[2]-data[V][tsteps[5]:tsteps[6]])

                elif experiment=="WTintAsppH55_40ClApp":
                    e0=(sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])
                    e1=(sweep[1][25:]-data[V][tsteps[3]+25:tsteps[4]])
                    e2=(sweep[2][25:]-data[V][tsteps[5]+25:tsteps[6]])

                for segment in [0,1,2]:
                    W_segment=[W0,W1,W2][segment]
                    e_segment=[e0,e1,e2][segment]
                    if not isinstance(e_segment, int):
                        e_segment=e_segment[10:] # testing without fitting I-inst peaks
                        if "App2" not in experiment:
                            e_segment[:len(e_segment)//5]*=2 # doubles err for 1st 20% if leaksubtracted

                    fitmultiplier=5#*9
                    if experiment=="WTintAsppH55_40ClApp": # weightrule
                        fitmultiplier*=2
                    error=fitmultiplier*e_segment**2
                    ERRS[segment].append(error)
                    err+=np.nansum(error)*W_segment

                peaks,weighedpeaks=peakweigh(sweep, err2)
                PEAKS.append(peaks**.5)
                err2+=weighedpeaks

            if mode in [1,3] or worsenfactor>0:
                OUTTEXT[experiment]={}
                if mode==1: print(f'{err-errprint}\t {experiment} ({round(100*(err-errprint)/(errs[n]*worsenfactor), 3)})')

                for segment in [0,1,2]:
                    OUTTEXT[experiment][segment]={"rmsd":np.nansum([np.nanmean(x) for x in ERRS[segment]]), "err":sum([np.nansum(x)*[W0,W1,W2][segment] for x in ERRS[segment]])}

            if mode==0 and worsenfactor>0 and errs[n] and sum([OUTTEXT[experiment][x]["err"] for x in [0,1,2]])>errs[n]*worsenfactor:
                return inf,

        err+=(10*err2)
        if mode!=0:
            if mode==1: print(err2, "\t weirdpeaks")
            OUTTEXT[protein]["peakweight"]={"max":max(PEAKS), "err":err2}
        ########################################################################################################
        errprint=err
        for i,key in enumerate(deps):
            dep=deps[key]
            Cls=dep["Cls"]; pHs=dep["pHs"]
            Vs=dep["Vs"]
            data=dep["data"]
            ys0=[]
            for Cl in Cls:
                for pH in pHs:
                    for V in Vs:
                        A=transitionmatrix(*start, pH, 7.4, Cl, .14, V)
                        AW=initialvalue(states,A)
                        ys0.append(stationarycurrent(A, AW, start, flux))#(normalized_anioncurrent(openstates,AW))
            ys0=np.asarray(ys0)
            ys=ys0/ys0[data.index(1)]

            diffs=(ys-data)**2#[abs(a) for a in ys-data]

            if ys[0]>1:
                diffs*=100
            if protein=="AspWT" and i==0: # weightrule
                diffs*=50
                if ys[0]>.3:
                    diffs[0]*=1000
            elif protein=="GlutWT" and i==0:
                err+=diffs[0]*1000000

            errprint2=err
            err+=[clweight,phweight][i]*sum(diffs)
            if mode!=0:
                OUTTEXT[protein][["clweight","phweight"][i]]={"rmsd":np.mean(diffs)**.5, "err":err-errprint2}
                if mode==1:
                    print(protein, ["Cldep","pHdep"][i], [round(x, 2) for x in ys])

        if mode==1: print (err-errprint,"\t direct pH/Cl dep")
        ########################################################################################################
        errprint=err #              pH,  M Cl,  mV
        for i,params in enumerate([[5.5, .04, -.16]]):
            ph,cl,v = params
            A=transitionmatrix(*start, ph, 7.4, cl, .140, v)
            AW=initialvalue(states,A)

            transport = stationarycurrent(A, AW, start, flux)
            TR=transportrate[SUB]
            err=err+transportweight*(transport-TR)**2
            # if mode==0 and not (TR*.98)>transport>(TR*1.02): return inf,
        if mode!=0:
            if mode==1: print(err-errprint, f'\t transport /s (vs {TR}{[" -G⁻ + +H⁺", ""][SUB]}):', transport)
            OUTTEXT[protein]["transportweight"]={"rmsd":abs((transport-TR)/TR), "err":err-errprint}
        ########################################################################################################
        errprint=err
        pKas=[]
        for i in Hdep:
            protonation=start[i]
            deprotonation=start[(i//4)*4+[1 if i/2==int(i/2) else 0][0]]
            pKas.append(-np.log10(deprotonation/protonation))
        MIN,MAX=3+1, 11-1
        err+=sum([pkaweight*(x**2) for x in [max(MIN-pka, pka-MAX, 0) for pka in pKas]])
        if protein=="GlutWT":
            target=5.5; a=abs(pKas[-1]-target); b=abs(pKas[-3]-target)
            err+=pkaweight*10000*((a+b)*min(a,b)**2)**2 # weightrule

        if mode!=0:
            if mode==1:
                print("pKa",",".join([variables[j] for j in Hdep]),"=",[round(x,1) for x in pKas])
                print(err-errprint,"\t err pka")
                print(err,"\t TOTAL ERR",protein)#,400000/openchance_Cl
            OUTTEXT[protein]["pkaweight"]={"max":max([max(MIN-pka, pka-MAX, 0) for pka in pKas]), "err":err-errprint}

        if err!=err:return inf,
        if err<0:return inf,
        RETURN.append(err,)
        ########################################################################################################
    if mode == 1:
        items=[]
        for KEY in OUTTEXT.keys():
            wait=[]
            for key in OUTTEXT[KEY].keys():
                value=OUTTEXT[KEY][key]
                name=[x for x in value.keys() if x!="err"][0]
                if key!="err":
                    if "_" not in KEY:
                        items.append(value[name])
                    else:
                        wait.append(value[name])
            if "_" in KEY:
                items.append(sum(wait))

        if mode==1:
            print("reference (error metric) =", items, end="\n\n")
            return np.sum(RETURN), OUTTEXT

    return np.sum(RETURN),#end
if __name__ == '__main__':
    "Running with a non-empty <from_parameter_set> below will simulate from it, rather than opening the <version> file below"
    from_parameter_set=[]#+optimized_parameter_set

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-name',default='GA_sym_output',type=str,help='Output file name')
    parser.add_argument('-id',default=0,type=int,help='Output ID')
    parser.add_argument('-protein',choices=['GlutWT','AspWT'],default='GlutWT',help='Constructs available')
    parser.add_argument('-nprocesses',default=1,type=int,help='Number of parallel processes to run')
    parser.add_argument('-pop_size',default=50,type=int,help='Population of each generation')
    parser.add_argument('-ngen',default=1000,type=int,help='Number of generations')
    parser.add_argument('-cxpb',default=0.7,type=float,help='Crossover rate')
    parser.add_argument('-mutpb',default=0.5,type=float,help='Mutation rate')
    parser.add_argument('-checkpoint',default=10,type=int,help='Frequency of data saving in number of generations')
    parser.add_argument('-resume',choices=[0,1],default=0,help='Resume from file identified by its name and ID')
    args = parser.parse_args()
    mode = 0
    NPROCESSES         = args.nprocesses # extend of multiprocessing
    pop_size           = args.pop_size # population size
    checkpoint         = args.checkpoint # how many generations pass between saving progress
    version            = args.id
    protein            = args.protein
    CXPB               = args.cxpb
    MUTPB              = args.mutpb
    NGEN               = args.ngen
    filename           = args.name 
    resume             = args.resume
    slowlim            = 10000 # upper limit for slower (conformation) transitions
    fastlim            = 5e9 # upper limit for faster (ligand assocating) transitions
    chargelim          = 1 # limit to charge movement/electrogeneity
    transportrate      = [-561*2, (-561*2)*2.3] # Glut, Asp
    sigroller          = 0 # used to rotate through different values for sigma below
    slowsigs           = [1, .75, .66, .5, .33, .25, .1, .05, .005, .001][6:] # factor by which sigma is reduced
    no_Asp             = 0 # disable the Aspartate component of the model
    MODEL              = 0 # in case of multiple accessible models

    VERSION=f'{version:04d}'
    fromfile=0
    errs = [np.inf] * 6
    worsenfactor = 0
    inputfile=f'GlutWT_AspWT_measurements.pkl'
    with open(inputfile,'rb') as f:
        experimental_data = pkl.load(f)

    if from_parameter_set:
        start0=from_parameter_set
    MODELS={
    0:{"ID":"0", "samelen":52, "g_len":108, "a_len":108, "slowones":[100, 101, 104, 105], "clapp":"", "start0":start0},
    }
    ID, samelen, g_len, a_len, slowones, clapp, start0=[MODELS[MODEL][x] for x in ["ID","samelen","g_len","a_len","slowones","clapp","start0"]]

    autoH=[0, 1e8][1] # automatically increases rates in this category to given nonzero number
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category=LinAlgWarning)
    lastgen=resumegen=0

    if from_parameter_set:
        print(f'Starting from parameter set instead of files, will save to {version} if needed.')
        START=start0
    else:
        try:
            with open(VERSION+filename, "rb") as out:
                gen_errs=pkl.load(out)
            (lastgen,lastrmsd),lastweights,START=gen_errs[-1] # -1 by default
            resumegen=[0,lastgen][resume]
            MODEL, ID, samelen, g_len, a_len, slowones, clapp, start0=selectmodel(START, MODELS)

            print(f'Using start found in {VERSION+filename},{[" replacing", " resuming"][1*(resume or mode!=0)]} gen {lastgen}, model {MODEL}, mode {mode}')

        except FileNotFoundError:
            if fromfile:
                with open(fromfile, "r") as out:
                    START=eval(out.readline())
                    MODEL, ID, samelen, g_len, a_len, slowones, clapp, start0=selectmodel(START, MODELS)
                    print(f'Entered file {fromfile} is not a save. Used start from it, model {MODEL}.')
            else:
                START=start0
                print(f'Start from {VERSION+filename} unavailable, used start from start0, default model {MODEL}.')
            resume=0
    #
    _,Hdep,Cldep,Sdep,SIG0,noVdep,flux,closingstates=modelselect(START, "SYM"+ID)
    if no_Asp: SIG0+=list(range(g_len, len(START)))
    basevariables="123456789"+"abcdefghijklmnopqrstuvwxyz"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    combinedvariables=basevariables[:samelen//4]+basevariables[samelen//4:g_len//4]+basevariables[samelen//4:a_len//4]

    def AUTOH(START, autoH=autoH, Hdep=Hdep, bound=1e5):
        """Increases rates in this category to given nonzero number"""
        if autoH and max(START)<bound:
            for i in Hdep:
                if START[i]<autoH:
                    START[i]=autoH
        return START

    START=AUTOH(START)

    limsmin=[0, 0, -chargelim, 0]*int(len(START)/4)
    limsmax=[slowlim, slowlim, chargelim, 1]*int(len(START)/4)
    for f in Hdep+Cldep+Sdep: # increased upper limit for (not just small?)  ion binding
        limsmax[f//4*4]=limsmax[f//4*4+1]=fastlim
        if f in Hdep: # optional autoH-based lower bound for H binding
            limsmin[f]=autoH
    for s in slowones: # rates limited to 1 (due to infeasible opening)
        limsmax[s]=1
    lims=list(zip(limsmin, limsmax))
    LIMRANGE=np.array(limsmax)-limsmin

    def sigcalc(start, sig0, limrange, slowsig):
        """Calculates sigma, the average standard deviation for mutation size."""
        sig=list(limrange*slowsig)
        for i in sig0:
            sig[i]=0
        return sig

    def STARTcalc(START, ID, samelen, g_len, a_len, startcalc, alt=0): # if alt: different micoscopic reversibility frees up sig0 variables
        """Splits combined parameter set START into substrate-specific ones"""
        startG=startcalc(START[:g_len], "alt"*alt+"GlutWT"+ID)
        startA=startcalc(START[:samelen] + START[g_len : g_len + a_len - samelen], "alt"*alt+"AspWT"+ID)
        return startG+startA[samelen:]

    def loadin(START, protein, model, SUB, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, alt=0):
        start=[START[:g_len], START[:samelen]+START[g_len:]][SUB]
        noVdep=sig0=None; model, Hdep, Cldep, Sdep, sig0, noVdep, flux, closingstates=modelselect(start, model+ID)
        if model == 'GlutWT':
            transitionmatrix=GlutWT0_transitionmatrix
            variables=GlutWT0_transitionmatrix_args
            states=GlutWT0_states
        elif model == 'AspWT':
            transitionmatrix=AspWT0_transitionmatrix
            variables=AspWT0_transitionmatrix_args
            states=AspWT0_states
        datasets, deps=loaddata(protein)

        slowones=(slowones if protein=="GlutWT" else []) # doubly protonated in/outward transition with Glut is restricted

        limsmin=[0, 0, -chargelim, 0]*int(len(start)/4)
        limsmax=[slowlim, slowlim, chargelim, 1]*int(len(start)/4)
        for f in Hdep+Cldep+Sdep: # increased upper limit for ion binding
            limsmax[f//4*4]=limsmax[f//4*4+1]=fastlim
            if f in Hdep: # optional autoH-based lower bound for H binding
                limsmin[f]=autoH
        for s in slowones: # rates limited to 1 (due to infeasible opening)
            limsmax[s]=1

        start=startcalc(start, "alt"*alt+model+ID) # detailedbalance

        return datasets, deps, start, model, Hdep, Cldep, Sdep, sig0, noVdep, flux,\
            closingstates, states, variables, limsmin, limsmax, errs, worsenfactor, transitionmatrix

    SimVarGen=[]; minlen=[1]*len(START)

    START=STARTcalc(START, ID, samelen, g_len, a_len, startcalc)
    if mode >0:
        try:
            err,outtext=evaluate(START,experimental_data, mode, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len)
        except ValueError:
            sys.exit("Invalid START.")

    check=evaluate(START,experimental_data, 0, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len)
    if inf not in check:#np.isfinite(check):
        olderfitness,outtext=evaluate(START,experimental_data, 1, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len)
    else:
        test=evaluate(START,experimental_data, 1, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len)
        if "*" in test:
            evaluate(START,experimental_data, 3, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len)
        sys.exit("Initial simulation is invalid, interrupting.")

    if resume==0:
        gen_errs=[[[0, olderfitness], deepcopy(weights), START]]
    SIG=sigcalc(START, SIG0, LIMRANGE, slowsigs[0])
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attribute", random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(START))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIG, indpb=0.5)
    toolbox.register("select", tools.selTournament, tournsize=3)
    if mode==0: reference=[inf]*19

    toolbox.register("evaluate", evaluate)
#    toolbox.register("evaluate", evaluate, mode=mode, autoH=autoH,
#             chargelim=chargelim, slowlim=slowlim, fastlim=fastlim, slowones=slowones, modelfile=modelfile,
#             ID=ID, samelen=samelen, g_len=g_len, a_len=a_len, reference=reference)
    pop = toolbox.population(n=pop_size) # population size

    for i in range(pop_size):
        pop[i][:]=START

    oldfitness=inf; savedfitness=inf
    fittest=START
    print(f'Gen {resumegen}, err {olderfitness}')
    counter=newfile=0
    reweight=[]
    for gen in range(NGEN)[:]:
        gen+=(lastgen*resume)

        if mode==0:
            slowsig=slowsigs[sigroller%len(slowsigs)]
            SIG=sigcalc(START, SIG0, LIMRANGE, slowsig)

        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=SIG, indpb=0.5)

        offspring = toolbox.select(pop, len(pop))

        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        if NPROCESSES > 1:
            pool = multiprocessing.Pool(processes=NPROCESSES)
            toolbox.register("map", pool.map)
            fitnesses = list(toolbox.map(evaluate_individual, invalid_ind))
            #fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
        else:
            fitnesses = list(toolbox.map(evaluate_individual, invalid_ind))
            #fitnesses = list(toolbox.map( lambda ind: toolbox.evaluate(ind, experimental_data, mode,autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len), invalid_ind))
#    toolbox.register("evaluate", evaluate, mode=mode, autoH=autoH,
#             chargelim=chargelim, slowlim=slowlim, fastlim=fastlim, slowones=slowones, modelfile=modelfile,
#             ID=ID, samelen=samelen, g_len=g_len, a_len=a_len, reference=reference)
#            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        newfitness=min(fitnesses)[0]

        if np.isfinite(newfitness) and newfitness<=1.0*oldfitness:
            pop[:] = offspring
            oldfitness=newfitness

            fitnesses = list(toolbox.map(evaluate_individual, offspring))
            fittest0=offspring[np.argmin(fitnesses)]

            fittest=STARTcalc(fittest0, ID, samelen, g_len, a_len, startcalc)
            if min(fittest[:(g_len if not a_len else None)])<-1:
                sys.exit("Gen",gen,"warning, interrupting because bounds are not applied:", min(fittest))

        if counter>=checkpoint and mode!=-2: ############################################### SimVarGen
            err,outtext=evaluate(fittest,experimental_data, 1, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len)
            gen_errs.append([[gen, err], deepcopy(weights), fittest])
            with open(VERSION+filename, "wb") as out: # "wb" to write new, "rb" to read
                pkl.dump(gen_errs, out)
            counter=0
        counter+=1; sigroller+=1
    print(f'Concluded at generation {gen+1}/{NGEN}.')
