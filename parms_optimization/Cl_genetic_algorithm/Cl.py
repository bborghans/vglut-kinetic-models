protein=["WT", "H120A"][0]
model="Cl" # model name is worked into the required _models file name below
import pickle
import argparse
from Cl_model import Cltransitionmatrix as transitionmatrix
from Cl_model import modelselect, loaddata, initialvalue, simulate, normalized_anioncurrent, startcalc

import os
import sys
from datetime import datetime
import warnings
import multiprocessing
from random import random
from deap import base, creator, tools
import inspect
import numpy as np

np.set_printoptions(legacy='1.25')

def evaluate(start, model="Cl", reference=[np.inf]*19):
    fresh_errs=[[] for _ in range(5)];
    start=startcalc(start, model)
    err=0; weirdpeaks=1
    startcheck=np.asarray(start)
    if min(np.concatenate([startcheck-limsmin, limsmax-startcheck]))<0:
        return np.inf,

    for n,dataset in enumerate(datasets):
        experiment,conds0,conds1,Vs,tsteps,freq,normrange=dataset

        if save_sim: # 1/3
            with open("write_sim.py", "a") as out:
                out.write(experiment+" = [\n"); np.set_printoptions(threshold=np.inf)

        data = experimental_data[experiment]
        #data=eval(experiment)
        iterable_Vs=Vs[-1] # selects list of variable Vs when there's also a stationary segment
        length=tsteps[-1]
        t=np.arange(length)/freq # total timespan
        errprint=err
        sim0=sim1=sim2= step0=step1=None
        for V in range(len(Vs[-1])): # =sweeps
            Imax=np.mean(data[V][normrange[0]:normrange[1]])

            sweep=np.asarray([0]*int((len(tsteps)-2)/2),dtype=object)
            step_0=transitionmatrix(*start,*conds0) # pre-V starting point
            step_groundstate=initialvalue(states,step_0)
            step0=transitionmatrix(*start,*conds0[:4],Vs[0][V%len(Vs[0])]) # first condition
            sim0=simulate(t[0:tsteps[2]-tsteps[0]],step0,step_groundstate) # first simulation
            if sim0[-1]["message"]!="Integration successful.":return np.inf,
            sim0=sim0[0]
            sweep[0]=normalized_anioncurrent(openstates,sim0)
            subnormer=sweep[0][-1]

            if len(tsteps)>4:
                step1_0=sim0[-1]
                step1=transitionmatrix(*start,*conds1[:4],Vs[-1][V]) # second condition
                sim1=simulate(t[0:tsteps[4]-tsteps[3]],step1,step1_0) # second simulation
                if sim1[-1]["message"]!="Integration successful.":return np.inf,
                sim1=sim1[0]
                sweep[1]=normalized_anioncurrent(openstates,sim1)
                if "forw" in experiment or experiment[-3:]=="App": subnormer=[sweep[1][-1]]

            if len(tsteps)>6:
                step2_0=sim1[-1]
                sim2=simulate(t[0:tsteps[6]-tsteps[5]],step0,step2_0) # simulated return to first condition
                if sim2[-1]["message"]!="Integration successful.":return np.inf,
                sim2=sim2[0]
                sweep[2]=normalized_anioncurrent(openstates,sim2)
                if "rev" in experiment: subnormer=sweep[2][-1]

            if "deact" in experiment:
                sweep[0]/=sweep[0][-1]*data[0][tsteps[3]]/data[0][tsteps[2]]
                sweep[1]*=(data[V][tsteps[3]]/sweep[1][0])
            else:
                sweep*=(Imax/subnormer)
            #########################################

            if experiment=="WTintCl140Cl_pH55leaksubtract":
                err=err+5e5*sum((sweep[0][tsteps[1]-tsteps[0]:tsteps[1]-tsteps[0]+50]-data[V][tsteps[1]:tsteps[1]+50])**2)
                err=err+9e5*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)

            elif experiment=="WTintCl140Cl_pH55Vdeact":
                err=err+1e5*sum((sweep[1]-data[V][tsteps[3]:tsteps[4]])**2)

            elif experiment=="WTintCl_180Cl_pH50":
                err=err+5e5*sum((sweep[0][tsteps[1]-tsteps[0]:tsteps[1]-tsteps[0]+50]-data[V][tsteps[1]:tsteps[1]+50])**2)
                err=err+1e4*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)

            elif experiment=="WTintCl_180Cl_pH65":
                err=err+5e6*sum((sweep[0][tsteps[1]-tsteps[0]:tsteps[1]-tsteps[0]+50]-data[V][tsteps[1]:tsteps[1]+50])**2)
                err=err+5e5*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)

            elif experiment=="WTintClpH5_40ClApp":
                err=err+1e2*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)
                err=err+1e5*sum((sweep[1]-data[V][tsteps[3]:tsteps[4]])**2)# *((4-V)*4)**2
                err=err+1e5*sum((sweep[2]-data[V][tsteps[5]:tsteps[6]])**2)# *((V+1)*4)**2
                weirdpeaks*=max(sweep[2]/sweep[2][0])

            elif experiment=="WTintClpH5_140ClApp":
                err=err+1e2*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)
                err=err+5e5*sum((sweep[1]-data[V][tsteps[3]:tsteps[4]])**2)
                err=err+5e5*sum((sweep[2]-data[V][tsteps[5]:tsteps[6]])**2) #*((V+1)*1)**2
                weirdpeaks*=max(sweep[2]/sweep[2][0])

            elif experiment=="WTintCl140Cl_pH55App":
                err=err+1e2*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)
                err=err+2e4*sum((sweep[1]-data[V][tsteps[3]:tsteps[4]])**2)
                err=err+2e4*sum((sweep[2]-data[V][tsteps[5]:tsteps[6]])**2)
                weirdpeaks*=max(sweep[2]/sweep[2][0])

            elif experiment=="WTintCl0Cl_pH5App":
                err=err+1e3*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)
                err=err+1e5*sum((sweep[1]-data[V][tsteps[3]:tsteps[4]])**2)
                err=err+1e5*sum((sweep[2]-data[V][tsteps[5]:tsteps[6]])**2)
                weirdpeaks*=max(sweep[2]/sweep[2][0])

            elif experiment=="WTintCl0Cl_pHdep_55":
                err=err+1e4*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)

            elif experiment=="WTintCl0Cl_pHdep_50":
                err=err+1e3*((sweep[0][tsteps[1]-tsteps[0]]-data[V][tsteps[1]])**2)
                err=err+3e3*sum((sweep[0][tsteps[1]-tsteps[0]:]-data[V][tsteps[1]:tsteps[2]])**2)

            elif experiment=="WTintCl_0Cl_pH55Vdeact_short":
                err=err+5e5*sum((sweep[1]-data[V][tsteps[3]:tsteps[4]])**2)

        if worsenfactor>0 and err-errprint>errs[n]*worsenfactor:
            return np.inf,

    if weirdpeaks:
        errprint=err
        err=err+1e11*(weirdpeaks-1)**2

    errprint=err; 
    deptypes=[np.argmax([len(deps[key]["pHs"]),len(deps[key]["Cls"]),len(deps[key]["Vs"])]) for key in deps]+["end"]
    for i,key in enumerate(deps):
        dep=deps[key]
        Cls=dep["Cls"]; x=Cls; pHs=dep["pHs"]
        Vs=dep["Vs"]
        data=dep["data"]
        CIs=dep["CIs"]
        deptype=deptypes[i]
        ys0=[]
        for Cl in Cls:
            for pH in pHs:
                for V in Vs:
                    A=transitionmatrix(*start,pH,7.4,Cl/1000,.140,V/1000)
                    AW=initialvalue(states,A)
                    ys0.append(sum([AW[i] for i in openstates]))
        ys0=np.asarray(ys0)
        ys=ys0/ys0[data.index(1)]; shape="od*"
        if Cls==[0]:
            Cl_fraction=deps["Cldep"]["data"][deps["Cldep"]["Cls"].index(0)]
            ys*=Cl_fraction
            data=np.array(data)*Cl_fraction
            CIs=np.array(CIs)*Cl_fraction
        reshaper=i+deptype-1

        diffs=[abs(a) for a in ys-data]
        err=err+depweight[i]*(sum(diffs)**2)
        if i==2:
            err+=1e4*depweight[i]*diffs[-1]**2##########################################################

            if save_sim: # 3/3
                with open("write_sim.py", "a") as out:
                    out.write(key+" = "+str(list(ys))+"\n\n")

    p_opens=[]; tclose=[]
    for i,params in enumerate([[5.5, 0, -.16],[5.5, .14, -.16]]):
        ph,cl,v = params
        A=transitionmatrix(*start, ph, 7.4, cl, .140, v)
        AW=initialvalue(states,A)
        tclose.append([A[x[0],x[1]] for x in closingstates])
        p_opens.append(sum([AW[x] for x in openstates]))
    openchance_0Cl,openchance_Cl=p_opens
    tclose_noCl_160,tclose_Cl_160=[1e6/sum(x) for x in tclose]

    for o,openchance in enumerate([openchance_0Cl, openchance_Cl]):
        target=[p_open*min(deps["Cldep"]["data"]), p_open][o]
        errprint=err
        if (not o and not 0.04>openchance>0.03) or (o and not .25>openchance>.23):
            err+=openchanceweight*(target-openchance)**2

    for ti,time in enumerate([tclose_noCl_160, tclose_Cl_160]):
        errprint=err
        if not 79>time>78:
            err+=opentimeweight*(time-opentime)**2

    errprint=err
    pKas=[]
    for i in Hdep:
        protonation,deprotonation=start[i],start[i+1]
        pka= -np.log10(deprotonation/protonation)
        pKas.append(round(pka,1))
        if not 9 >= pka >= 4:
            err=err+pkaweight*(pka-6)**2#return np.inf,
    if err!=err:return np.inf,
    if err<0:return np.inf,
    return [(err,), (err,fresh_errs)][False]
if __name__ == '__main__':#######################################################################################
    g=0 # use as: Miniforge Prompt -> mamba activate env-> python Cl.py mode=0
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-protein',choices=['WT','H120A'],default='WT',help='Constructs available')
    parser.add_argument('-name',default='Cl_sym_output',type=str,help='Output file name')
    parser.add_argument('-id',default=0,type=int,help='Output ID')
    parser.add_argument('-nprocesses',default=1,type=int,help='Number of parallel processes to run')
    parser.add_argument('-pop_size',default=50,type=int,help='Population of each generation')
    parser.add_argument('-ngen',default=1000,type=int,help='Number of generations')
    parser.add_argument('-cxpb',default=0.7,type=float,help='Crossover rate')
    parser.add_argument('-mutpb',default=0.5,type=float,help='Mutation rate')
    parser.add_argument('-checkpoint',default=10,type=int,help='Frequency of data saving in number of generations')
    args = parser.parse_args()
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="scipy.integrate")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")
        CXPB = args.cxpb
        MUTPB = args.mutpb
        NGEN = args.ngen
        # background settings
        savefile   = args.name
        version    = args.id
        cluster    = 1 # enable cluster for multiprocessing
        save_sim   = 0 # write simulated serial time course and pH/Cl dependence to file
        NPROCESSES = args.nprocesses
        pop_size   = args.pop_size
        checkpoint = args.checkpoint
        autoH      = [0, 1e8][1] # automatically increases rates in this category to given nonzero number
        slowones   = [12, 68] # limit infeasible rates, to 1
        slowlim    = 10000 # upper limit for slower (conformation) transitions
        fastlim    = 5e9 # upper limit for faster (ligand assocating) transitions
        chargelim  = 1 # limit to charge movement/electrogeneity
        p_open     = 0.24 # open probability at pH 5, 140 mM Cl-, 160 mV
        opentime   = 90 # microseconds

        # foreground settings
        pkaweight        = 1e4 # weight multipliers
        depweight        = [9e11, 9e11, 5e9] # Cl & pH dependency
        openchanceweight = 1e11
        opentimeweight   = 1e6
        slowsigs         = [1, .75, .66, .5, .33, .25, .1, .05, .005, .001][:]; sigroller=0 # factor by which sigma is reduced

        if protein == 'WT':
            with open('Cl_WT_measurements.pkl','rb') as f:
                experimental_data = pickle.load(f)
        elif protein == 'H120A':
            with open('Cl_H120A_measurements.pkl','rb') as f:
                experimental_data = pickle.load(f)

        start0=[1000, 1000, 0, 0.5]*20 # parameter set with initial guesses: rates=1000, z=0, d=0.5
        for i in slowones:
            start0[i]=1 # unprotonated channel opening is restricted

        try:
            with open(f'{version:04d}{savefile}.txt', "r") as out:
                read_in=out.readlines()[-1].replace("\n", "")#[-1]
                start=list(map(float, (read_in.split("[")[1].split("]")[0]).split(', ')))
                print("Resuming from file ID", version)
        except FileNotFoundError:
            print(f'No file found with ID {version}, starting from parameters in start0.')
            start = start0

        noVdep=sig0=None; model, Cldep, Hdep, sig0, noVdep, flux, closingstates=modelselect(start, model)
        # sig0+=list(range(16)) # place to add extra variables that need to be kept the same

        errs = np.inf
        worsenfactor=0 #Same weight growth for all parameters
        limsmin=[0, 0, -chargelim, 0]*int(len(start)/4)
        limsmax=[slowlim, slowlim, chargelim, 1]*int(len(start)/4)
        for f in Cldep+Hdep: # increased upper limit for (not just small?)  ion binding
            limsmax[f//4*4]=limsmax[f//4*4+1]=fastlim
            if f in Hdep: # optional autoH-based lower bound for H binding
                limsmin[f]=autoH
        for s in slowones: # rates limited to 1 (due to infeasible opening)
            limsmax[s]=1
        lims=list(zip(limsmin, limsmax))
        limrange=np.array(limsmax)-limsmin

        def sigcalc(start, sig0, slowsig):
            """Calculates sigma, the average standard deviation for mutation size."""
            sig = [list(limrange*slowsig), [x/4 for x in start], [abs(x)**.5 for x in start]][0]
            for i in sig0:
                sig[i]=0
            return sig

        variables=inspect.getfullargspec(transitionmatrix)[0][0:len(start)]
        states=inspect.getfullargspec(transitionmatrix)[3][0]
        openstates=[states.index(i) for i in [i for i in states if i[0]=="o"]]
        if autoH!=0:
            for i in Hdep:
                if start[i]<autoH:start[i]=autoH

        ##############################################################################
        creator.create("FitnessMin", base.Fitness, weights=(-1.0, ))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        toolbox = base.Toolbox()
        toolbox.register("attribute", random)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(start))
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("evaluate", evaluate)
        pop = toolbox.population(n=pop_size) # population size

        running_updates=0#+1
        start_time = datetime.now()
        olderfitness=evaluate(start)
        if olderfitness==(np.inf,):
            print("Initial fitness invalid, check constraints - script aborted")
            sys.exit()
        else:
            print(f'Script started at {datetime.now().strftime("%H:%M:%S")}, {olderfitness[0]}')
        for i in range(pop_size):
            pop[i][:]=start

        checkpoints=[checkpoint*x for x in range(1, 1+100000)]
        bestfitness=np.inf
        oldfit=list(toolbox.map(toolbox.evaluate, [start]))[0]
        x = -1
        fittest=start
        for g in range(NGEN):
            if g%3 in [0, 1]:
                sig0_2=sig0+[i for i in range(80) if i%4==0 or i%4==1]
            else:
                sig0_2=sig0
            slowsig=slowsigs[sigroller%len(slowsigs)]
            sig=sigcalc(startcalc(fittest, model), sig0_2, slowsig)###############################
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sig, indpb=0.5)

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

            if cluster:
                pool = multiprocessing.Pool(processes=NPROCESSES)
                toolbox.register("map", pool.map)
            fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            newfitness=min(fitnesses)[0]

            if np.isfinite(newfitness) and newfitness<bestfitness:
                pop[:] = offspring
                bestfitness=newfitness
                fitnesses = list(toolbox.map(toolbox.evaluate, offspring))
                fittest=startcalc(offspring[np.argmin(fitnesses)], model)
                x = -1
            else:
                x+=1

            if g in checkpoints:
                if g==checkpoints[0]:
                    writetype="w"
                else:
                    writetype="a"
                with open(f'{version:04d}{savefile}.txt', writetype) as out:
                    if g==checkpoints[0]:
                        out.write(f'# gen\tx\tbestfitness\tbest parameters\n')
                        out.write(f'{0}\t{0}\t{olderfitness[0]}\t{start}\n')
                    out.write(f'{g}\t{x}\t{bestfitness}\t{fittest}\n')
                if running_updates:
                    print(f'Save {g}, {datetime.now().strftime("%H:%M:%S")}, duration: {str(datetime.now() - start_time).split(".")[0]}, {bestfitness}.')
                start_time = datetime.now()
                olderfitness=newfitness
            sigroller+=1
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        if g:
            with open(f'{version:04d}{savefile}.txt', "a") as out:
                out.write(f'# manual stop\n{g}\t{x}\t{bestfitness}\t{fittest}\n')
                print(f'Saved on script termination: gen {g}, {bestfitness}, fail/dupe {x}: {fittest}.')
