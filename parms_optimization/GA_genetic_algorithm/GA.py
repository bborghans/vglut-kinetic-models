mode=( # 0:optimization via genetic fit algorithm, 1:details, 2:state distribution, 3:flux, 4:test optimization without starting it, 5:manual save, 6:worsenfactor, -1:VariantGen, -2:VarGen compile, -3:rangecheck
1
)
modelfile="GA_model"
from GA_model import modelselect, loaddata, startcalc, initialvalue, simulate, gatingcurrent, stationarycurrent, fluxstates3, plotstates2#, paramplot, writestart, toplegend2
import sys; from copy import deepcopy; from deap import base, creator, tools; from random import random; import inspect; import numpy as np; from math import inf; import pickle as pkl; import multiprocessing; import warnings; from scipy.linalg import LinAlgWarning; from datetime import datetime; import matplotlib.pyplot as plt;
np.set_printoptions(legacy='1.25')

def evaluate(START, mode, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim,
        slowones, modelfile, ID, samelen, g_len, a_len, modelselect, alt=0, reference=[inf]*19):
    RETURN=[]
    OUTTEXT={"GlutWT":{}, "AspWT":{}}
    for SUB,(protein,model) in enumerate([["GlutWT"]*2, ["AspWT"]*2][:(1 if no_Asp else None)]):
        if not [g_len, a_len][SUB]:
            continue
        err=err2=0
        datasets, deps, start, model, Hdep, Cldep, Sdep, sig0, noVdep, flux, closingstates, states,\
            variables, limsmin, limsmax, errs, worsenfactor, transitionmatrix=loadin(START, protein, model, SUB, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect, loaddata, alt)
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
            if mode in [0, -1, -2]: return inf,
            else: print("START VALUE OUT OF BOUNDS:",list(np.where(np.concatenate([startcheck-limsmin,limsmax-startcheck])<0)[0]%len(start)))

        for n,dataset in enumerate(datasets):
            PEAKS=[]
            ERRS=[],[],[]
            experiment,conds0,conds1,Vs,tsteps,freq,normrange=dataset
            e0,e1,e2=[0]*3
            W0,W1,W2=[weights[experiment][x] for x in [0,1,2]]
            data=eval(experiment+(clapp if "ClApp" in experiment else ""))#####################################################

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

                if mode==2 and V==0:
                    plotstates2([sim0,sim1,sim2], freq, states, tsteps, experiment, label=1, show=show, save=save)

                if mode==3 and V==0:
                    fluxstates3(A0=[step0,step1,step0], y0=[sim0,sim1,sim2], freq=freq, states=states, tsteps=tsteps, flux=flux, experiment=experiment, svg_or_png=1, label=1, full=0,\
                        focus=[x for x in flux if (states[x[1]]+states[x[2]]).count("Cl")==1], show=show, save=save)

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

            if mode in [1,2,3,4,5,6] or worsenfactor>0:
                OUTTEXT[experiment]={}
                if mode==1: print(f'{err-errprint}\t {experiment} ({round(100*(err-errprint)/(errs[n]*worsenfactor), 3)})')

                for segment in [0,1,2]:
                    OUTTEXT[experiment][segment]={"rmsd":np.nansum([np.nanmean(x) for x in ERRS[segment]]), "err":sum([np.nansum(x)*[W0,W1,W2][segment] for x in ERRS[segment]])}

            if mode==0 and worsenfactor>0 and errs[n] and sum([OUTTEXT[experiment][x]["err"] for x in [0,1,2]])>errs[n]*worsenfactor:
                return inf,

            if mode==4 and worsenfactor>0 and errs[n] and sum([OUTTEXT[experiment][x]["err"] for x in [0,1,2]])>errs[n]*worsenfactor:
                errsum=sum([OUTTEXT[experiment][x]["err"] for x in [0,1,2]])
                return inf, "*"*bool(errsum>errs[n]*worsenfactor)

            if mode==6:
                errsum=sum([OUTTEXT[experiment][x]["err"] for x in [0,1,2]])
                print(f'{errsum}_{experiment}{"*"*bool(errsum>errs[n]*worsenfactor)}')

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
    if mode in [4,5]:
        return np.sum(RETURN), OUTTEXT

    if mode in [-2, -1, 1]:
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

        if mode==-1:
            return np.sum(RETURN), items
        elif mode==-2:
            if inf in reference:
                return np.sum(RETURN), items
            elif any(x2 > x1*1.25 for x2,x1 in zip(items, reference)):
                return inf,
            else:
                return np.sum(RETURN),
        elif mode==1:
            print("reference (error metric) =", items, end="\n\n")
            return np.sum(RETURN), OUTTEXT

    return np.sum(RETURN),#end
if __name__ == '__main__':
    "Running with a non-empty <from_parameter_set> below will simulate from it, rather than opening the <version> file below"
    optimized_parameter_set=[1095269177.4675012, 1000, -0.15421903708066254, 0.8509538641493256, 294174013.83132875, 16376705.773529066, -0.9967660039561052, 5.105759887117164e-08, 7526.779981268539, 8755.95071396453, -0.5700467744733442, 0.7059504792055679, 146565504.59883535, 1000, -0.9236113885823142, 0.1721768604503449, 149429515.2266242, 1000, -0.9649094515782813, 0.9930663686610615, 1392791622.0090842, 366687390.89139754, 0.022637854495137083, 0.7430194493895187, 858176802.0900823, 1000, -0.5163234394000882, 0.9986691280243428, 3212.92106651059, 9926.839789042622, -0.8286873677961262, 0.9859689244668174, 371641264.50691557, 1000, -0.9630734564362926, 0.26552353076057417, 948675410.5158807, 1000, -0.8141637772363048, 0.010505666995604337, 1000, 6.3782488035057705, -0.22555708926142104, 0.10310636478890667, 619913.6615236911, 10025.933625721707, -0.18609502140744272, 0.0316887043585846, 434034053.1486239, 44565489.24233581, -0.33684069574941927, 0.006224657894693473, 1000, 68816726.18266672, -0.9959485781587907, 0.9985902696952598, 2851119199.849195, 384065263.05951196, -0.3116734708889899, 0.5904175573026531, 716.252408479346, 6.502617277496788, -0.7275316147558513, 0.00018545550144734605, 4412370905.225374, 34907639.8473314, -0.8461944782689655, 0.8150262350336182, 139675303.04375777, 4405553317.2328, 0.3120455741787267, 0.0030948318551721568, 39155219.24301096, 4381507102.29029, -0.35479583886563254, 0.0364869609120432, 1323711382.8038626, 57204735.049285, 0.3282948635263183, 0.3824270354721183, 1553.254393481456, 991.96640568565, -0.19652507020138077, 0.3414320950549635, 4973370471.788941, 1981918.4533993315, -0.9578382417484005, 0.8615551441521289, 3172613847.747088, 676691602.8925138, 0.991233353454394, 0.9907389196556018, 37519000.22619294, 32126.169801267664, 0.5652744534936812, 0.2922048000260236, 4998862778.58027, 215608.26464810251, 0.4536306900142463, 0.00034077558946806006, 0.32678865898770515, 0.00017424001288291087, -0.2620526221358268, 0.16972523065214234, 0.80733037830023, 0.004754468375198573, -0.4826581754760997, 0.1858679668584576, 601376.111043681, 849220.7908282045, -0.05184241768891382, 0.15486559508278752, 554120689.9684634, 198900316.6133083, -0.37380529960455966, 0.06916396389343288, 196.7439166746461, 303.7106685375859, -0.9498109714536906, 0.22508623366342045, 4926299389.519943, 1143815345.4690511, -0.9422169878493903, 0.1305938063378658, 3093598987.925261, 3749791491.1145782, 0.9999816970859092, 0.995104556427794, 18449165.888446257, 1000, -0.5723491620342643, 0.6741512438644036, 3984484877.5182137, 1000, 0.5408139140095878, 0.9996818060832617, 18.179586437540788, 0.2371140431561412, -0.4913384018459648, 0.9779002456877648, 100062232.24021704, 1000, -0.43460869029574284, 4.420393898556687e-05, 109751994.2081706, 1000, 0.7895771996105664, 0.5335021459124639, 1000, 1.1139129108834525, -0.9435698842904114, 0.8706681897734239, 866714089.9156774, 41.55485685508216, -0.4359615867367639, 0.9674364178488436, 219.5960595239116, 219.27407435618878, -0.5182226596985211, 0.004870366039147044, 6375.505570847915, 3311.234222226865, -0.46676100615129545, 0.8902492258134688]
    from_parameter_set=[]#+optimized_parameter_set
    if sys.argv==[""] or 0: # activates when run in script editor, change 0 to nonzero to use <version> file below in console as well
        version=1000#1234 #
        resume=0; fromfile=0; argv_args=[]
    else: #      ^- manual version select outside of console, Ctrl+F6 to switch
        resume=0
        if len(sys.argv)<2: sys.exit('Run as "python GA.py <version>" (mode=<mode> or other keywords)')
        version = sys.argv[1]
        del sys.argv[1]

        if len(version)>3 and "." in version:
            fromfile=version; version=int(version.split(".")[0])
        else:
            fromfile=0; version=int(version)
    VERSION=str(version).zfill(4)

    save_index_in_file = 0-1 # choose to continue from (0: first/oldest, or) -1: last/newest entry in file
    cluster            = 0 # enable cluster for multiprocessing
    NPROCESSES         = [2, 128][cluster] # extend of multiprocessing
    pop_size           = [50, 1000][cluster] # population size
    checkpoint         = 500 # how many generations pass between saving progress
    sigroller          = 0 # used to rotate through different values for sigma below
    slowsigs           = [1, .75, .66, .5, .33, .25, .1, .05, .005, .001][6:] # factor by which sigma is reduced
    no_Asp             = 0 # disable the Aspartate component of the model
    MODEL              = 0 # in case of multiple accessible models
    show               = 0 # plot settings
    save               = 1 # plot settings

    A="""2687280566.8554854_WTintGlut40Cl_pH55
    124764969.72033526_WTintGlut40Cl_pH5
    2088197643914.9985_WTintGlut40Cl_pH5App
    903721147.0244515_WTintGlutpH5_40ClApp
    1754102.1613024566_WTintGlutpH55_140ClApp
    244923.7993450683_WTintGlutpH55_140ClApp2"""
    B=A.split("\n")
    errs=[eval(er.split("_")[0]) for er in B]#"      "
    worsenfactor=1.*0
    WFG=[errs,worsenfactor]

    A="""2365285.1840912295_WTintAsp40Cl_pH5
    8781541040413425.0_WTintAsp40Cl_pH5App
    208095960180.2058_WTintAsppH55_40ClApp"""
    B=A.split("\n")
    errs=[eval(er.split("_")[0]) for er in B]#"      "
    worsenfactor=1.*0
    WFA=[errs,worsenfactor]

    start0=[1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1, 1, 0, 0.5, 1, 1, 0, 0.5]+[1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 0, 0.5, 1000, 1000, 0, 0.5]
    if from_parameter_set:
        start0=from_parameter_set
    MODELS={
    0:{"ID":"0", "samelen":52, "g_len":108, "a_len":108, "slowones":[100, 101, 104, 105], "clapp":"", "start0":start0},
    }
    ID, samelen, g_len, a_len, slowones, clapp, start0=[MODELS[MODEL][x] for x in ["ID","samelen","g_len","a_len","slowones","clapp","start0"]]

    slowlim=10000 # upper limit for slower (conformation) transitions
    fastlim=5e9 # upper limit for faster (ligand assocating) transitions
    chargelim=1 # limit to charge movement/electrogeneity
    transportrate=[-561*2, (-561*2)*2.3] # Glut, Asp
    filename="GA_sym_output" # models from GA_models
    moredata="outtext"

    autoH=[0, 1e8][1] # automatically increases rates in this category to given nonzero number
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings(action='ignore', category=LinAlgWarning)

    if len(sys.argv)>1:
        args = dict(arg.split('=', 1) for arg in sys.argv[1:]) # allows console kwargs to override variables
        for key, value in args.items():
            if key in globals():
                globals()[key] = int(value) if value.isdigit() else value # update variables received through argv
                print(f'passed with argument {key}={value}')

    ###########################################################################
    if mode==-3: # fuse SimVarGen files
        OVERWRITE=0+1
        nr=version#"9999"
        SimVarGen0=[]
        for part in range(15):
            try:
                with open(f'SimVarGen{nr}{part}.pkl', "rb") as out:
                    test=pkl.load(out)
                SimVarGen0.extend(test)
            except FileNotFoundError: break
        minlen=[len(set(col)) for col in zip(*SimVarGen0)]

        unduped=[]
        for variableset in SimVarGen0:
            if variableset not in unduped:
                unduped.append(variableset)
        print(f'{len(SimVarGen0)} reduced to {len(unduped)} uniqes, lowest={minlen.count(min(minlen))}x{min(minlen)}')
        with open(f'SimVarGen{nr}{["_"+"".join(map(str, range(part-1))), "0"][OVERWRITE]}.pkl', "wb") as out:
            pkl.dump(unduped, out)
        sys.exit()
    ###########################################################################

    def selectmodel(start, MODELS): # selects internal model format
        model=[int(MODELS[x]["ID"]) for x in MODELS.keys() if len(start)==MODELS[x]["g_len"]+MODELS[x]["a_len"]-MODELS[x]["samelen"]]
        if model==[]:
            raise FileNotFoundError
        else:
            model=model[0]

        ID, samelen, g_len, a_len, slowones, clapp, start0=[MODELS[model][x] for x in ["ID","samelen","g_len","a_len","slowones","clapp","start0"]]
        return (model, ID, samelen, g_len, a_len, slowones, clapp, start0)

    for (protein,model) in [["GlutWT"]*2, ["AspWT"]*2]:
        exec(f'from {protein}_measurements import *')

    from GA_weights import weights

    lastgen=resumegen=0

    if from_parameter_set:
        print(f'Starting from parameter set instead of files, will save to {version} if needed.')
        START=start0
    else:
        try:
            with open(VERSION+filename, "rb") as out:
                gen_errs=pkl.load(out)
            (lastgen,lastrmsd),lastweights,START=gen_errs[save_index_in_file] # -1 by default
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

    def loadin(START, protein, model, SUB, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect, loaddata, alt=0):
        import importlib
        transitionmatrix=getattr(importlib.import_module(modelfile), f'{model+ID}_transitionmatrix')
        datasets, deps=loaddata(protein)

        start=[START[:g_len], START[:samelen]+START[g_len:]][SUB]

        noVdep=sig0=None; model, Hdep, Cldep, Sdep, sig0, noVdep, flux, closingstates=modelselect(start, model+ID)
        variables=inspect.getfullargspec(transitionmatrix)[0][0:len(start)]
        states=inspect.getfullargspec(transitionmatrix)[3][0]

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

        errs,worsenfactor=[WFG,WFA][SUB]
        return datasets, deps, start, model, Hdep, Cldep, Sdep, sig0, noVdep, flux,\
            closingstates, states, variables, limsmin, limsmax, errs, worsenfactor, transitionmatrix

    SimVarGen=[]; minlen=[1]*len(START)
    if mode==-2:
        try:
            with open("SimVarGen"+VERSION+"0.pkl", "rb") as out:
                SimVarGen=pkl.load(out)
                minlen=[len(set(col)) for col in zip(*SimVarGen)]
                print(f'Loaded SimVarGen from file, {len(SimVarGen)} individuals, at worst has {minlen.count(min(minlen))}x{min(minlen)} uniques.')
        except FileNotFoundError:
            print(f'SimVarGen{VERSION} does not exist, will be generated.')
            SimVarGen=[START]
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
    #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

    if mode==-1:
        paired=0+1+1
        long=0+1
        n=51; results={}; do=[[x for x in range(len(START)) if (x%4 in [0,1] or True)], np.array(Cldep)+0, np.array(Sdep)+1][paired]
        base_err,reference=evaluate(START, -2, WFG, WFA, loadin, autoH, chargelim,
                        slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
        for r in do[:]:
            start2=START.copy()
            rate=start2[r]
            if r%4==3:
                x=np.concatenate([np.linspace(0, rate, int(n/2), endpoint=False), [rate], np.linspace(1, rate, int(n/2), endpoint=False)[::-1]])
            elif r%4==2:
                x=np.concatenate([np.linspace(-1, rate, int(n/2), endpoint=False), [rate], np.linspace(1, rate, int(n/2), endpoint=False)[::-1]])
            else:
                if not long:
                    x=np.logspace(-2, 2, n)*rate
                else:
                    fulllim=np.logspace(0, np.log10(fastlim), n-1) # rate constants: spread over full range
                    x=np.sort(np.append(fulllim, rate))

            if r//4 in set([x//4 for x in SIG0]):#[1,2, 9,10, 17,18, 33,34, 45,46, 61,62, 69,70, 77,78]
                alt=1 # model="altSYM"+ID;
            else:
                alt=0 # model="SYM"+ID;

            results[r]={"baserate":rate, "values":x, "errors":[]}
            for v,var in enumerate(x): # main loop
                start2[r]=var

                if paired:
                    start2[r+1]=var * (START[r+1] / rate) # modify kX0 to maintain ratio with k0X # v-- alt: manually select detailed balance where neither is calculated

                variant=evaluate(start2, -2, WFG, WFA, loadin, autoH, chargelim, slowlim,
                                 fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect, alt=[alt, 1, 0][paired])
                variant=variant[-1]
                statistic=max(variant/np.array(reference))
                results[r]["errors"].append(min(statistic, 10)) # limit output to 10x norm RSS

            plt.plot(x, results[r]["errors"], c="b")#AX[r]
            plt.scatter(rate, 1, c="b")
            plt.axhline(1.5, c="k")
            plt.title(f'{r}, {["rate1","rate2","z","d"][r%4]}')
            plt.ylim(0.5,10)
            if r%4<2: plt.xscale("log")
            plt.xlabel("Rate value")
            plt.ylabel("Relative error")
            if save: plt.savefig(f'{r}_{["rate1","rate2","z","d"][r%4]}.png')
            if show: plt.show()
            else: plt.close()
        with open(f'GA_{["long"*long, "Cl", "S"][paired]}rangecheck{VERSION}.pkl', "wb") as out: # "wb" to write new, "rb" to read
            pkl.dump([START, results], out)
        sys.exit()

    START=STARTcalc(START, ID, samelen, g_len, a_len, startcalc)
    if mode >0:
        try:
            err,outtext=evaluate(START, mode, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
        except ValueError:
            if mode not in [2,3,6]:
                sys.exit("Invalid START.")
        if mode==1:
            print(START)
            print(err)
        if mode==5:
            gen_errs=[[[0, err], deepcopy(weights), START]]
            with open("manual_"*0 +VERSION+filename, "wb") as out: # "wb" to write new, "rb" to read
                pkl.dump(gen_errs, out)
            print(f'Parameters saved to {version}GA_sym_output.')

    elif mode in [-2, 0]:
        check=evaluate(START, 0, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
        if inf not in check:#np.isfinite(check):
            olderfitness,outtext=evaluate(START, 1, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
        else:
            test=evaluate(START, 1, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
            if "*" in test:
                evaluate(START, 3, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
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
        if mode==-2:  ############################################################# SimVarGen
            SIG=[#
            # [x/100 for x in LIMRANGE], # 1%
            # [x/1000 for x in LIMRANGE], # .1%
            # [x/100 if x<=10000 else 100 for x in LIMRANGE], # 1%, 100 for fastlim
            [x/1000 if x<=10000 else 10 for x in LIMRANGE], # .1%, 10 for fastlim
                ][sigroller%1]
            check,reference=evaluate(START, -1, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
            print("SimVarGen, reference =", reference)
            if not np.isfinite(check) or inf in reference: sys.exit("Evaluate not ready to run SimVarGen.")

        toolbox.register("evaluate", evaluate, mode=mode, WFG=WFG, WFA=WFA, loadin=loadin, autoH=autoH,
                 chargelim=chargelim, slowlim=slowlim, fastlim=fastlim, slowones=slowones, modelfile=modelfile,
                 ID=ID, samelen=samelen, g_len=g_len, a_len=a_len, modelselect=modelselect, reference=reference)
        pop = toolbox.population(n=pop_size) # population size
        CXPB, MUTPB, NGEN = 0.7, 0.5, 100000000 # crossover rate, mutation rate, generations

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

            if cluster:
                pool = multiprocessing.Pool(processes=NPROCESSES)
                toolbox.register("map", pool.map)
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))
            else:
                fitnesses = list(toolbox.map(toolbox.evaluate, invalid_ind))

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            newfitness=min(fitnesses)[0]

            if np.isfinite(newfitness) and newfitness<=1.0*oldfitness:
                pop[:] = offspring
                oldfitness=newfitness

                fitnesses = list(toolbox.map(toolbox.evaluate, offspring))
                fittest0=offspring[np.argmin(fitnesses)]

                if mode==-2: ############################# SimVarGen
                    if not SimVarGen and newfile==0:
                        SimVarGen.append(START)
                    for ind, fit in zip(offspring, fitnesses):
                        if np.isfinite(fit):
                            SimVarGen.append(STARTcalc(ind, ID, samelen, g_len, a_len, startcalc))
                    if len(SimVarGen)>10000:
                        with open(f'SimVarGen{VERSION}{newfile}.pkl', "wb") as out: # "wb" to write new, "rb" to read
                            pkl.dump(SimVarGen, out)
                        SimVarGen=[]; print(f'Saved to SimVarGen{VERSION}{newfile}.pkl ({datetime.now().strftime("%H:%M")}).'); newfile+=1
                    if newfile>=15:
                        sys.exit("SimVarGen finished.")

                fittest=STARTcalc(fittest0, ID, samelen, g_len, a_len, startcalc)
                if min(fittest[:(g_len if not a_len else None)])<-1:
                    sys.exit("Gen",gen,"warning, interrupting because bounds are not applied:", min(fittest))

            if counter>=checkpoint and mode!=-2: ############################################### SimVarGen
                err,outtext=evaluate(fittest, 1, WFG, WFA, loadin, autoH, chargelim, slowlim, fastlim, slowones, modelfile, ID, samelen, g_len, a_len, modelselect)
                gen_errs.append([[gen, err], deepcopy(weights), fittest])
                with open(VERSION+filename, "wb") as out: # "wb" to write new, "rb" to read
                    pkl.dump(gen_errs, out)
                counter=0
            counter+=1; sigroller+=1
        print(f'Concluded at generation {gen+1}/{NGEN}.')
