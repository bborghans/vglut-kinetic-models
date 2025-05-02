mode=( # 0=optimization via genetic fit algorithm, 1=details & plots, 2=state distribution, 3=flux (additional modes may be used by external scripts)
1
)
protein=["WT", "H120A"][0]
model="Cl" # model name is worked into the required _models file name below
from Cl_model import Cltransitionmatrix as transitionmatrix
from Cl_model import modelselect, loaddata, initialvalue, simulate, normalized_anioncurrent, plotstates2, fluxstates2, startcalc, toplegend2
datasets, deps=loaddata(protein)

import sys; from datetime import datetime; import warnings; import multiprocessing; import matplotlib.pyplot as plt; from random import random; from deap import base, creator, tools; import inspect; import numpy as np;from scalebars import add_scalebar
exec(f'from {model}_{protein}_measurements import *')
np.set_printoptions(legacy='1.25')

if __name__ == '__main__':#######################################################################################
    g=0 # use as: Miniforge Prompt -> mamba activate env-> python Cl.py mode=0
    try:
        warnings.filterwarnings("ignore", category=UserWarning, module="scipy.integrate")
        warnings.filterwarnings("ignore", category=RuntimeWarning, module="deap.creator")

        "background settings"
        savefile   = "Cl_sym_output"#"fittest"+protein+model
        version    = 1234 #
        cluster    = 0 # enable cluster for multiprocessing
        save_sim   = 0 # write simulated serial time course and pH/Cl dependence to file
        fancy_plot = 0 # plot settings
        show       = 0 # plot settings
        save       = 1 # plot settings
        svg_or_png = 1 # plot settings
        NPROCESSES = [1, 128][cluster]
        pop_size   = [50, 1000][cluster]#(10 if int(version)<1000 else 1)
        checkpoint = 1000
        autoH      = [0, 1e8][1] # automatically increases rates in this category to given nonzero number
        slowones   = [12, 68] # limit infeasible rates, to 1
        slowlim    = 10000 # upper limit for slower (conformation) transitions
        fastlim    = 5e9 # upper limit for faster (ligand assocating) transitions
        chargelim  = 1 # limit to charge movement/electrogeneity
        p_open     = 0.24 # open probability at pH 5, 140 mM Cl-, 160 mV
        opentime   = 90 # microseconds

        "foreground settings"
        pkaweight        = 1e4 # weight multipliers
        depweight        = [9e11, 9e11, 5e9] # Cl & pH dependency
        openchanceweight = 1e11
        opentimeweight   = 1e6
        slowsigs         = [1, .75, .66, .5, .33, .25, .1, .05, .005, .001][:]; sigroller=0 # factor by which sigma is reduced

        if len(sys.argv)>1:
            if sys.argv[1].isdigit():
                version = sys.argv[1]
                del sys.argv[1]
            if len(sys.argv)>1:
                args = dict(arg.split('=', 1) for arg in sys.argv[1:]) # allows console kwargs to override variables
                for key, value in args.items():
                    if key in globals():
                        globals()[key] = int(value) if value.isdigit() else value # update variables received through argv
                        print(f'passed with argument {key}={value}')

        start0=[1000, 1000, 0, 0.5]*20 # parameter set with initial guesses: rates=1000, z=0, d=0.5
        for i in slowones:
            start0[i]=1 # unprotonated channel opening is restricted

        try:
            with open(f'{version}{savefile}.txt', "r") as out:
                read_in=out.readlines()[-1].replace("\n", "")#[-1]
                start=list(map(float, (read_in.split("[")[1].split("]")[0]).split(', ')))
                print("Resuming from file ID", version)
        except FileNotFoundError:
            print(f'No file found with ID {version}, starting from parameters in start0.')
            start = start0

        "optimized parameter sets, can be used to overwrite starting points"
        WTstart    = [226139525.62263933, 925780.249145457, 0.5133913141823201, 0.9921991511626496, 136.54179039313146, 63.84792197287469, -0.9861683284660898, 0.9736273980777502, 4951011730.633228, 7185.0459126956, -0.1449593399086245, 0.012151181678638729, 0.9996440946480286, 1318.6283864610878, -0.32781767437514514, 0.19117106581908963, 861705023.1694026, 748.1558697044702, -0.8100750161484201, 0.13725394712497024, 2122.295930045119, 4461.012496855736, 0.704105961792246, 0.7683633030565026, 797493661.8029584, 3112.4832854383453, 0.8801992741099157, 0.4811116498672727, 4975591822.150006, 62872793.94770426, -0.4132960398381479, 0.011926155468348179, 1514077107.0915139, 58519.8485061709, 0.9119458456442145, 0.7206481889344445, 13794.286403879043, 1.6456634335815365, -0.014741508376253426, 0.03474673511204136, 88817148.21303245, 7205.112027333352, 0.029736828546152307, 0.6099346512933612, 4870065875.627025, 2875.2157019382375, -0.7655966792260144, 0.7471746964573478, 3687868688.482063, 2544018231.095263, -0.12256533800764789, 0.5011749158712182, 4832752760.736315, 35410.00231413007, 0.039387352999829726, 0.9989580958019267, 330614907.0470687, 1151497467.0613759, 0.061781354900806336, 0.09170262844817567, 168972278.78573352, 5.934425945490309, 0.30753334970029467, 0.9646292869195627, 4174344260.976915, 130831488.99877608, -0.5108845695088147, 0.014383760531795674, 0.9996796592585547, 0.0011679505488329668, -0.11300999295011482, 0.3306069290861617, 411.51106730721636, 0.09114257549870786, -0.9855684855944997, 0.9268797586483364, 41.60096099618438, 0.0005481138774988212, 0.08756154333180932, 0.9990677929511343]
        H120Astart = [731659608.2265503, 78794.24444886671, 0.16861571300966693, 0.5008669543131274, 52.87017471052541, 108.07226632071175, -0.46469628154410364, 0.1494165830862079, 2950531117.327667, 3496.0917643936027, -0.22664220488928782, 0.8712400986796094, 0.9361177678672407, 173.91490061852565, -0.06943836364514887, 0.9342500547513526, 1289356838.09933, 118.41395990869967, 0.008597086606130622, 0.1624249521945948, 36.149566803676606, 4209.367225223248, 0.4850149966221655, 0.6315018571119881, 238898003.43223193, 1249.8344797813797, 0.9583083647723998, 0.9993221652221386, 2984556408.9943285, 3384914991.2298393, -0.5598251922647092, 0.3190290358455314, 4988015145.462323, 658638.2444194036, 0.3559872102860765, 0.6793281574212604, 17386.904235693353, 24178.202065592657, -0.3724536949882996, 0.4417646477218748, 662162359.086542, 4899229840.896017, -0.6507253373194698, 0.33616654580908234, 4980839736.321778, 2433.850673630862, -0.26967455572503957, 0.8423305486851067, 291228424.4813745, 720887968.7338889, -0.2905779971498819, 0.6532995057437077, 126977165.88902472, 2589.227861792704, 0.7150934731845375, 0.49649030326056254, 107095436.90473305, 4562123121.030809, 0.6511576809239434, 0.47703139091837343, 469952305.8027879, 12.016626391821678, -0.0967394427273518, 0.8788107701520891, 4777.972346367912, 994.7832387918091, -0.4038901265758081, 0.014383760531795674, 0.11750717319621688, 0.8796677718647032, -0.829530947745123, 0.3306069290861617, 840.3924381352509, 971.5427140121601, -0.47042468484666194, 0.2273503494215105, 17.323650892776094, 1.0479886990006988, -0.2974895718489742, 0.9990677929511343]
        WT_sim_fit = [760677582.5302367, 96058.17772864964, 0.5672517318101049, 0.5, 254.3091113811927, 1335.1687600890727, -0.7969912958959019, 0.021106547445597534, 4895507155.780722, 2332.915556648012, 0.261414789999055, 0.5, 0.6016071258016908, 836.9887297579008, -0.49115435408485203, 0.6344843441009665, 147371667.12345457, 8504.880254730873, -0.9475163366079169, 0.04055736115511417, 1909.2449424681345, 1426.8305578415852, 0.061057733818045454, 0.6950968982027994, 121733268.28009559, 1000.0, -0.08946730689396962, 0.0036037934661067, 516431581.47858655, 2151765510.1340384, 0.05599263503451402, 0.5, 4484312439.174452, 135908.7570825637, 0.27601230910786145, 0.5546679730157009, 1000.0, 1000.0, -0.23524678766772944, 0.5, 1000.0, 1000.0, -0.2860516911798743, 0.5, 3782018873.165045, 218261.88347625514, -0.9983212401200617, 0.6477999221147986, 1126160615.9738755, 893222584.6223168, 0.5630801034365419, 0.5, 1857231729.2779198, 1000.0, -0.4663640427965329, 0.5, 6270316.243731831, 5619289.166175489, -0.16469872935904595, 0.5, 2474348905.1411963, 622811.8314965542, -0.23576138124350113, 0.5, 116631695.01716824, 3202681276.568205, -0.31099280370857746, 0.7557187881028276, 0.8943201713461358, 7.098997937764758, -0.21877095483603914, 0.7205968302605726, 9089.323887176957, 1281.793106071942, -0.9611473067404335, 0.5856975398778355, 8532.756548879603, 5248.285056720285, -0.19858744786387283, 0.4161532259335085]
        # start = [WTstart, H120Astart, WT_sim_fit][2]

        noVdep=sig0=None; model, Cldep, Hdep, sig0, noVdep, flux, closingstates=modelselect(start, model)
        # sig0+=list(range(16)) # place to add extra variables that need to be kept the same

        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        A="\n".join(["0\tx" for _ in range(11)])
        A="""
        23438760.735646706	WTintCl140Cl_pH55leaksubtract
        2732377.3847712763	WTintCl140Cl_pH55Vdeact
        1276231.3700885102	WTintCl_180Cl_pH50
        20830445.393335186	WTintCl_180Cl_pH65
        2479879.715790063	WTintClpH5_40ClApp
        22589232.13582605	WTintClpH5_140ClApp
        608060.9929432273	WTintCl140Cl_pH55App
        7297929.622900769	WTintCl0Cl_pH5App
        233930.43700620532	WTintCl0Cl_pHdep_50
        743262.5443987548	WTintCl0Cl_pHdep_55*
        24599309.39466366	WTintCl_0Cl_pH55Vdeact_short
        """
        B=[x for x in A.rstrip().split("\n") if x]#[x for x in A.split("\n") if x] # RSS value for each experiment, capped at *worsenfactor if >0
        #B=[B[x] for x in [0,1,5,6,7,8,10]]
        errs=[eval(er.split("\t")[0]) for er in B]
        worsenfactor=1.3*0

        def evaluate(start, mode=0, model="Cl", reference=[np.inf]*19):#, datasets, autoH, chargelim, slowlim, fastlim, limsmin, limsmax, slowones, reference=[np.inf]*19):
            fresh_errs=[[] for _ in range(5)];
            start=startcalc(start, model)
            err=0; weirdpeaks=1
            startcheck=np.asarray(start)
            if min(np.concatenate([startcheck-limsmin, limsmax-startcheck]))<0:
                if mode in [0,4]: return np.inf,
                else: print("START VALUE OUT OF BOUNDS:",list(np.where(np.concatenate([startcheck-limsmin,limsmax-startcheck])<0)[0]%len(start)))

            for n,dataset in enumerate(datasets):
                experiment,conds0,conds1,Vs,tsteps,freq,normrange=dataset

                if save_sim: # 1/3
                    with open("write_sim.py", "a") as out:
                        out.write(experiment+" = [\n"); np.set_printoptions(threshold=np.inf)

                data=eval(experiment)
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
                    if mode==1:
                        if V==0:
                            fancyconversion=-100
                            if fancy_plot==0:
                                fig, b = plt.subplots(2-1,1)
                                plt.title(experiment)
                            if fancy_plot==1:
                                fig, (a,b) = toplegend2(data, Vs, [0]+[tsteps[0]]+tsteps[2:], conds0=conds0, conds1=conds1, plotsize=[6, 5], lw=2-len(Vs[-1])*.25/2, experiment=experiment)
                            for realsweep in data[:len(iterable_Vs)]:
                                if experiment=="WTintCl_180Cl_pH65" or fancy_plot==1:
                                    for i in range(len(realsweep))[:tsteps[1]]:
                                        if realsweep[i]>.8:
                                            realsweep[i]=.8
                                b.plot(t*1e3, np.array(realsweep[:tsteps[-1]])*fancyconversion, color="k", alpha=0.75)

                        if "deact" not in experiment and len(sweep)<2: b.plot(t[tsteps[0]:tsteps[2]]*1e3,sweep[0]*fancyconversion,color="b")
                        for segment in range(1,int(len(tsteps)/2)-1): # segments in tsteps after the first
                            b.plot(t[tsteps[2*segment+1]:tsteps[2*segment+2]]*1e3, sweep[segment]*fancyconversion, color="b")
                        if V==range(len(Vs[-1]))[-1]:
                            plt.ylim(1.3*fancyconversion,2*-.1*fancyconversion)
                            if fancy_plot==1: add_scalebar(b ,xunit=" ms",yunit=" %", loc=3)##################
                            if save: plt.savefig(experiment+[".svg",".png"][svg_or_png], bbox_inches='tight')
                            if show: plt.show()
                            if not show: plt.close()

                        if save_sim: # 2/3
                            with open("write_sim.py", "a") as out:
                                out.write(str(list(np.concatenate(sweep*fancyconversion)))+",\n")
                                if V==range(len(Vs[-1]))[-1]:
                                    out.write("]\n\n")

                    if mode==2 and V==0:
                        plotstates2([sim0,sim1,sim2], freq, states, tsteps, experiment, svg_or_png, label=1)
                    if mode==3 and V==0:
                        fluxstates2([step0,step1,step0], [sim0,sim1,sim2], freq, states, tsteps, flux, experiment, svg_or_png, label=1, full=1)

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

                if mode==1: print(f"{err-errprint}\t{experiment}{['*' if err-errprint>errs[n]*worsenfactor else ''][0]}")
                if mode>=4: fresh_errs[0].append(err-errprint)
                if worsenfactor>0 and mode==0 and err-errprint>errs[n]*worsenfactor:
                    return np.inf,

            if weirdpeaks:
                errprint=err
                err=err+1e11*(weirdpeaks-1)**2
                if mode!=0: print(err-errprint,"\t","weirdpeaks")

            errprint=err; plotcount=0
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
                if mode==1:
                    if plotcount==0: plt.figure(figsize=[5,4])
                    plt.xlabel("[Cl$\mathregular{^{-}}$]")
                    if len(pHs)>1:
                        x=10**-np.array(pHs); plt.xscale("log"); plt.xlabel("[H$\mathregular{^{+}}$]")
                    plt.errorbar(x,data,yerr=CIs,fmt="k"+shape[reshaper],label=", ".join([[f"pH {pH}",f"{Cl} mM Cl",f"{V} mV"][X] for X in range(3) if X!=deptype]))#plt.plot(x,data,"k"+shape[reshaper],label=", ".join([[f"pH {pH}",f"{Cl} mM Cl",f"{V} mV"][X] for X in range(3) if X!=deptype]))
                    plt.plot(x,ys,"b-",label="simulation"*(0+1*(deptype!=deptypes[i+1])), zorder=1000)
                    if deptype!=deptypes[i+1]:
                        plt.ylim(0, 1.05); ax = plt.gca(); ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none'); plt.legend()
                        plt.ylabel("Normalised current")
                        if save: plt.savefig(f"{protein} measure vs simulated {['pHs','Cls','Vs'][deptype]} dependency{['.svg','.png'][svg_or_png]}",bbox_inches="tight")
                        if show: plt.show()
                        if not show: plt.close()
                        plotcount=0
                    else: plotcount=1

                    if save_sim: # 3/3
                        with open("write_sim.py", "a") as out:
                            out.write(key+" = "+str(list(ys))+"\n\n")

                if mode==4: fresh_errs[1].append(err)
                if mode==5: fresh_errs[1].append(ys0)
            if mode==1: print(err-errprint,"\t direct pH/Cl dep")

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
                if mode==1: print(err-errprint, f"\t openchance {['0Cl','Cl'][o]}: ",openchance)
                if mode==4: fresh_errs[2].append(err-errprint)

            for ti,time in enumerate([tclose_noCl_160, tclose_Cl_160]):
                errprint=err
                if not 79>time>78:
                    err+=opentimeweight*(time-opentime)**2
                if mode==1: print(err-errprint,f"\t {['0 Cl','140 mM Cl'][ti]} opentimes",time)
                if mode==4: fresh_errs[3].append(err-errprint)

            errprint=err
            pKas=[]
            for i in Hdep:
                protonation,deprotonation=start[i],start[i+1]
                pka= -np.log10(deprotonation/protonation)
                pKas.append(round(pka,1))
                if not 9 >= pka >= 4:
                    err=err+pkaweight*(pka-6)**2#return np.inf,
            if mode==1:
                print(err-errprint, "\t pKa",",".join([variables[j] for j in Hdep]),"=",pKas)
                print(err,"\t TOTAL ERR")
            if mode==4:
                fresh_errs[4]=err-errprint
                fresh_errs=[*fresh_errs[0], *fresh_errs[1], sum(fresh_errs[2]), sum(fresh_errs[3]), fresh_errs[4]]
                if reference[-1]==np.inf:
                    return err,fresh_errs
                else:
                    if any(x2 > x1 for x1,x2 in zip(reference, fresh_errs)): return np.inf,

            if mode==5: fresh_errs.append([[openchance_0Cl, openchance_Cl], [tclose_noCl_160, tclose_Cl_160], pKas])#weirdpeaks
            if err!=err:return np.inf,
            if err<0:return np.inf,
            return [(err,), (err,fresh_errs)][mode>4]
        #XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        cl=1
        if cl==0:
            datasets=datasets[10:]

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
        CXPB, MUTPB, NGEN = 0.7, 0.5, 1000000 # crossover rate, mutation rate, generations

        running_updates=0#+1
        if mode!=0:
            print("limits=", lims, "\nvariables=", variables, len(states), "states, model=", model, ", len(start)=", len(start), "\nstart=", start)
            evaluate(start, mode=mode)
        else:
            start_time = datetime.now()
            olderfitness=evaluate(start, mode)
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
                    with open(f'{version}{savefile}.txt', writetype) as out:
                        if g==checkpoints[0]:
                            out.write(f'generation {0}, x={0}, fitness={olderfitness[0]}: {start}\n')
                        out.write(f'generation {g}, x={x}, fitness={bestfitness}: {fittest}\n')
                    if running_updates:
                        print(f'Save {g}, {datetime.now().strftime("%H:%M:%S")}, duration: {str(datetime.now() - start_time).split(".")[0]}, {bestfitness}.')
                    start_time = datetime.now()
                    olderfitness=newfitness
                sigroller+=1
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        if g and mode==0:
            with open(f'{version}{savefile}.txt', "a") as out:
                out.write(f'manual stop, generation {g}, x={x}, fitness={bestfitness}: {fittest}\n')
                print(f'Saved on script termination: gen {g}, {bestfitness}, fail/dupe {x}: {fittest}.')