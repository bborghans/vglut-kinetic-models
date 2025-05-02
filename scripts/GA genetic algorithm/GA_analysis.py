mode=( # make sure the working directory is set correctly (F5)
0
) # 0: default, 1: prints latest parameters in start
modelfile="GA_model"
from GA_model import modelselect, loaddata, startcalc, initialvalue, simulate, gatingcurrent, stationarycurrent, rainbow2, dicts, paramplot#, plotstates2, fluxstates3, toplegend2, formatstart,
import os; import sys; import matplotlib.pyplot as plt; import inspect; import numpy as np; from math import inf; import pickle as pkl; import warnings; from scipy.linalg import LinAlgWarning#; from scalebars import add_scalebar
np.set_printoptions(legacy='1.25')
if sys.argv==[""]:

    versions=1234 # <-- manual version select outside of console, Ctrl+F6 to switch
    versions0=str(versions)
    versions=([versions] if isinstance(versions, int) else list(versions)); argv_args=[]
else: # console use: python GA_analysis.py versions mode=0
    if len(sys.argv)<2: sys.exit('Run as "python GA_analysis.py <versions>"')
    versions0=sys.argv[1]
    versions=eval(versions0)
    del sys.argv[1]
    if [x for x in versions0 if x not in "0123456789"]:
        check=[x for x in versions0 if x not in "0123456789"][0]
        if check in ":-":
            versions=range(int(versions0.split(check)[0]), int(versions0.split(check)[1])+1)
    else:
        IN=eval(versions0)
        versions=([IN] if isinstance(IN, int) else list(IN))

ALL=0+1
top=1+1+1 # show this many traces. best file numbers, or interspaced saves if ALL (set float to limit weight plots)
START=[ # any parameter list in START below will bypass files and number in versions
#[102064895.48863456, 1000, -0.8420912425372568, 9.543148904192825e-05, 113004633.46036373, 32183699.91917863, -0.997574163116501, 0.0008963699640230907, 9652.088588637578, 6241.51974528119, -0.4652079594969484, 0.7262968794857646, 366884599.9306862, 1000, -0.8887704445309168, 0.9873143996945732, 155894069.78120843, 1000, -0.9595966282656041, 0.13561233937870878, 3162681464.5658307, 786913343.1685503, 0.13651624181608352, 0.26765417171364636, 1491896730.1163766, 1000, -0.21485410436925825, 0.24192355393575873, 2645.5063002823076, 9987.089406918687, -0.8929093953140423, 0.9990480083571331, 175630553.49845877, 1000, -0.9579306734690223, 0.04835058093413512, 1306567894.7973375, 1000, -0.9958532435948368, 0.0021441397562417695, 1000, 28.027232362897692, -0.46741059636790694, 0.9588534113021424, 17040303.65631462, 228627.32297398968, -0.39825036742980147, 0.0008782080198134912, 356494954.3368821, 40087278.80059441, -0.3619937521005686, 0.02460072242108396, 1000, 93342924.80765481, -0.9996510491531282, 0.9993341319675056, 4999153055.119049, 4025042321.137767, 0.8234712928868508, 0.00034886117462298383, 9685.555874369113, 182.97380950809418, 0.8862955784414268, 0.5111603911644096, 2335683656.396435, 17811444.50635969, -0.8815534962660676, 0.9787901076882094, 149505290.67081445, 3861396504.331461, 0.6468276996323286, 0.009205316140234234, 17197187.124566883, 4323843376.439306, -0.4536610019635566, 0.9928045821716254, 4505465979.704372, 147564644.51533931, 0.23023867272812293, 0.07106600233869108, 1065.9657362530688, 443.00278762990774, -0.2847324545447545, 0.5268044794400398, 4996067781.548299, 1115098.2220789492, -0.9975315843091471, 0.6482441895276939, 4785414610.854129, 902625055.8910549, 0.9267456817684613, 0.7960574029527977, 5751292.771176295, 40220.36292668858, -0.0715145261886947, 0.9996484341981745, 4996191774.766897, 1022631.7926665772, -0.1874926142317742, 0.9979424158644664, 0.8787959126989459, 0.00015724017202038714, 0.18127078928850837, 0.2108699454483538, 0.6020139238414184, 0.0017049511896232617, -0.5125027115820244, 0.20784167359905822, 555654.4550954821, 118502.15623956855, -0.7967083994693208, 0.8616292276998586, 2852994849.912186, 4996781982.163919, 0.2545827933581347, 0.988491246289296, 1174.6252314309659, 8457.544463657725, 0.18243159073725357, 0.2768241733453882, 1591262468.2821636, 4463677384.005713, -0.6090529613156437, 0.31700260852568507, 781249476.3816221, 2971314835.475611, 0.9872953960911027, 0.9775339007473219, 37413453.21965375, 1000, -0.5055930024454657, 0.49715630126385085, 163426149.04004154, 1000, 0.45016587441866784, 0.9699380106851664, 52.2265337206856, 139.31757968992122, -0.7560326980236898, 0.7051325577299081, 237100863.13996512, 1000, -0.4517369950514832, 0.014513963128273504, 76816550.68990777, 1000, 0.729380825374736, 0.5076462718649198, 1000, 63.80439477396719, -0.8826411333484341, 0.060327602040053746, 93360575.17108685, 8.95632817758329, -0.7253251670842736, 0.06021941985244997, 858.0671524224196, 9895.273597789039, 0.31879583606347506, 0.05387831798865382, 4840.100126440418, 8899.333384033114, -0.6579355674938406, 0.7766645128222525]
]
if START: versions0="START"; versions=VERSIONS=[versions0]

show=0#+1
save=0+1
no_Asp=0#+1
recalc=0
autoH=[0, 1e8][0] # automatically increases rates in this category to given nonzero number
fancy=[1, 2][0]; svg_or_png=1 # plot settings
if len(versions)>1 or mode not in [0,1]: ALL=0
plt.rc('font', **{'family' : 'Arial', 'size' : 12})
warnings.filterwarnings(action='ignore', category=LinAlgWarning)

filename="GA_sym_output"
moredata="outtext"
savedir=""

slowones=[] # limit infeasible rates, to 1
slowsig=1 # factor by which sigma is reduced
slowlim=10000 # upper limit for slower (conformation) transitions
fastlim=5e9 # upper limit for faster (ligand assocating) transitions
chargelim=1 # limit to charge movement/electrogeneity
transportrate=[-561*2, (-561*2)*2.3]

if len(sys.argv)>1:
    args = dict(arg.split('=', 1) for arg in sys.argv[1:]) # allows console kwargs to override variables
    for key, value in args.items():
        if key in globals():
            globals()[key] = int(value) if value.isdigit() else value # update variables received through argv
            print(f'passed with argument {key}={value}')

for (protein,model) in [["GlutWT"]*2, ["AspWT"]*2]:
    exec(f'from {protein}_measurements import *')

def evaluate(START, mode=0, params=0, name=""): # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    figs={"GlutWT":{}, "AspWT":{}}
    for LIG,(protein,model) in enumerate([["GlutWT"]*2, ["AspWT"]*2][:(1 if no_Asp else None)]): #(1 if "1" in MODEL else None)###########################################################
        datasets, deps, start, model, Hdep, Cldep, Liganddep, sig0, noVdep, connections, closingstates, states,\
            variables, limsmin, limsmax, errs, worsenfactor, transitionmatrix=loadin(START, protein, model, LIG)
        datasets=[x for x in datasets if "_leaksubtract" not in x[0]]##########################################################
        if params and name:
            basevariables="123456789"+"abcdefghijklmnopqrstuvwxyz"+"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if mode==0:
                paramplot(start, model, Cldep, Hdep, Liganddep, save=0, store=1, title=f'{name}, {protein}', variables=basevariables[:len(start)//4])
                plt.savefig(f'{folder}{savedir}{versions0}, top {top}, {protein} paramplot.png', bbox_inches='tight')

        startcheck=np.asarray(start)
        if min(np.concatenate([startcheck-limsmin,limsmax-startcheck]))<0:
            if mode==0: return inf,
            else: print("START VALUE OUT OF BOUNDS:",list(np.where(np.concatenate([startcheck-limsmin,limsmax-startcheck])<0)[0]%len(start)))

        for n,dataset in enumerate(datasets):
            experiment,conds0,conds1,Vs,tsteps,freq,normrange=dataset
            data=eval(experiment+(clapp if "ClApp" in experiment else ""))##################################################
            figs[experiment]=[]
            iterable_Vs=Vs[-1] # selects list of variable Vs when there's also a stationary segment

            length=tsteps[-1]
            t=np.arange(length)/freq # total timespan

            for V in range(len(Vs[-1])): # =sweeps
                Imax=np.mean(data[V][normrange[0]:normrange[1]])

                sweep=np.asarray([0]*int((len(tsteps)-2)/2),dtype=object)
                step_0=transitionmatrix(*start,*conds0) # pre-V starting point

                step_groundstate=initialvalue(states,step_0)
                step0=transitionmatrix(*start,*conds0[:4],Vs[0][V]) # first condition
                sim0=simulate(t[0:tsteps[2]-tsteps[0]],step0,step_groundstate) # first simulation
                if sim0[-1]["message"]!="Integration successful.":return inf,
                sim0=sim0[0]
                sweep[0]=gatingcurrent(step0,sim0,start,connections)#normalized_anioncurrent(openstates,sim0)
                subnormer=sweep[0][-1]

                if len(tsteps)>4:
                    step1_0=sim0[-1]
                    step1=transitionmatrix(*start,*conds1[:4],Vs[-1][V]) # second condition
                    sim1=simulate(t[0:tsteps[4]-tsteps[3]],step1,step1_0) # second simulation
                    if sim1[-1]["message"]!="Integration successful.":return inf,
                    sim1=sim1[0]
                    sweep[1]=gatingcurrent(step1,sim1,start,connections)#normalized_anioncurrent(openstates,sim1)
                    if "forw" in experiment or ("App" in experiment and "App2" not in experiment):
                        subnormer=[sweep[1][-1]]

                if len(tsteps)>6:
                    step2_0=sim1[-1]
                    sim2=simulate(t[0:tsteps[6]-tsteps[5]],step0,step2_0) # simulated return to first condition
                    if sim2[-1]["message"]!="Integration successful.":return inf,
                    sim2=sim2[0]
                    sweep[2]=gatingcurrent(step0,sim2,start,connections)

                sweep*=(Imax/subnormer)

                sim=np.concatenate(sweep)
                figs[experiment].append([np.arange(len(sim))/freq, sim, data[:len(iterable_Vs)][V][tsteps[0]:tsteps[0]+len(sim)]])

        for i,key in enumerate(deps):
            dep=deps[key]
            Cls=dep["Cls"]; pHs=dep["pHs"]
            Vs=dep["Vs"]
            data=dep["data"]
            ys0=[]
            for Cl in Cls:
                for pH in pHs:
                    for V in Vs:
                        A=transitionmatrix(*start, pH, 7.4, Cl, .140, V)
                        AW=initialvalue(states,A)
                        ys0.append(stationarycurrent(A,AW,start, connections))#(normalized_anioncurrent(openstates,AW))
            ys0=np.asarray(ys0)
            ys=ys0/ys0[data.index(1)]

            figs[protein][["Cl","pH"][i]]=[[Cls,pHs][i], ys, data]

        #                           pH,  M Cl,  mV
        for i,params in enumerate([[5.5, .04, -.16]]):
            ph,cl,v = params
            A=transitionmatrix(*start, ph, 7.4, cl, .14, v)
            AW=initialvalue(states,A)
            transport = stationarycurrent(A,AW,start,connections)
        TR=transportrate[LIG]
        if mode==1 and show:
            print(f'\t transport /s (vs {TR}{[" -G$^-$ + +H$^+$", ""][LIG]}):', transport)
        figs[protein]["transport"]=[transport, TR]

        pKas=[]
        for i in Hdep:
            protonation=start[i]
            deprotonation=start[(i//4)*4+[1 if i/2==int(i/2) else 0][0]]
            pka= -np.log10(deprotonation/protonation)
            pKas.append(pka)

        if mode==1 and show:
            print("pKa", ", ".join([variables[j] for j in Hdep]), "\n  ",[round(x,1) for x in pKas])
        figs[protein]["protonation"]=[[variables[j] for j in Hdep], pKas]

    return figs

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

def selectmodel(start):
    MODEL=[int(MODELS[x]["ID"]) for x in MODELS.keys() if len(start)==MODELS[x]["g_len"]+MODELS[x]["a_len"]-MODELS[x]["samelen"]]
    MODEL=(sys.exit("Error: loaded start length does not match available models.") if MODEL==[] else MODEL[0])
    ID, samelen, g_len, a_len, slowones, clapp, start0=[MODELS[MODEL][x] for x in ["ID","samelen","g_len","a_len","slowones","clapp","start0"]]
    return (MODEL, ID, samelen, g_len, a_len, slowones, clapp, start0)

def STARTcalc(START):
    startG=startcalc(START[:g_len], "GlutWT"+ID)
    startA=startcalc(START[:samelen]+START[g_len:], "AspWT"+ID)
    return startG+startA[samelen:]

def loadin(START, protein, model, LIG, slowones=slowones):
    import importlib
    transitionmatrix=getattr(importlib.import_module(modelfile), f'{model+ID}_transitionmatrix')
    datasets, deps=loaddata(protein)

    start=[START[:g_len], START[:samelen]+START[g_len:]][LIG]

    noVdep=sig0=None; model, Hdep, Cldep, Liganddep, sig0, noVdep, connections, closingstates=modelselect(start, model+ID)
    variables=inspect.getfullargspec(transitionmatrix)[0][0:len(start)]
    states=inspect.getfullargspec(transitionmatrix)[3][0]

    slowones=slowones if protein=="GlutWT" else []

    limsmin=[0, 0, -chargelim, 0]*int(len(start)/4)
    limsmax=[slowlim, slowlim, chargelim, 1]*int(len(start)/4)
    for f in Hdep+Cldep+Liganddep: # increased upper limit for (not just small?)  ion binding
        limsmax[f//4*4]=limsmax[f//4*4+1]=fastlim
        if f in Hdep: # optional autoH-based lower bound for H binding
            limsmin[f]=autoH
    for s in slowones: # rates limited to 1 (due to infeasible opening)
        limsmax[s]=1

    start=startcalc(start, model+ID)#detailedbalance

    errs,worsenfactor=0,None
    return datasets, deps, start, model, Hdep, Cldep, Liganddep, sig0, noVdep, connections,\
        closingstates, states, variables, limsmin, limsmax, errs, worsenfactor, transitionmatrix

#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

folder=""
if not START:
    testfile=str(versions[0]).zfill(4)
    if testfile+filename not in os.listdir(): # looks for (first) file in root, else assumes data is in <folder>
        sys.exit(f'{testfile+filename} not in workdir.')

FIGS={}; GEN_ERRS={}; Xs=[]; Ys=[]; check=""
n=0; gen_size=1
for version in versions:
    VERSION=str(version).zfill(4)
    MODELS={
    0:{"ID":"0", "samelen":52, "g_len":108, "a_len":108, "slowones":[100, 101, 104, 105], "clapp":"",              "start0":[1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1, 1, 0, 0.5, 1, 1, 0, 0.5]+[1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 0, 0.5, 1000, 1000, 0, 0.5]},#164
    1:{"ID":"1", "samelen":48, "g_len":88,  "a_len":76,  "slowones":[],                   "clapp":"",              "start0":[1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 1, 0.5]+[1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 0, 0.5, 1000, 1000, 1, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 0, 0.5, 1000, 1000, 1, 0.5]},#116
    2:{"ID":"2", "samelen":24, "g_len":60,  "a_len":60,  "slowones":[48, 49],             "clapp":"_leaksubtract", "start0":[1000, 1000, 0.9, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, -1, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, -0.9, 0.5, 1000, 1000, -0.9, 0.5, 1000, 1000, -0.9, 0.5, 1000, 1000, 0.6, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, -0.9, 0.5, 1000, 1000, -0.9, 0.5, 1, 1, -0.2, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, 0.9, 0.5]+[1000, 1000, -0.9, 0.5, 1000, 1000, 0.6, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, -0.9, 0.5, 1000, 1000, -0.9, 0.5, 1000, 1000, -0.2, 0.5, 1000, 1000, 0.9, 0.5, 1000, 1000, 0.9, 0.5]},#96
    }
    try:
        if START:
            gen_errs=[]; ALL=0
            start=START
            MODEL, ID, samelen, g_len, a_len, slowones, clapp, start0=selectmodel(start)
            FIGS[VERSION]=evaluate(start, mode)
        else:
            with open(folder+VERSION+filename, "rb") as out: # output of generations vs err
                gen_errs=pkl.load(out)
            start=gen_errs[-1][-1]
            MODEL, ID, samelen, g_len, a_len, slowones, clapp, start0=selectmodel(start)


        print(VERSION, "model", MODEL, "no_Asp"*no_Asp)
        if no_Asp: start[g_len:]=start0[g_len:]####################################################################################################################################

        if (not ALL or len(versions)>1) and not START:
            GEN_ERRS[VERSION]={"gen":[], "err":[], "start":[]}
            for GEN in gen_errs:
                (gen,err),weights,fittest=GEN
                GEN_ERRS[VERSION]["gen"].append(gen)
                GEN_ERRS[VERSION]["err"].append(err)
            GEN_ERRS[VERSION]["start"].append(fittest) # takes last start
            FIGS[VERSION]=evaluate(fittest, mode)
            check=""

        #######################################################################
        if ALL and len(versions)==1 and not START:
            inds=np.linspace(0, len(gen_errs)-1, min(len(gen_errs), int(top)))
            inds=[int(x) for x in inds]
            for G,GEN in enumerate(gen_errs):
                (gen,err),weights,fittest=GEN
                if G==1:
                    gen_size=gen
                if G in inds:
                    VG="."+str(G).zfill(4)
                    GEN_ERRS[VERSION+VG]={"gen":[], "err":[], "start":[]}
                    GEN_ERRS[VERSION+VG]["gen"].append(gen)
                    GEN_ERRS[VERSION+VG]["err"].append(err)
                    GEN_ERRS[VERSION+VG]["start"].append(fittest)
                    FIGS[VERSION+VG]=evaluate(fittest, mode)
            check=VG

            WTS=[] # representation 1
            VERSION=str(version).zfill(4)
            with open(folder+VERSION+filename, "rb") as out:
                gen_errs=pkl.load(out)

            if mode==0:
                for i in range(len(gen_errs)):
                    if i in inds or isinstance(top, int):
                        (gen,err),weights,fittest=gen_errs[i]
                        wts=[weights[key1][key2] for key1 in weights for key2 in weights[key1] if weights[key1][key2]!=0]
                        WTS.append(wts)
                        if i==0:
                            nms=[f'{key1} {key2}' for key1 in weights for key2 in weights[key1] if weights[key1][key2]!=0]
                        plt.plot(nms, wts, c=rainbow2(i, len(gen_errs)), label=(i if i in inds else None))
                plt.xticks(rotation=90)
                plt.yscale("log")
                plt.legend()
                plt.title(VERSION)
                plt.ylabel("Weight")
                if save: plt.savefig(f'{folder}{savedir}{versions0}, weights by comparison.png', bbox_inches='tight')
                if show: plt.show()
                if not show: plt.close()

                wts=np.array(WTS).transpose() # representation 2
                for i in range(len(nms)):#
                    plt.plot((inds if isinstance(top, float) else range(len(gen_errs))), wts[i], c=rainbow2(i, len(nms)), label=nms[i])#, label=(i if i in [0, len(gen_errs)-1] else None)
                plt.xticks(rotation=90)
                plt.yscale("log")# plt.ylim(0,None)
                plt.legend(fontsize="6", loc="upper left")
                plt.title(VERSION)
                plt.xlabel(f'Generation saves (x{gen_size})')
                plt.ylabel("Weight")
                if save: plt.savefig(f'{folder}{savedir}{versions0}, weights by generation.png', bbox_inches='tight')
                if show: plt.show()
                if not show: plt.close()
            #######################################################################

        if np.inf in FIGS[VERSION+check]:
            print(f'Warning: {VERSION} cannot be evaluated, has been emitted.')
            versions=[x for x in versions if x!=version]
            del GEN_ERRS[VERSION+check]
            del FIGS[VERSION+check]
            continue

        try:
            with open(folder+VERSION+moredata, "rb") as out: # output of generations vs err
                outtext=pkl.load(out)
            if mode==1: dicts(outtext, full=1)
            X=[]; Y=[]
            for KEY in outtext.keys():
                for key in outtext[KEY]:

                    data=outtext[KEY][key]
                    for entry in data:
                        if entry!="err" and (data[entry]!=0 or key=="pkaweight") and "WTintAsppH5_40ClApp" not in KEY:
                            X.append(f'{KEY} {key}'.replace("_leaksubtract","").replace("leaksubtract","").replace("_long",""))
                            Y.append(float(data[entry]))
            Xs.append(X)
            Ys.append(Y)

        except FileNotFoundError:
            continue
        n+=1
    except FileNotFoundError:
        continue

if not START:
    if not len(FIGS.keys()):
        sys.exit("No valid outputs found.")
    top=int(top)
    if len(GEN_ERRS.keys())<top:
        top=len(GEN_ERRS.keys())
    ranking=sorted(GEN_ERRS.keys(), key=lambda VERSION: min(GEN_ERRS[VERSION]["err"]))#GEN_ERRS[VERSION]["err"][-1]
    VERSIONS=sorted(ranking[:top])
    if ALL and len(versions)==1:
        VERSIONS=GEN_ERRS.keys()
    if mode==1:
        for VERSION in VERSIONS:
            print(VERSION, GEN_ERRS[VERSION]["err"][-1],
              "\n",  GEN_ERRS[VERSION]["start"][0], "\n")
if mode==0:
    N=0
    fig1,ax1=plt.subplots()
    for VERSION in GEN_ERRS.keys():
        if VERSION in VERSIONS:
            if ALL and len(versions)==1:
                ax1.scatter(GEN_ERRS[VERSION]["gen"], GEN_ERRS[VERSION]["err"], color=rainbow2(int(VERSION.split(".")[1]), len(gen_errs)), label=VERSION.split(".")[1])
            ax1.plot(GEN_ERRS[VERSION]["gen"], GEN_ERRS[VERSION]["err"], color=("k" if ALL else rainbow2(N, top)), label=VERSION*(1-ALL), zorder=0)
            N+=1
    ax1.set_yscale("log")
    ax1.set(xlabel="Generations", ylabel="Total error (RSS)", title=filename)
    if len(VERSIONS)<=10: ax1.legend()
    if save: fig1.savefig(f'{folder}{savedir}{versions0}, top {top}, gen vs errs.png', bbox_inches='tight')

    if (not ALL or len(versions)>1) and Xs:
        N=0
        fig2,ax2=plt.subplots()
        [ax2.axvline(x, color="k", lw=.5) for x in [4.5, 9.5, 20.5]]
        for n,VERSION in enumerate(GEN_ERRS.keys()):
            if VERSION in VERSIONS:
                ax2.plot(Xs[n], Ys[n], marker="s", lw=.5, color=rainbow2(N, top), label=VERSION, alpha=1)
                ax2.axhline(np.mean(Ys[n]), ls="-", lw=.5, color=rainbow2(N, top), alpha=1)
                N+=1
        ax2.set(yscale="log", ylabel="rmsd or max RSS")
        plt.tick_params(labelbottom=False)
        if Xs:
            for i in range(len(Xs[0])):
                ax2.text(i, .02, Xs[0][i], color="k", rotation=90, zorder=5, ha="center", transform=ax2.get_xaxis_transform())
        if len(VERSIONS)<=10: ax2.legend()
        if save: fig2.savefig(f'{folder}{savedir}{versions0}, top {top}, rmsd comparison.png', bbox_inches='tight')

    fig4,ax4=plt.subplots()
    for plot in FIGS[list(FIGS.keys())[0]].keys():
        if plot in ["GlutWT", "protonation", "AspWT"][:(-1 if no_Asp else None)]:#(-1 if "1" in MODEL else None)###########################################################
            fig1,ax1=plt.subplots()
            fig2,ax2=plt.subplots()
            fig3,ax3=plt.subplots()

            N=0
            ax4vals=[]
            for n,VERSION in enumerate(FIGS.keys()):
                if VERSION in VERSIONS:
                    x,y,data=FIGS[VERSION][plot]["Cl"]
                    ax1.plot(x, data, color="k")
                    ax1.plot(x, y, color=rainbow2(N, top), label=VERSION)
                    ax1.set(title=plot+" Cl$^-$ dependence", xlabel="[Cl$^-$]", ylabel="Norm. current", alpha=1, ylim=[0, None])

                    x,y,data=FIGS[VERSION][plot]["pH"]
                    ax2.plot(x, data, color="k")
                    ax2.plot(x, y, color=rainbow2(N, top), label=VERSION)
                    ax2.set(title=plot+" pH dependence", xlabel="[pH]", ylabel="Norm. current", alpha=1, ylim=[0, None])

                    x,y=FIGS[VERSION][plot]["protonation"]
                    ax3.plot(x, y, color=rainbow2(N, top), marker="s", label=VERSION, alpha=1)
                    ax3.set(title=plot+" protonation", xlabel="rate", ylabel="pKa")
                    [ax3.axhline(x, color="k") for x in [3+1, 11-1]]

                    y,data=FIGS[VERSION][plot]["transport"]
                    ax4.axhline(-data, color=("b" if plot=="GlutWT" else "y"))
                    ax4.plot(VERSION, -y, color=("b" if plot=="GlutWT" else "y"), marker="s")
                    ax4vals.append(-data)
                    N+=1
            ax4.plot(np.nan, np.nan, marker="s", color=("b" if plot=="GlutWT" else "y"), label=plot)

            axylim=ax4vals+[500, 1500]
            yticks=[round(min(np.floor(axylim))), round(max(np.ceil(axylim)))]+[round(abs(x)) for x in transportrate]
            ax4.set(title="pH 5.5, 40 mM Cl$^-$, -160 mV", ylabel="absolute transport", yscale="log", yticks=yticks, yticklabels=yticks)
            [axs.legend() for axs in [ax1,ax2,ax3,ax4] if len(GEN_ERRS.keys())<=10]
            if save:
                fig1.savefig(f'{folder}{savedir}{versions0}, top {top}, {plot} Cl.png', bbox_inches='tight')
                fig2.savefig(f'{folder}{savedir}{versions0}, top {top}, {plot} pH.png', bbox_inches='tight')
                fig3.savefig(f'{folder}{savedir}{versions0}, top {top}, {plot} protonation.png', bbox_inches='tight')
                fig4.savefig(f'{folder}{savedir}{versions0}, top {top}, transport rates.png', bbox_inches='tight')
            if show: plt.show()
            if not show: plt.close()


        if "WTint" in plot:
            fig5,ax=plt.subplots()
            N=0
            for n,VERSION in enumerate(FIGS.keys()):
                if VERSION in VERSIONS:
                    for V in FIGS[VERSION][plot]:
                        x,y,data=V
                        ax.plot(x, data, color="k", zorder=0)
                        ax.plot(x, y, color=rainbow2(N, top))
                    ax.plot(np.nan, np.nan, color=rainbow2(N, top), label=VERSION, alpha=1)
                    N+=1
            ax.set(title=plot, xlabel="Time (s)", ylabel="Norm. current", ylim=[[None, None], [-1, 2]][1]) # ylim switch
            ax.invert_yaxis()
            if len(VERSIONS)<20:
                plt.legend()
            if save: fig5.savefig(f'{folder}{savedir}{versions0}, top {top}, {plot}.png', bbox_inches='tight')
            if show: plt.show()
            if not show: plt.close()

print(list(VERSIONS))
if not START:
    print([GEN_ERRS[x]["err"][-1] for x in VERSIONS])
    _=[evaluate(GEN_ERRS[x]["start"][-1], mode, name=VERSION, params=1) for x in {X[:4]:X for X in VERSIONS}.values()]
