## Parameter optimization
We provide the data and the script used to optimize the transition rates of the conductive mechanism. 
### VGLUT1 chloride channel function
**Cl_genetic_algorithm/Cl.py**

Performs parameter optimization for a kinetic model describing VGLUT1 chloride channel function, for either the wild-type (`WT`) or the `H120A` mutant version of the protein.
The script optimizes model parameters for the specified protein version using residual sum of squares. 

#### Inputs
* Time course data are loaded from: `Cl_WT_measurements.pkl` and `Cl_H120A_measurements.pkl`
    The following experimental conditions were used for the recordings of the wild-type (WT) construct:
    ```
    The naming of each experimental condition is encoded as follows (refer to the manuscript for details):
    Construct, conducted/internal anion, external [Cl], external pH, App (for application) when this condition activates current mid voltage pulse, deact (for deactivation) if voltage jumps reduce current after a prepulse, leak-subtracted version if recorded using a P/4 protocol
    
    WT Cl- dataset name:
    
    WTintCl140Cl_pH55leaksubtract
    WTintCl140Cl_pH55Vdeact
    WTintCl_180Cl_pH50
    WTintCl_180Cl_pH65
    WTintClpH5_40ClApp
    WTintClpH5_140ClApp
    WTintCl140Cl_pH55App
    WTintCl0Cl_pH5App
    WTintCl0Cl_pHdep_50
    WTintCl0Cl_pHdep_55
    WTintCl_0Cl_pH55Vdeact_short

    H120A Cl- dataset name:

    H120AintCl140Cl_pHdep_55
    H120AintCl140Cl_pH55Vdeact
    H120AintCl140ClpH50
    H120AintCl140ClpH60
    H120AintClpH5_140ClApp
    H120AintCl140Cl_pH55App
    H120AintCl0Cl_pH5App
    H120AintCl_0Cl_pH55leaksubtract
    H120AintCl_0Cl_pH55Vdeact
    ```
* The model, time course conditions, and optimization functions are defined in: `Cl_model.py`
* Parameter sets are loaded from files named as `####Cl_sym_output.txt` (e.g., `1234Cl_sym_output.txt`)
  If is not provided, a built-in default set with uniform intermediate values is used. 
* The output is `####Cl_sym_output.txt` and contains: 
    - 1 generation number
    - 2 number of iterations
    - 3 weighted error
    - 4 optimized parameters
#### Code usage
[Cl.py](Cl_genetic_algorithm/Cl.py) optimizes starting transition rates given the provided experimental data
```
  usage: Cl.py [-h] [-protein {WT,H120A}] [-name NAME] [-id ID] [-nprocesses NPROCESSES] [-pop_size POP_SIZE]
             [-ngen NGEN] [-cxpb CXPB] [-mutpb MUTPB]

options:
  -h, --help            show this help message and exit
  -protein {WT,H120A}   Output file name (default: WT)
  -name NAME            Output file name (default: Cl_sym_output)
  -id ID                Output ID (default: 0)
  -nprocesses NPROCESSES
                        Number of parallel processes to run (default: 1)
  -pop_size POP_SIZE    Population of each generation (default: 50)
  -ngen NGEN            Number of generations (default: 1000)
  -cxpb CXPB            Crossover rate (default: 0.7)
  -mutpb MUTPB          Mutation rate (default: 0.5)
```

**GA_genetic_algorithm/GA.py**

Performs parameter optimization and limited result visualization for a kinetic model describing active transport by VGLUT1, for glutamate and aspartate substrate in parallel.
The script optimizes model parameters for the specified protein version using residual sum of squares. 


#### Inputs
- Time course data are loaded from: `GlutWT_measurements.py` and `AspWT_measurements.py`
    The following experimental conditions were used for the recordings of the wild-type (WT) construct:
    ```
    The naming of each experimental condition is encoded as follows (refer to the manuscript for details):
    Construct, conducted/internal anion, external [Cl], external pH, App (for application) when this condition activates current mid voltage pulse
    
    WT Glut dataset name:

    WTintGlut40Cl_pH55
    WTintGlut40Cl_pH5
    WTintGlut40Cl_pH5App
    WTintGlutpH5_40ClApp
    WTintGlutpH55_140ClApp
    WTintGlutpH55_140ClApp2

    WT Asp dataset name:

    WTintAsp40Cl_pH5
    WTintAsp40Cl_pH5App
    WTintAsppH55_40ClApp
    ```
- The model, time course conditions, and optimization functions are defined in: `GA_model.py`
- Weighting factors for time course data and calculated metrics are loaded from: `GA_weights.py`
- Parameter sets are loaded from output files (e.g., `0000GA_sym_output`)
* The output is `####GA_sym_output` is a pickle file and contains:
    - 1 generation number
    - 2 weighted error

If no output file is provided and no parameter set is pasted directly into `GA.py`, a built-in default set with uniform intermediate values is used.
#### Code Usage
[GA.py](GA_genetic_algorithm/GA.py) optimizes starting transition rates given the provided experimental data
```
usage: GA.py [-h] [-name NAME] [-id ID] [-protein {GlutWT,AspWT}] [-nprocesses NPROCESSES] [-pop_size POP_SIZE] [-ngen NGEN] [-cxpb CXPB] [-mutpb MUTPB]
             [-checkpoint CHECKPOINT] [-resume {0,1}]

options:
  -h, --help            show this help message and exit
  -name NAME            Output file name (default: GA_sym_output)
  -id ID                Output ID (default: 0)
  -protein {GlutWT,AspWT}
                        Constructs available (default: GlutWT)
  -nprocesses NPROCESSES
                        Number of parallel processes to run (default: 1)
  -pop_size POP_SIZE    Population of each generation (default: 50)
  -ngen NGEN            Number of generations (default: 1000)
  -cxpb CXPB            Crossover rate (default: 0.7)
  -mutpb MUTPB          Mutation rate (default: 0.5)
  -checkpoint CHECKPOINT
                        Frequency of data saving in number of generations (default: 10)
  -resume {0,1}         Resume from file identified by its name and ID (default: 0)
```
