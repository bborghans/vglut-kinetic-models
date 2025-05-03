## Parameter optimization
We provide the data and the script used to optimize the transition rates of the conductive mechanism. 
### VGLUT1 chloride channel function
**Cl genetic algorithm/Cl.py**

Performs parameter optimization for a kinetic model describing VGLUT1 chloride channel function, for either the wild-type (`WT`) or the `H120A` mutant version of the protein.
The script optimizes model parameters for the specified protein version using residual sum of squares. Results are saved using a version ID (e.g., `1234`), producing files like `1234Cl_sym_output.txt`, which is provided containing an optimized parameter set for the WT. Parameters corresponding to the best fit are saved whenever the generation number reaches a defined checkpoint (default: 1000).
If the script is restarted, it will overwrite the output file at the first checkpoint, using the most recent parameter set from the previous run.
#### Inputs
- Time course data are loaded from: `Cl_WT_measurements.py` and `Cl_H120A_measurements.py`
    The following experimental conditions were used for recordings of the wild-type (WT) construct:
    ```
    Construct, internal [Cl], external [CL], external pH, leak subtracted version
    
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
    WT Cl- dataset name:

    H120A Cl- dataset name

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
- The model, time course conditions, and optimization functions are defined in: `Cl_model.py`
- Parameter sets are loaded from files named as `####Cl_sym_output.txt` (e.g., `1234Cl_sym_output.txt`)
  If is not provided, a built-in default set with uniform intermediate values is used. 
#### Outputs
- `####Cl_sym_output.txt` contains: 
    - 1 generation number
    - 2 number of iterations
    - 3 weighted error
    - 4 optimized parameters

#### Example Usage
```bash
python Cl.py 1234 protein="WT"
```
The first argument defines a unique naming for the output file defined as `####Cl_sym_output.txt` for the output file, updated every  and `protein` refers to the model used: either **WT** or **H120A**.
