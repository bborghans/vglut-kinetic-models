## Parameter optimization
We provide the data and the script used to optimize the transition rates of the conductive mechanism. 
### VGLUT1 chloride channel function
**Cl genetic algorithm/Cl.py**

Performs parameter optimization and result visualization for a kinetic model describing VGLUT1 chloride channel function, for either the wild-type (`WT`) or the `H120A` mutant version of the protein.
The script optimizes model parameters for the specified protein version using residual sum of squares. Results are saved using a version ID (e.g., `1234`), producing files like `1234Cl_sym_output.txt`, which is provided containing an optimized parameter set for the WT. Parameters corresponding to the best fit are saved whenever the generation number reaches a defined checkpoint (default: 1000).
If the script is restarted, it will overwrite the output file at the first checkpoint, using the most recent parameter set from the previous run.
#### Inputs
- Time course data are loaded from: `Cl_WT_measurements.py` and `Cl_H120A_measurements.py`
    The following experimental conditions are used for the data recording:
    ```
    WTintCl140Cl_pH55leaksubtract
    WTintCl140Cl_pH55Vdeactu
    WTintCl_180Cl_pH50
    WTintCl_180Cl_pH65
    WTintClpH5_40ClApp
    WTintClpH5_140ClApp
    WTintCl140Cl_pH55App
    WTintCl0Cl_pH5Appu
    WTintCl0Cl_pHdep_55
    WTintCl0Cl_pHdep_50
    WTintCl_0Cl_pH55Vdeact_short
    ```
- The model, time course conditions, and optimization functions are defined in: `Cl_model.py`
- Parameter sets are loaded from files named as `####Cl_sym_output.txt` (e.g., `1234Cl_sym_output.txt`)
  If is not provided, a built-in default set with uniform intermediate values is used. 

#### Example Usage
```bash
python Cl.py 1234 protein="WT"
```
The first argument defines a unique naming for the output file defined as `####Cl_sym_output.txt` for the output file, updated every  and `protein` refers to the model used: either **WT** or **H120A**.
