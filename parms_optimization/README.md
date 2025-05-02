## Parameter optimization
We provide the data and the script used to optimize the transition rates of the conductive mechanism. 
### VGLUT1 chloride channel function
**Cl genetic algorithm/Cl.py**

Performs parameter optimization and result visualization for a kinetic model describing VGLUT1 chloride channel function, for either the wild-type (`WT`) or the `H120A` mutant version of the protein.
The script optimizes model parameters for the specified protein version using residual sum of squares. Results are saved using a version ID (e.g., `1234`), producing files like `1234Cl_sym_output.txt`, which is provided containing an optimized parameter set for the WT. Parameters corresponding to the best fit are saved whenever the generation number reaches a defined checkpoint (default: 1000).
If the script is restarted, it will overwrite the output file at the first checkpoint, using the most recent parameter set from the previous run.
#### Input Data and Dependencies
- Time course data are loaded from: `Cl_WT_measurements.py` and `Cl_H120A_measurements.py`
- The model, time course conditions, and optimization functions are defined in: `Cl_model.py`
- Parameter sets are loaded from output files (e.g., `1234Cl_sym_output.txt`)

If no output file is provided and no parameter set is pasted directly into `Cl.py`, a built-in default set with uniform intermediate values is used.

#### Example Usage
```bash
python Cl.py 1234 protein="WT"
```
The script accepts additional keyword arguments and supports further customization. See the source code for details.
