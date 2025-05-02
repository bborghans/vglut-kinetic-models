### Cl.py README

`Cl.py` performs parameter optimization and result visualization for a kinetic model describing VGLUT1 chloride channel function, for either the wild-type (`WT`) or the `H120A` mutant version of the protein.

#### Mode 0 — Parameter Optimization
When run with `mode=0`, the script optimizes model parameters for the specified protein version using residual sum of squares. Results are saved using a version ID (e.g., `1234`), producing files like `1234Cl_sym_output.txt`, which is provided containing an optimized parameter set for the WT. Parameters corresponding to the best fit are saved whenever the generation number reaches a defined checkpoint (default: 1000).

If the script is restarted, it will overwrite the output file at the first checkpoint, using the most recent parameter set from the previous run.

#### Modes 1–3 — Result Visualization
These modes generate plots based on the latest optimized parameter set:
- `mode=1`: Numerical details and fit plots
- `mode=2`: State distribution plots
- `mode=3`: Flux plots

Additional options:
- `show`: If `true`, plots are displayed on screen.
- `save`: If `true`, plots are saved to files.

#### Input Data and Dependencies
- Time course data are loaded from: `Cl_WT_measurements.py` and `Cl_H120A_measurements.py`
- The model, time course conditions, and optimization functions are defined in: `Cl_model.py`
- Parameter sets are loaded from output files (e.g., `1234Cl_sym_output.txt`)

If no output file is provided and no parameter set is pasted directly into `Cl.py`, a built-in default set with uniform intermediate values is used.

#### Example Usage
```bash
python Cl.py 1234 mode=1 protein="WT" show=0 save=1

The script accepts additional keyword arguments and supports further customization. See the source code for details.
