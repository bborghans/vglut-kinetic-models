### GA.py README

`GA.py` performs parameter optimization and limited result visualization for a kinetic model describing active transport by VGLUT1, for glutamate and aspartate substrate in parallel.

#### Mode 0 — Parameter Optimization
When run with `mode=0`, the script optimizes model parameters using residual sum of squares. Results are saved using a version ID (e.g., `1234`), producing files like `1234GA_sym_output`, which is provided containing an optimized parameter set. Parameters corresponding to the best fit are saved whenever the generation number reaches a defined checkpoint (default: 500).

If the script is restarted, it will overwrite the output file at the first checkpoint, using the most recent parameter set from the previous run.

#### Modes -3-6 — Result Visualization
These modes generate plots based on the latest optimized parameter set:
- `mode=1`: Numerical details
- `mode=2`: State distribution plots
- `mode=3`: Flux plots
- `mode=4`: Test optimization without starting it
- `mode=5`: Manual parameter save
- `mode=6`: Worsenfactor read-out (used to limit residual sum of squares by experiment)
- `mode=-1`: Generate variant parameters (VariantGen)
- `mode=-2`: Compile variant outputs (VarGen compile)
- `mode=-3`: Check across a parameter range (rangecheck)

Additional options:
- `show`: If `true`, plots are displayed on screen.
- `save`: If `true`, plots are saved to files.

#### Input Data and Dependencies
- Time course data are loaded from: `GlutWT_measurements.py` and `AspWT_measurements.py`
- The model, time course conditions, and optimization functions are defined in: `GA_model.py`
- Weighting factors for time course data and calculated metrics are loaded from: `GA_weights.py`
- Parameter sets are loaded from output files (e.g., `1234GA_sym_output`)

If no output file is provided and no parameter set is pasted directly into `GA.py`, a built-in default set with uniform intermediate values is used.

#### Example Usage
```bash
python GA.py 1234 mode=1 show=0 save=1

---

### GA_analysis.py README

`GA_analysis.py` separately performs the fit result visualization for a kinetic model describing active transport by VGLUT1, for glutamate and aspartate substrate in parallel, from GA.py-generated file by version ID (e.g., `1234`).

#### Modes 0-1 — Fit Visualization
These modes generate plots based on one or more parameter sets in one or more opened file:
- `mode=0`: Default output plots
- `mode=1`: Provides additional numerical/text output

Additional options:
- `show`: If `true`, plots are displayed on screen.
- `save`: If `true`, plots are saved to files.

#### Input Data and Dependencies
- Time course data are loaded from: `GlutWT_measurements.py` and `AspWT_measurements.py`
- The model, time course conditions, and various functions are defined in: `GA_model.py`
- Parameter sets are loaded from output files (e.g., `1234GA_sym_output`)

#### Example Usage
```bash
python GA_analysis.py 1234 mode=0 show=0 save=1

The scripts accept additional keyword arguments and supports further customization. See the source code for details.