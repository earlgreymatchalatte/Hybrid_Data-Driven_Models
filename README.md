[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
# Hybrid mechanistic and data-driven personalized prediction models for platelet dynamics

This repository accompanies the puplication "Developing hybrid mechanistic 
and data-driven personalized prediction models for platelet dynamics". This project predicts blood cell (platelet) counts in chemotherapy patients using **mechanistic**, **hybrid**, and **data-driven NARX models**. It leverages known pharmacological models (Friberg), physics-informed neural networks (Julia UDEs), and deep learning (GRU-based RNNs). 

Performance is evaluated **per patient** and based on **data availability** 
at training time.

---

## 📁 Project Structure

```
.
├── data_example/ # Input data (platelet & treatment CSVs), patient-specific model folders
├── data-driven_models/ # Python-based models (ARX-GRU with Ray Tune)
├── hybrid_models/ # Julia-based hybrid UDE models
├── mech_models/ # Mechanistic modeling code (Friberg + population toxicity models)
├── Example_data_generator.ipynb # Example notebook for synthetic data
├── LICENSE.md # Project license
├── Manifest.toml, Project.toml # Julia package manifests
├── requirements.txt # python requirements
├── REFERENCES.md # Bibliography and citations

```
---

## 🧪 Goal

The objective is to **predict platelet counts** in cancer patients undergoing chemotherapy. Different modeling paradigms are compared:

- **Mechanistic**: Friberg pharmacokinetic/pharmacodynamic (PK/PD) model [1]
  and extensions of it [2,3]
- **Hybrid**: Physics-informed neural networks implemented in Julia (UDEs) [5]
- **Data-driven**: GRU-RNNs with autoregressive inputs (NARX) [7]

---

## 📥 Inputs

Files for testing are supplied, as real patient data cannot be shared.
All input files for testing are in `data_example/`:

- `example_platelets.csv`: Time-series platelet measurements per patient
- `example_treatment.csv`: Chemotherapy dosage schedules per patient
- `p_trained_pop.jld2`: Population-level pre-trained model (Julia)
- `pat_0_cyc*/`: Per-cycle simulation results (one folder per patient-cycle)

These input files are generated with the notebook `Example_data_generator.
ipynb`, except the Julia model, which was obtained with the 
`hybrid_models/UDE_rep.jl` by only pre-training on the Friberg population 
solution.

---

## 📤 Outputs

- **Prediction CSVs**: Time series platelet predictions (`pred.csv`) per cycle
- **Evaluation Metrics**: SMSE, Saved per cycle and patient in config/result 
  folders
---

## ⚙️ Requirements
### Python
for mechanistic models and NARX models
- Python 3.9+
- TensorFlow 2.x
- pandas, numpy, scikit-learn, scipy
- ray[tune] 1.13

Install with:

```bash
pip install -r requirements.txt
```

The NARX models in `data-driven_models` need the `NARX-Hematox` package [6].
Download the repository and include them on the same level. Make sure the 
folder structure looks like this:
```
your-project/
├── NARX-Hematox/
├── data-driven_models/
├── data_example/
├── ...
```



### Julia 
for hybrid models

- Julia 1.8+
- DifferentialEquations.jl
- ModelingToolkit.jl
- DiffEqFlux.jl

Run Julia from the root directory:
```bash
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

---

## 🚀 How to Run

### Mechanistic Models (Friberg, Mangas-Sanjuan, Henrich)

To run the mechanistic PK/PD models for each patient-cycle combination:

```bash
python mech_models/tox_models_pop.py --arr 0 --run_dir mech_output
```
Arguments:
```
--arr : Index into patient-cycle combinations
Used to select which patient and cycle to run (e.g. for SLURM array jobs)
--run_dir : Subfolder under results directory
Output predictions will be saved under ./mech_output/pat_<ID>_cyc<CYCLE>/
```



### Hybrid models (Julia)

From the project root:
```bash
julia hybrid_models/UDE_add.jl --arr 1
julia hybrid_models/UDE_rep.jl --arr 1
```

Arguments:
```
--arr : Index into patient-cycle combinations
Used to select which patient and cycle to run (e.g. for SLURM array jobs)
```

Edit model or simulation parameters directly in the Julia scripts.


### Data-driven (GRU) models

From data-driven_models/, run:

```bash
python ARX-GRU_ray.py --num_cpus 4 --ptr 1
```
Arguments:

    --num_cpus: Number of CPUs for Ray Tune

    --ptr: Use pre-training on Friberg model (1 for yes, 0 for no). 
Published results were obtained with pre-training.

Predictions and logs will be saved under ./test_\<timestamp>/.



---

## 📚 References

See REFERENCES.md for citations and model origins. Key reference:

    Friberg et al., Semimechanistic model for myelosuppression, CPT 2002.

    NARX-Hematox GitHub Repository https://github.com/mariesteinacker/NARX-Hematox

---

## 📄 License

This project is licensed under the terms of the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

You may use, modify, and distribute this software under the conditions of the AGPL license. If you deploy the software over a network (e.g., as a web service), the complete source code must be made available to users.

See [`LICENSE.md`](LICENSE.md) for full details.


## Issues 
If you encounter any issues or have questions, please 
open an issue in the repository. 