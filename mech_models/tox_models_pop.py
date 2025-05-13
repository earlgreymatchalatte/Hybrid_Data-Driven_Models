import pandas as pd
from scipy.optimize import minimize
import os
import argparse
import itertools
from tox_models import *
import json

class NpEncoder(json.JSONEncoder):
    """
    JSON encoder for NumPy data types.

    This encoder allows serialization of NumPy integers, floats, and arrays
    to standard Python types, making them compatible with JSON output.

    Adapted from:
    https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def run_pat(
    trial_config: dict,
    dfs: tuple[pd.DataFrame, pd.DataFrame]
    ) -> None:
    """
    Fit multiple PKPD models (Friberg, Mangas-Sanjuan, Henrich) for a
    specific patient and cycle, and store predictions and parameter estimates.

    This function fits models to longitudinal blood count data using
    nonlinear optimization, stores the estimated parameters, and writes
    predicted trajectories and metadata to disk.

    Parameters
    ----------
    trial_config : dict
        Configuration dictionary containing:
        - 'pat': patient ID
        - 'cycle': cycle number to fit
        - 'pat_dir': output directory for saving results
    dfs : tuple of pd.DataFrame
        Tuple of:
        - patient measurements (dfp)
        - therapy schedule (dft)

    Returns
    -------
    None
        All relevant information is written to disk.
    """
    # Default model parameters
    # Friberg params
    gamma = 0.316
    MTT = 195
    c0 = 270
    Emax = 2

    # MS model params
    kcyc = 1.9 / 24
    fprol = 0.58

    #MS model-revised params
    kcyc2 = 1/60
    Emax_rev = Emax * 60

    # Henrich model params
    ftr = 0.7

    # Extract patient and cycle
    dfp, dft = dfs
    pat = trial_config['pat']
    cyc = trial_config['cycle']
    p_dir = trial_config['pat_dir']

    df_p = dfp[dfp['ID'] == pat]
    df_p.index = df_p['TIME']
    t_arr = df_p['TIME'].astype(int).to_numpy()
    t = np.arange(0, t_arr[-1] + 1, 1)

    # Data up to and including selected cycle
    df_p1_c2 = df_p[df_p['CYCLE'] <= cyc]
    cycle = df_p1_c2['CYCLE'].max().astype(int)
    t_c = df_p1_c2['TIME'].to_numpy().astype(int)
    y_arr_c = df_p1_c2['Y'].to_numpy() * 1e9

    dft_p = dft[dft['ID'] == pat]
    therapies_days = dft_p['TIME'].to_numpy().astype(int)
    therapies_doses = dft_p['DOSE_ST'].to_numpy()
    therapies = (therapies_days, therapies_doses)

    therapies_c = (therapies[0][:cycle], therapies[1][:cycle])

    # --- Friberg model fit ---
    # (gamma, MTT in h, slope, c0 in cells/(l * 10^9))
    p0fri = np.array([0.316, 195, 2., 270.])
    pop_params_fri = p0fri
    bounds_fri = [(0.05, 0.45), (20, 350), (1., 5),
              (25, 450)]

    resfri = minimize(poplog_fit_friberg_mse, p0fri,
                     args=(pop_params_fri, y_arr_c, t_c, therapies_c),
                     bounds=bounds_fri,
                     method='Nelder-Mead', tol=10 ** -6)

    pc_fri = resfri.x

    np.savetxt(os.path.join(p_dir, 'pc_fri.txt'), pc_fri)
    y_fripat = pred_fri(pc_fri, t, therapies)
    np.savetxt(os.path.join(p_dir, 'fri_pred.csv'), y_fripat)


    # --- Mangas-Sanjuan model fit ---
    p0ms = np.array([gamma, MTT, Emax, c0, kcyc, fprol])
    pop_params_ms = p0ms
    bounds_ms = [(0.05, 0.45), (20, 350), (0.1, 5),
                 (25, 450), (0., 10 ** 9), (0., 1.)]

    resms = minimize(poplog_fit_ms_mse, p0ms,
                     args=(pop_params_ms, y_arr_c, t_c, therapies_c),
                     bounds=bounds_ms,
                     method='Nelder-Mead', tol=10 ** -6)

    pc_ms = resms.x

    np.savetxt(os.path.join(p_dir, 'pc_ms.txt'), pc_ms)
    y_mspat = pred_ms(pc_ms, t, therapies)
    np.savetxt(os.path.join(p_dir, 'ms_pred.csv'), y_mspat)

    # --- Mangas-Sanjuan revised model fit ---
    p0msr = np.array([gamma, MTT, Emax_rev, c0, kcyc, kcyc2, fprol])
    pop_params_msr = p0msr
    bounds_msr = [(0.05, 0.45), (20, 350), (0.1, 500),
                 (25, 450), (0., 10 ** 9), (0., 1.), (0., 1.)]

    resmsr = minimize(poplog_fit_ms_mse_rev, p0msr,
                     args=(pop_params_msr, y_arr_c, t_c, therapies_c),
                     bounds=bounds_msr,
                     method='Nelder-Mead', tol=10 ** -6)

    pc_msr = resmsr.x

    np.savetxt(os.path.join(p_dir, 'pc_msr.txt'), pc_msr)
    y_msrpat = pred_ms_rev(pc_msr, t, therapies)
    np.savetxt(os.path.join(p_dir, 'msr_pred.csv'), y_msrpat)


    # --- Henrich model fit ---
    p0h = np.array([gamma, MTT, Emax, c0, ftr])
    pop_params_h = p0h
    bounds_h = [(0.05, 0.45), (20, 350), (0.1, 5),
                (25, 450), (0., 1.)]
    resh = minimize(poplog_fit_h_mse, p0h,
                    args=(pop_params_h, y_arr_c, t_c, therapies_c),
                    bounds=bounds_h,
                    method='Nelder-Mead', tol=10 ** -6)
    pc_h = resh.x
    np.savetxt(os.path.join(p_dir, 'pc_h.txt'), pc_h)
    y_hpat = pred_h(pc_h, t, therapies)
    np.savetxt(os.path.join(p_dir, 'h_pred.csv'), y_hpat)

    # Save trial configuration
    with open(os.path.join(p_dir, 'trial_config.json'), 'w') as fp:
        json.dump(trial_config, fp, sort_keys=True, indent=4, cls=NpEncoder)


def main() -> None:
    """
    Entry point for running patient-specific model fitting.

    This script parses command-line arguments (e.g., for Slurm array jobs),
    selects a patient-cycle combination, prepares input/output paths, and
    runs the model fitting pipeline for that patient.

    Command-line Arguments
    ----------------------
    --arr : int
        Index into the patient-cycle combinations
        (e.g., from SLURM_ARRAY_TASK_ID).
    --run_dir : str
        Subdirectory name under the base results directory.
    Returns
    -------
    None
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arr", type=int, required=False, help="slurm array", default=0
    )
    parser.add_argument(
        "--run_dir", type=str, required=False, help="run directory",
        default=""
    )

    args = parser.parse_args()
    arr = args.arr
    run_dir = args.run_dir

    res_dir = "." #your_res_dir
    data_dir = "data_example" #your_data_dir

    ws_dir = os.path.join(res_dir, run_dir)

    dfp = pd.read_csv(os.path.join(data_dir,
                                   'example_platelets.csv'),
                      sep=',')
    dft = pd.read_csv(os.path.join(data_dir,
                                   'example_treatment.csv'), sep=',')
    pats = dfp["ID"].unique()


    cycles = [1,2,3,4,5,6]

    combinations = itertools.product(pats, cycles)
    combo = [x for x in combinations]
    pat_id, cycle = combo[arr]

    trial_config = {"pat": pat_id,
                    "cycle": cycle,
                    "arr": arr
                    }
    pat_dir = os.path.join(ws_dir,
                           f"pat_{pat_id}_cyc{cycle}")
    if not os.path.exists(pat_dir):
        os.mkdir(pat_dir)
    trial_config["pat_dir"] = pat_dir
    dfs = (dfp, dft)

    run_pat(trial_config, dfs)


if __name__ == '__main__':
    main()

