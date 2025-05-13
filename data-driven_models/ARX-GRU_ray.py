"""
This script needs the NARX-Hematox project [6] to run properly.
"""

# --- Imports ---
# Core numerical and ML libraries
import numpy as np
import tensorflow as tf
import os
import json
import pandas as pd

# Custom model/loss functions from NARX-Hematox
from NARX-Hematox.utils import *
from NARX-Hematox.NARX import *
from NARX-Hematox.friberg import *

# Local model definitions
from mech_models.tox_models_pop import NpEncoder

import datetime
import argparse
from ray import tune
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

# --- Preprocessing functions ---
log_transformer = FunctionTransformer(np.log10, inverse_func=lambda x: 10.**x,
                                      validate=True, check_inverse=True)
log_transformere = FunctionTransformer(np.log, inverse_func=lambda x:
    np.exp(x), validate=True, check_inverse=True)
mse = MeanSquaredError()
msem = MSE_missing()

# Reduce TensorFlow logging noise
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()



def buildrnn(cell, hp):
    """
    Constructs a recurrent neural network model using a specified RNN cell
    (e.g., GRU or LSTM).

    Parameters
    ----------
    cell : tf.keras.layers.Layer
        Recurrent cell class to use (e.g., `tf.keras.layers.GRUCell`).
    hp : dict
        Dictionary of hyperparameters. Keys include:
            - 'units' (int): Number of units in the recurrent cell.
            - 'seed' (int): Random seed.
            - 'stddev' (float): Standard deviation for GaussianNoise layer.
            - 'in_dr' (float): Input dropout rate.
            - 're_dr' (float): Recurrent dropout rate.
            - 'in_bias' (bool): Whether to use bias in the RNN layer.
            - 'activation' (str): Activation function for RNN.
            - 'out_act' (str): Activation for output layer.
            - 'out_bias' (bool): Whether to use bias in output layer.
            - 'delta' (float): Pruning constraint value.
            - 'out_l2' (float): L2 regularization for output layer.
            - 'es' (bool): Whether to use early stopping variant (ARX_RNN_ES).
            - 'lag' (int): Number of time steps to use for lag.

    Returns
    -------
    NARX-Hematox.NARX.ARX_RNN_ES or NARX-Hematox.NARX.ARX_RNN
        A compiled RNN model with specified architecture.
    """
    if hp['stddev'] != 0.:
       inp = tf.keras.layers.GaussianNoise(hp['stddev'], seed=0)
    else:
        inp = tf.keras.layers.Identity()

    cellb = cell(
            units=hp['units'],
            kernel_initializer=tf.keras.initializers.GlorotUniform(
                seed=hp['seed']), dropout=hp['in_dr'],
        kernel_constraint=Prune(hp['delta']),
            recurrent_dropout=hp['re_dr'], use_bias=hp['in_bias'],
            activation=hp['activation']
    )
    out = tf.keras.layers.Dense(units=1, activation=hp['out_act'],
        use_bias=hp['out_bias'], kernel_constraint=Prune(hp['delta']),
        kernel_regularizer=tf.keras.regularizers.L2(hp['out_l2'])
        )

    if hp['es']:
        model = ARX_RNN_ES(inp=inp, cell=cellb, lags=hp['lag'],
            out=out)
    else:
        model = ARX_RNN(cell=cellb, lags=hp['lag'],
                    out=out)
    return model


def data_prep(hp, df, dft, sc_df, patient, friberg_path, scalery_gen,
              trial_dir=None):
    """
      Prepares input and target sequences for training and testing, including
      Friberg model [1] integration.

      Parameters
      ----------
      hp : dict
          Hyperparameters controlling model type, lag sizes, and training
          options.
      df : pandas.DataFrame
          Time series data containing patient measurements.
      dft : pandas.DataFrame
          Treatment data corresponding to patient IDs.
      sc_df : pandas.DataFrame
          Therapy scenarios for generating synthetic Friberg simulations.
      patient : str or int
          Patient ID to filter data for.
      friberg_path : str
          Path to directory containing Friberg model predictions.
      scalery_gen : sklearn.preprocessing.MinMaxScaler
          Scaler used to normalize output data.
      trial_dir : str, optional
          Directory where additional artifacts like scalers may be saved.

      Returns
      -------
      tuple
          Contains:
          - y_pred_c3 (np.ndarray): Friberg predictions.
          - cycle (int): Last cycle used for training.
          - y_arr_c3 (np.ndarray): True measurements up to selected cycle.
          - y_arr (np.ndarray): Full ground-truth data.
          - t_c3 (np.ndarray): Timestamps for training data.
          - t_arr (np.ndarray): Full time array.
          - y_test (np.ndarray): Ground truth for test portion.
          - y_testlog (np.ndarray): Log-transformed test target.
          - feat_lstm (np.ndarray): Feature tensor for model input.
          - target_lstm (np.ndarray): Target output tensor.
          - test_lstm (np.ndarray): Full input sequence for evaluation.
          - feat_fr (np.ndarray): Friberg training input (if pre-training).
          - target_fr (np.ndarray): Friberg training targets.
          - val_fr (np.ndarray): Friberg validation input.
          - target_val_fr (np.ndarray): Friberg validation targets.
      """
    xlag, ylag = hp['xlag'], hp['ylag']

    # Select data for specific patient
    df_p = df[df['ID'] == patient]
    df_p.index = df_p['TIME']
    t_arr = df_p['TIME'].to_numpy()
    t = np.arange(0, t_arr[-1] + 1, 1)
    y_arr = df_p['Y'].to_numpy() * 1e9

    # Get data up to target cycle
    df_p1_c2 = df_p[df_p['CYCLE'] <= hp['cycle']]
    cycle = df_p1_c2['CYCLE'].max()
    t_c3 = df_p1_c2['TIME'].to_numpy()
    l = t_c3[-1] + 1
    y_arr_c3 = df_p1_c2['Y'].to_numpy() * 1e9

    # Get treatment information
    dft_p = dft[dft['ID'] == patient]
    therapies_days = dft_p['TIME'].to_numpy().astype(int)
    therapies_doses = dft_p['DOSE_ST'].to_numpy()
    therapies = (therapies_days, therapies_doses)
    therapies_c3 = (therapies[0][:cycle], therapies[1][:cycle])

    # Load Friberg model data
    pc = np.loadtxt(os.path.join(friberg_path, 'pc_fri.txt'))
    y_pred_c3 = np.loadtxt(os.path.join(friberg_path, 'fri_pred.csv'))

    # Scale target values for NARX
    y_sc_c3 = scalery_gen.transform(
        log_transformer.transform(y_arr_c3.reshape(-1, 1))).flatten()

    # Estimate initial state (steady state)
    # stst = y_sc_c3[0]
    if not np.isfinite(y_sc_c3[0]):
        if hp['pre-training']:
            stst = scalery_gen.transform(
                log_transformer.transform(np.array(
                    pc[-1]).reshape(-1, 1))).flatten()
        else:
            stst = np.nanmean(y_sc_c3)

    # Construct exogenous input X based on model type
    if hp['pre-training']:
        # Friberg-based simulation for pre-training
        test_fri = test_friberg(pc, therapies, np.arange(0,300))
        X_raw = test_fri[0]
        y_frtrain = test_fri[2]
        scalerX = test_fri[-1]
        if hp['mixed-model']:
            # Save scaler for reproducibility
            scalerX_dict = {'scale_': scalerX.scale_, 'min_': scalerX.min_}
            with open(os.path.join(trial_dir, 'scaler_X.json'), 'w') as f:
                f.write(json.dumps(scalerX_dict, cls=NpEncoder))
            X = scalerX.transform(X_raw.reshape(-1, 1)).flatten()
        else:
            X = np.zeros(len(t))
            X[therapies[0]] = therapies[1]

        # Prepare multiple synthetic training scenarios
        y_frsc = [y_frtrain]
        X_frsc = [X]
        for i in range(7):
            sc_fri = test_friberg(pc, (
                sc_df[f'therapy_days_set{i}'].astype(int).to_numpy(),
                sc_df[f'dosages_set{i}'].to_numpy()), np.arange(0,300,1))
            X_frsc.append(sc_fri[0])
            y_frsc.append(sc_fri[2])
        y_frsc = np.array(y_frsc)

        if hp['mixed-model']:
            # Reshape and scale
            X_frsc = np.array(X_frsc)
            Xshape = X_frsc.shape
            X_frsc = scalerX.fit_transform(X_frsc.reshape(-1, 1)).reshape(
                Xshape)
            scalerX_dict = {'scale_': scalerX.scale_, 'min_': scalerX.min_}
            with open(os.path.join(trial_dir, 'scaler_X.json'), 'w') as f:
                f.write(json.dumps(scalerX_dict, cls=NpEncoder))
        else:
            # Construct dosing matrix directly
            X_frsc = np.zeros((8, 300))
            X_frsc[0, therapies[0]] = therapies[1]
            for i in range(7):
                day = sc_df[f'therapy_days_set{i}'].astype(int).to_numpy(),
                dose = sc_df[f'dosages_set{i}'].to_numpy()
                X_frsc[i + 1, day] = dose

    else:
        X = np.zeros(len(t))
        X[therapies[0]] = therapies[1]

    # Extend inputs with initial padding
    X_add = np.concatenate([np.zeros((28)), X], 0)

    # Scale target and pad
    y = scalery_gen.transform(
        log_transformer.transform(y_arr.reshape(-1, 1))).flatten()
    ylog = log_transformer.transform(y_arr.reshape(-1, 1)).flatten()
    y_pad = np.nan * np.ones(len(X))
    y_pad[t_arr] = y
    y_padlog = np.nan * np.ones(len(X))
    y_padlog[t_arr] = ylog

    y_add = np.concatenate([stst * np.ones(28), y_pad], 0)
    sparse_all = np.nan * np.ones(len(X))
    sparse_all[t_arr] = 1.
    y_test = y_pad[l:]
    y_testlog = y_padlog[l:]

    # Prepare training sequences
    y_train = y_add[:l + 28]
    X_train = X_add[:l + 28]

    # Pretraining feature prep (if enabled)
    if hp['pre-training']:
        X_frtrain, X_frtest = X_frsc[:6], X_frsc[6:]
        y_frtrain, y_frtest = y_frsc[:6], y_frsc[6:]

        # Add padding for lag and steady state
        X_frtest = np.concatenate([np.zeros((2,28)), X_frtest], 1)
        X_frtrain = np.concatenate([np.zeros((6, 28)), X_frtrain], 1)
        shape_train = y_frtrain.shape
        shape_test = y_frtest.shape

        # Transform targets and pad
        y_frtrain = np.concatenate(
            [np.ones((6, hp['lag'])) * stst, scalery_gen.transform(
            log_transformer.transform(
            y_frtrain.reshape(-1,1))).reshape(shape_train)], 1)
        y_frtest = np.concatenate(
            [np.ones((2,hp['lag'])) * stst, scalery_gen.transform(
                log_transformer.transform(
            y_frtest.reshape(-1,1))).reshape(shape_test)],1)

    # Format for input (RNN expects 3D)
    feat_lstm = np.vstack([X_train, y_train]).T
    feat_lstm[xlag:, 1] = np.zeros(len(y_train) - xlag) * np.NaN
    shape = feat_lstm.shape
    feat_lstm = feat_lstm.reshape(1, shape[0], shape[1])
    target_lstm = y_train[hp['lag']:].reshape(1, shape[0] - xlag,
                                              1)
    test_lstm = np.vstack([X_add, y_add]).T
    test_lstm[xlag:, 1] = np.zeros(len(y_add) - xlag) * np.NaN
    shape = test_lstm.shape
    test_lstm = test_lstm.reshape(1, shape[0], shape[1])
    if hp['pre-training']:
        feat_fr = np.dstack([X_frtrain, y_frtrain])
        feat_fr[:,xlag:, 1] = np.zeros(len(y_frtrain[0]) - xlag) * np.NaN

        target_fr = y_frtrain[:,hp['lag']:]
        val_fr = np.dstack([X_frtest, y_frtest])
        val_fr[:,xlag:, 1] = np.zeros(len(y_frtest[0]) - xlag) * np.NaN
        target_val_fr = y_frtest[:,hp['lag']:]
    else:
        feat_fr  = 0.
        target_fr = 0.
        val_fr = 0.
        target_val_fr = 0.

    return (y_pred_c3, cycle, y_arr_c3, y_arr, t_c3, t_arr, y_test,
            y_testlog, feat_lstm,
            target_lstm,  test_lstm, feat_fr, target_fr, val_fr, target_val_fr)


def best_trainable(hp,df,dft,sc_df, fit_dir):
    """
    Trains the model with given hyperparameters and reports performance to
    Ray Tune.

    Parameters
    ----------
    hp : dict
        Dictionary of hyperparameters, including architecture, training regime,
        etc.
    df : pandas.DataFrame
        Time series data of patient measurements.
    dft : pandas.DataFrame
        Treatment data corresponding to patients.
    sc_df : pandas.DataFrame
        Synthetic scenario data for Friberg simulation.
    fit_dir : str
        Directory where model results and predictions are stored.

    Returns
    -------
    None
    """
    np_config.enable_numpy_behavior()
    tf.keras.utils.set_random_seed(
        hp['seed']
    )

    lower = hp['scale_range_lower'][1]
    upper = hp['scale_range_upper'][0]

    # Scale outputs (e.g., platelet counts)
    scalery_gen = MinMaxScaler(feature_range=(lower, upper))
    scalery_gen.fit(log_transformer.transform(
        np.array([hp['4degree'], hp['normal']]).reshape(-1, 1)))

    # prep lists for performance evaluation
    single_perf = []
    single_step = []
    fri_perf = []
    fri_step = []
    single_deg = []
    es_alpha = []

    # objective function
    smsem = Step_MSE_missing(s=hp['s'])

    def err_func():
        if hp['step_mse']:
            return Step_MSE_missing(s=hp['s_train'])
        else:
            return MSE_missing()

    # get patient info
    trial_dir = tune.get_trial_dir()
    # load patient data
    patient = hp['ID']
    df_p = df[df['ID'] == patient]
    p_dir = os.path.join(fit_dir, f'pat_{patient}_cyc{hp["cycle"]}')

    # Prepare input/output sequences
    (y_pred_c3, cycle, y_arr_c3, y_arr, t_c3, t_arr, y_test,
     y_testlog, feat_lstm,
     target_lstm, test_lstm, feat_fr, target_fr, val_fr, target_val_fr) \
        = data_prep(hp, df, dft, sc_df, patient, p_dir, scalery_gen,
                    trial_dir)
    l = t_c3[-1] + 1

    tf.keras.backend.clear_session()

    # first set dropout to zero if needed:
    hpx = hp.copy()
    hpx['in_dr'] = 0.
    hpx['re_dr'] = 0.
    hpx['stddev'] = hp['stddev_ft']

    if hp['celltype'] == 'GRU':
        cell = tf.keras.layers.GRUCell
    else:
        raise ValueError("Unsupported cell type")

    # === Pretraining Phase ===
    if hp['pre-training']:
        model = buildrnn(cell=cell, hp=hp)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp['lr'],
            decay_steps=1800,
            decay_rate=0.8)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        loss_fn = tf.keras.losses.MeanSquaredError()
        model.compile(optimizer=optimizer, loss=loss_fn)

        # call on some data to initialize weights
        yt = model(np.ones((6, hp['lag'] + 1, 2)))

        if hp['index']:
            model.load_weights(hp['index_path'])

        callbacks = tf.keras.callbacks.CallbackList(
            [tf.keras.callbacks.History(),
             EarlyStoppAveraged(monitor='val_loss', av_epochs=hp['av_epochs'],
                                              restore_best_weights=True,
                                              patience=hp['patience'],
                                min_delta=10**-4)])
        callbacks.set_model(model)
        # pre-training fit
        hist_pre = model.fit(x=feat_fr, y=target_fr,
                              validation_data=(val_fr, target_val_fr),
                             batch_size=None,
                              epochs=hp['max_epochs'], verbose=0,
                              callbacks=callbacks)

        model.stop_training = False
        fri_weights = model.get_weights()

        # === Finetuning Phase 1 (frozen layers) ===
        # make new model for transfer to freeze weights
        model = buildrnn(cell=cell, hp=hpx)

        # call to initialize weigths again
        yt = model(np.ones((1, hp['lag'] + 1, 2)))

        model.set_weights(fri_weights)
        for layer in model.layers[:-1]:
            layer.trainable = False
        # opt params
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp['lr_ft'],
            decay_steps=1800,
            decay_rate=0.8)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        # combined loss with smoothing needs to run eagerly
        tf.config.run_functions_eagerly(True)
        loss_fn = tf.function(smse_smoothing(k=int(hpx['k']), c=hpx['c'],
                                             s=hpx['s_train']))
        model.compile(optimizer=optimizer, loss=loss_fn)

        # call for sanity
        yt = model(np.ones((1, hp['lag'] + 1, 2)))
        model.set_weights(fri_weights)

        callbacks = tf.keras.callbacks.CallbackList(
            [tf.keras.callbacks.History(),
             EarlyStoppAveraged(monitor='loss', av_epochs=hp['av_epochs'],
                                              restore_best_weights=True,
                                              patience=hp['patience'],
                                min_delta=10**-4)])
        callbacks.set_model(model)
        hist_pre2 = model.fit(x=feat_lstm, y=target_lstm,
                             epochs=hp['max_epochs'], verbose=0,
                             callbacks=callbacks)

        model.stop_training = False
        for layer in model.layers[:-1]:
            layer.trainable = True

        # pre-training weights
        ptr_weights = model.get_weights()

        yhat = model(test_lstm).numpy().flatten()
        ynoc = log_transformer.inverse_transform(
            scalery_gen.inverse_transform(
                yhat.reshape(-1, 1))).flatten()
        np.savetxt(os.path.join(trial_dir, 'pred_ptr2.csv'), ynoc,
                   delimiter=',')

        tf.config.run_functions_eagerly(False)

        # === Final Finetuning (all layers trainable) ===
        # new model, this is needed due to mixed objective function
        model = buildrnn(cell=cell, hp=hpx)

        # opt params
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=hp['lr_ft'],
            decay_steps=1800,
            decay_rate=0.8)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        loss_fn = Combined_loss([err_func(),
                         smoothing(k=int(hpx['k']), c=hpx['c'])])
        model.compile(optimizer=optimizer, loss=loss_fn)

        # call to initialize weights again
        yt = model(np.ones((1, hp['lag'] + 1, 2)))
        model.set_weights(ptr_weights)

        callbacks = tf.keras.callbacks.CallbackList(
            [tf.keras.callbacks.History(),
             EarlyStoppAveraged(monitor='loss', av_epochs=hp['av_epochs'],
                                              restore_best_weights=True,
                                              patience=hp['patience'],
                                min_delta=10**-5)])
        callbacks.set_model(model)

        hist = model.fit(x=feat_lstm, y=target_lstm,
                              batch_size=len(target_lstm),
                              epochs=hp['max_epochs'], verbose=0,
                              callbacks=callbacks)
    # === Evaluation ===
    model.save_weights(
        os.path.join(trial_dir, 'weights.h5'))
    yhat = model(test_lstm).numpy().flatten()

    # Guard against NaNs in prediction
    if not np.all(np.isfinite(yhat)):
        mse_metric = np.infty
        smse_metric = np.infty
        ynoc = np.zeros(yhat.shape) * np.nan
    else:
        ynoc = log_transformer.inverse_transform(
            scalery_gen.inverse_transform(yhat.reshape(-1, 1))).flatten()

        # update metric
        ylog = scalery_gen.inverse_transform(yhat.reshape(-1, 1)).flatten()

        if hp['cycle'] >= df_p['CYCLE'].max():
            # if cycle=6 (full time series fit, set whole time series for
            # evaluation
            l = 0
            y_arrlog = log_transformer.transform(y_arr.reshape(-1,
                                                               1)).flatten()
            y_testlog = np.nan * np.ones(len(yhat))
            y_testlog[t_arr] = y_arrlog

            # full ts fits should not count to optimization!
            if hp['cycle'] != 6:
                mse_metric = np.nan
                smse_metric = np.nan

            else:

                mse_metric = msem(y_pred=ylog[l:],
                                  y_true=y_testlog).numpy()
                smse_metric = smsem(y_pred=ylog[l:],
                                    y_true=y_testlog).numpy()

        else:
            mse_metric = msem(y_pred=ylog[l:],
                              y_true=y_testlog).numpy()
            smse_metric = smsem(y_pred=ylog[l:],
                                y_true=y_testlog).numpy()
    # Log results
    np.savetxt(os.path.join(trial_dir, 'pred.csv'), ynoc, delimiter=',')

    y_fri_log = log_transformer.transform(
        y_pred_c3.reshape(-1, 1)).flatten()

    # update metric
    mse_fri = msem(y_pred=y_fri_log[l:],
                   y_true=y_testlog).numpy()
    smse_fri = smsem(y_pred=y_fri_log[l:],
                     y_true=y_testlog).numpy()

    single_perf.append(mse_metric)
    single_step.append(smse_metric)
    fri_perf.append(mse_fri)
    fri_step.append(smse_fri)
    if hp['celltype'] != 'NARX' and hp['es']:
        es_alpha.append(model.alpha.numpy().flatten())
    else:
        es_alpha.append(1.)

    chkpt_dict = {}
    chkpt_dict['av_mse'] = np.nanmean(single_perf)
    chkpt_dict['av_dd'] = np.nanmean(single_deg)
    chkpt_dict['av_fri'] = np.nanmean(fri_perf)
    chkpt_dict['av_step'] = np.nanmean(single_step)
    chkpt_dict['av_fri_step'] = np.nanmean(fri_step)
    chkpt_dict['av_alpha'] = np.nanmean(es_alpha)

    with (open(os.path.join(trial_dir, 'chpkt.json'),
               'w') as file_pi):
        file_pi.write(json.dumps(chkpt_dict, cls=NpEncoder))
    tune.report(av_step=chkpt_dict['av_step'], dict=chkpt_dict)
    tf.keras.backend.clear_session()
    del model


def trial_dir_namer(trial):
    """
    Custom naming function for Ray Tune trial directories.

    Parameters
    ----------
    trial : ray.tune.Trial
        Ray Tune trial object.

    Returns
    -------
    str
        Directory name string for the trial.
    """
    return f'p_{trial.config["ID"]}'


def main():
    """
    Main execution entry point for tuning and evaluating models over multiple
    cycles.

    - Parses command line arguments
    - Loads input data
    - Sets up experiment directories
    - Iterates over therapy cycles to train and evaluate models
    - Uses Ray Tune to optimize over patient-specific configurations

    Returns
    -------
    None
    """
    # Parse arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_cpus", type=int, default=4,
                        help="Number of available CPUs to allocate")
    parser.add_argument("--ptr", type=int, default=1,
                        help="Whether to use pre-training (1=yes, 0=no)")
    args = parser.parse_args()
    ptrset = bool(args.ptr)

    # Create result and data paths
    now = datetime.datetime.now()
    res_par_dir = "."  # base results directory
    data_dir = "../data_example"  # input data directory
    os.makedirs(res_par_dir, exist_ok=True)

    # Subdirectory for this run
    res_dir = os.path.join(res_par_dir,
                           f'test_{now.strftime("%Y%m%d-%H%M%S")}')
    os.makedirs(res_dir, exist_ok=True)

    # === Load data ===
    # patient measurements
    dfp = pd.read_csv(os.path.join(data_dir,
                                   "example_platelets.csv"))
    # treatment plans
    dft = pd.read_csv(
        os.path.join(data_dir, "example_treatment.csv"))
    # synthetic therapy scenarios
    sc_df = pd.read_csv(os.path.join('NARX-Hematox',
                                     'therapies_all.csv'))
    # pretrained weights
    index_path = os.path.join("NARX-Hematox",
                              "index_weights.h5")

    IDs_all = dfp['ID'].unique().tolist()  # all patient IDs

    # === Loop over cycles 1–6 ===
    for cyc in range(1,7):
        ptr = ptrset
        cycle = cyc
        lr = 0.01
        lr_ft = 0.001
        cyc_dir = os.path.join(res_dir, f'ptr_{ptr}_cyc_{cycle}')
        os.makedirs(cyc_dir, exist_ok=True)

        fit_dir = os.path.join(data_dir, f'cycle_{cycle}')
        if not os.path.exists(fit_dir):
            raise ValueError(f"Missing fit_dir: {fit_dir}")

        # === Define hyperparameters ===
        hp = {'seed': 0,
              'ID': tune.grid_search(IDs_all),
              'xlag': 28,
              'ylag': 21,
              'lag': 28,
              'units': 64,
              'activation': 'sigmoid',
              'out_act': 'linear',
              'in_bias': True,
              'pre-training': ptr,
              'mixed-model': False,
              'step_mse': True,
              'celltype': 'GRU',
              'scale_range_lower': [-1, -0.5],
              'scale_range_upper': [0.5, 1.5],
              '4degree': 25 * 10 ** 9,
              'normal': 300 * 10 ** 9,
              'k': 2,
              'c': 0.01,
              's': 1.,
              'delta': 0.007,
              'patience': 250,
              'max_epochs': 10200,
              'lr': lr,
              'lr_ft': lr_ft,
              'cycle': cycle,
              'min_delta': 10 ** -4,
              'in_dr': 0.,
              're_dr': 0,
              'out_dr': 0.,
              'ker_l2': 0.,
              're_l2': 0.,
              'out_l2': 0.01,
              'out_bias': False,
              's_train': 0.1,
              'es': True,
              'av_epochs': 20,
              'stddev_ft': 0.2,
              'stddev': 0.,
              'index': True,
              'index_path': index_path
              }

        resume = 'AUTO'
        max_conc_trials = 32  # Adjust for local machine or cluster

        with open(os.path.join(cyc_dir, 'config.txt'), 'w') as f:
            f.write(str(hp))
            f.close()

        # === Run Ray Tune optimization ===
        best_analysis = tune.run(
            tune.with_parameters(best_trainable, df=dfp, dft=dft,
                                 best=True,
                                 sc_df=sc_df, fit_dir=fit_dir),
            config=hp, num_samples=1, metric='av_step', mode='min',
            name='trials', local_dir=cyc_dir,
            max_concurrent_trials=max_conc_trials,
            resources_per_trial={
                "cpu": 2,
                "gpu": 0  # number of gpus avaiable/ max concurrent trials
            }, time_budget_s=int(15 * 60 * 60),
            trial_dirname_creator=trial_dir_namer,
            resume=resume,
            progress_reporter=tune.CLIReporter(max_report_frequency=60 * 5)
        )

        # === Save best trial results ===
        bdf = best_analysis.results_df.sort_values(by='av_step')
        bdf.to_csv(os.path.join(cyc_dir, 'best_last_results.csv'))

        # Simplify result DataFrame for downstream plotting/comparison
        bdf_simple = bdf[
            ['dict.av_mse', 'dict.av_dd', 'dict.av_old',
             'dict.av_fri', 'dict.av_step', 'dict.av_fri_step',
             'dict.av_alpha', 'config.ID', 'config.c', 'config.delta'
             ]]

        clist = bdf_simple.columns.to_list()
        csl = []
        for col in clist:
            if col[:4] == 'dict':
                csl.append(col[5:])
            elif col[:6] == 'config':
                csl.append(col[7:])
            else:
                csl.append(col)

        cdict = {f'{clist[i]}': csl[i] for i in range(len(csl))}
        bdfs = bdf_simple.rename(columns=cdict).dropna(how='any')
        bdfs.to_csv(os.path.join(cyc_dir, 'best_simple_results.csv'))
        bdfm = bdfs.groupby(['c', 'delta']).mean()[
            ['av_mse', 'av_dd', 'av_old', 'av_fri', 'av_step',
             'av_fri_step',
             'av_alpha']]
        bdfm = bdfm.reset_index()
        bdfm.to_csv(os.path.join(cyc_dir, 'best_av_results.csv'))

    # Done!
    print('Done', datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"))

if __name__ == '__main__':
    main()