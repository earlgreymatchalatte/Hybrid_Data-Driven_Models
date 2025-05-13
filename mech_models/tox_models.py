import numpy as np
from typing import Tuple, Sequence
from scipy.integrate import solve_ivp


def E_drug_step(t: float, slope: float,
                therapies_tuple: Tuple[Sequence[float],
                Sequence[float]]) -> float:
    """
    Calculate the drug effect (E_drug) at a given time using a step
    function.

    This function determines the drug effect based on a predefined
    therapy schedule. If the current time `t` is before the start of
    any therapy, the effect is zero. Otherwise, the effect is computed
    based on the most recent therapy start time and its corresponding
    dosage, lasting for a 24-hour window.

    Parameters
    ----------
    t : float
        Time point at which to calculate the drug effect, in hours.
    slope : float
        Slope of the drug effect response function per unit dosage.
    therapies_tuple : tuple of (array-like, array-like)
        Tuple containing:
        - an array or list of therapy start times, in hours
        - an array or list of corresponding dosages

    Returns
    -------
    float
        The calculated drug effect (E_drug) at time `t`.

    References
    ----------
    See [6].
    """

    therapies, dosage = therapies_tuple

    if t < np.min(therapies):
        E_drug = 0
    else:
        # create local time from the last therapy start,
        # where t_loc(start) = 0
        local_start = max(np.extract(therapies <= t, therapies))
        local_dosage = dosage[np.argwhere(therapies == local_start)].flatten()[
            0]
        t_loc = t - local_start
        if t_loc <= 23:
            E_drug = slope * local_dosage
        else:
            E_drug = 0.
    return E_drug


def friberg_drug(t: float, y: np.ndarray, ktr: float, gamma: float,
                 c0: float, slope: float,
                 therapies: Tuple[Sequence[float], Sequence[float]]
                 ) -> tuple[float, float, float, float, float]:
    """
    Compute the derivatives of the Friberg PKPD model describing
    drug effects on cell proliferation.

    This model captures the dynamics of proliferating cells under
    drug influence with a proliferation compartment, three transit
    compartments, and a circulating cell compartment.

    Parameters
    ----------
    t : float
        Time point at which to evaluate the derivatives, in hours.
    y : np.ndarray
        Current state of the system. Contains values for:
        - prol: Proliferating compartment
        - t1, t2, t3: Transit compartments
        - circ: Circulating compartment
    ktr : float
        Rate constant for transitions between compartments.
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    slope : float
        Slope of the drug effect function per unit dose.
    therapies : tuple of (Sequence[float], Sequence[float])
                Tuple containing:
                - an array or list of therapy start times
                - an array or list of corresponding dosages

    Returns
    -------
    tuple of float
        Derivatives for each compartment:
        (dprol, dt1, dt2, dt3, dcirc)


    References
    ----------
    See [1] for the original model, and [6] for implementation.
    """

    # get vec components
    prol, t1, t2, t3, circ = y

    # drug effect
    Edrug = E_drug_step(t, slope, therapies)

    # diff eq system
    dprol = ktr * prol * (1 - Edrug) * (c0 / circ) ** gamma - ktr * prol
    dt1 = ktr * prol - ktr * t1
    dt2 = ktr * t1 - ktr * t2
    dt3 = ktr * t2 - ktr * t3
    dcirc = ktr * t3 - ktr * circ

    return dprol, dt1, dt2, dt3, dcirc


def solve_friberg(gamma: float, c0: float, MTT: float, slope: float,
                  therapies_tuple: Tuple[Sequence[float], Sequence[float]],
                  end_fu: float, t_eval: np.ndarray) -> np.ndarray:
    """
    Solve the Friberg PKPD model for drug-induced effects on cell
    proliferation.

    This function integrates the Friberg model over time using a given
    drug administration schedule and parameter set.

    Parameters
    ----------
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    MTT : float
        Mean transit time governing the compartmental delay.
    slope : float
        Slope of the drug effect function per unit dose.
    therapies_tuple : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times (in days)
        - an array or list of corresponding dosages
    end_fu : float
        Duration of follow-up (in days) after therapy administration.
    t_eval : np.ndarray
        Time points (in days) at which to evaluate the model solution.

    Returns
    -------
    np.ndarray
        Solution for the circulating cells compartment over time.

    References
    ----------
    See [1] for the original model, and [6] for implementation.
    """

    therapies, dosage = therapies_tuple
    # model works in hours, not days
    therapies = therapies * 24

    # params
    ktr = 4. / MTT
    y0 = c0 * np.ones(5)

    # time arrs
    tmax = np.max(therapies) + end_fu * 24

    # solve friberg
    sol = solve_ivp(friberg_drug, (0, tmax), y0, method='LSODA',
                    t_eval=t_eval*24,
                    args=(
                    ktr, gamma, c0, slope, (therapies, dosage)),
                    rtol=1e-6, atol=1e-9, max_step=24, jac=None)
    return sol.y[-1]


def henrich_drug(t: float, y: np.ndarray, ktr: float, ftr: float,
                 gamma: float, c0: float, slope: float,
                 therapies: Tuple[Sequence[float], Sequence[float]]
                 ) -> tuple[float, float, float, float, float, float]:
    """
    Compute the rate of change for each compartment in the Henrich PKPD
    model of drug-induced effects on hematopoietic and circulating cells.

    This function models the dynamics of stem cell differentiation and
    proliferation under drug influence. It evaluates the system of
    differential equations governing six compartments, incorporating a
    feedback mechanism and drug effect modulation.

    Parameters
    ----------
    t : float
        Current time point.
    y : np.ndarray
        Current state of the system. Contains values for:
        - stem: Hematopoietic stem cells
        - prol: Proliferating cells
        - t1, t2, t3: Transit compartments
        - circ: Circulating cells
    ktr : float
        Rate constant for transitions between compartments.
    ftr : float
        Fraction of the transition rate allocated to proliferating cells.
        Controls the expression of cumulative toxicity.
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    slope : float
        Slope of the drug effect function per unit dose.
    therapies : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times
        - an array or list of corresponding dosages

    Returns
    -------
    tuple of float
        Derivatives for each compartment:
        (dstem, dprol, dt1, dt2, dt3, dcirc)

    References
    ----------
    See [2] for the original model.
    """
    # get vec components
    stem, prol, t1, t2, t3, circ = y

    # get Edrug
    Edrug = E_drug_step(t, slope, therapies)
    fb = (c0 / circ) ** gamma
    kprol = ftr * ktr
    kstem = (1 - ftr) * ktr

    # diff eq system
    dstem = kstem * stem * (1 - Edrug) * fb - kstem * stem
    dprol = kprol * prol * (1 - Edrug) * fb - ktr * prol + kstem * stem
    dt1 = ktr * prol - ktr * t1
    dt2 = ktr * t1 - ktr * t2
    dt3 = ktr * t2 - ktr * t3
    dcirc = ktr * t3 - ktr * circ

    return dstem, dprol, dt1, dt2, dt3, dcirc


def solve_henrich(gamma: float, c0: float, MTT: float, slope: float,
                  ftr: float,
                  therapies_tuple: Tuple[Sequence[float], Sequence[float]],
                  end_fu: float, t_eval: np.ndarray) -> np.ndarray:
    """
    Solve the Henrich PKPD model for drug-induced effects on hematopoietic
    and circulating cells.

    This function integrates the Henrich model over time using a given drug
    administration schedule and parameter set.

    Parameters
    ----------
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    MTT : float
        Mean transit time governing the compartmental delay.
    slope : float
        Slope of the drug effect function per unit dose.
    ftr : float
        Fraction of the transition rate allocated to proliferating cells.
        Controls the expression of cumulative toxicity.
    therapies : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times (in days)
        - an array or list of corresponding dosages
    end_fu : float
        Duration of follow-up (in days) after therapy administration.
    t_eval : np.ndarray
        Time points (in days) at which to evaluate the model solution.

    Returns
    -------
    np.ndarray
        Solution for the circulating cells compartment over time.

    References
    ----------
    See [2] for the original model.
    """

    therapies, dosage = therapies_tuple
    # add therapy time points
    therapies = therapies * 24

    # params
    ktr = 4. / MTT
    y0 = c0 * np.ones(6)

    # time arrs
    tmax = np.max(therapies) + end_fu * 24
    # solve friberg
    sol = solve_ivp(henrich_drug, (0, tmax), y0, method='LSODA',t_eval=t_eval*24,
                    args=(
                    ktr, ftr, gamma, c0, slope, (therapies, dosage)),
                    rtol=1e-13, atol=1e-15, max_step=24, jac=None)
    return sol.y[-1]


def MSmodel_drug(t: float, y: np.ndarray, ktr: float, fprol: float,
                 kcyc: float, gamma: float, c0: float, slope: float,
                 therapies: Tuple[Sequence[float], Sequence[float]]
                 )-> tuple[float, float, float, float, float, float, float]:
    """
    Compute the rate of change for each compartment in the Mangas-Sanjuan
    PKPD model of drug-induced effects on cell populations.

    This function models the dynamics of proliferative, quiescent, transit,
    and circulating cell compartments under the influence of a cytotoxic
    drug. It evaluates the system of differential equations defined by the
    Mangas-Sanjuan model using a given therapy schedule and parameter set.

    Parameters
    ----------
    t : float
        Current time point.
    y : np.ndarray
        Current state of the system. Contains values for:
        - prol: Proliferating cells
        - q1, q2: Quiescent compartments
        - t1, t2, t3: Transit compartments
        - circ: Circulating cells
    ktr : float
        Rate constant for transitions between compartments.
    fprol : float
        Fraction of cells in the proliferative state.
    kcyc : float
        Rate constant governing the cell cycle through proliferative and
        quiescent compartments.
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    slope : float
        Slope of the drug effect function per unit dose.
    therapies : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times
        - an array or list of corresponding dosages

    Returns
    -------
    tuple of float
        Derivatives for each compartment:
        (dprol, dq1, dq2, dt1, dt2, dt3, dcirc)

    References
    ----------
    See [3] for the original model.
    """
    # get vec components
    prol, q1, q2, t1, t2, t3, circ = y

    # get Edrug
    Edrug = E_drug_step(t, slope, therapies)
    fb = (c0 / circ) ** gamma
    kprol = ktr * fprol

    # diff eq system
    dprol = kprol * prol * fb * (1. - Edrug) + \
            kcyc * q2 - ktr * fprol * prol - kcyc * (1. - fprol) * prol
    dq1 = kcyc * (1. - fprol) * prol - kcyc * q1
    dq2 = kcyc * (q1 - q2)
    dt1 = ktr * fprol * prol - ktr * t1
    dt2 = ktr * t1 - ktr * t2
    dt3 = ktr * t2 - ktr * t3
    dcirc = ktr * t3 - ktr * circ

    return dprol, dq1, dq2, dt1, dt2, dt3, dcirc


def solve_MSmodel(
    gamma: float, c0: float, MTT: float, slope: float, fprol: float,
    kcyc: float, therapies_tuple: Tuple[Sequence[float], Sequence[float]],
    end_fu: float, t_eval: np.ndarray) -> np.ndarray:
    """
    Solve the Mangas-Sanjuan PKPD model for drug-induced effects on blood
    cell dynamics.

    This function integrates the Mangas-Sanjuan model over time using a
    given drug administration schedule and parameter set. It returns the
    simulated circulating cell counts at specified time points.

    Parameters
    ----------
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    MTT : float
        Mean transit time governing the compartmental delay.
    slope : float
        Slope of the drug effect function per unit dose.
    fprol : float
        Fraction of cells in the proliferative state.
    kcyc : float
        Rate constant governing the cell cycle through proliferative and
        quiescent compartments.
    therapies_tuple : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times
        - an array or list of corresponding dosages
    end_fu : float
        Duration of follow-up (in days) after therapy administration.
    t_eval : list or array of float
        Time points (in days) at which to evaluate the model solution.

    Returns
    -------
    np.ndarray
        Solution for the circulating cells compartment.

    References
    ----------
    See [3] for the original model.
    """

    therapies, dosage = therapies_tuple
    # add therapy time points
    therapies = therapies * 24

    # params
    ktr = 4. / MTT
    y0 = c0 * np.ones(7)
    y0[0] = c0 / fprol
    y0[1:3] = (1 - fprol) * y0[0] * np.ones(2)

    # time arrs
    tmax = np.max(therapies) + end_fu * 24

    # solve friberg
    sol = solve_ivp(MSmodel_drug, (0, tmax), y0, method='LSODA',
                    t_eval=t_eval * 24,
                    args=(
                        ktr, fprol, kcyc, gamma, c0, slope,
                        (therapies, dosage)),
                    rtol=1e-6, atol=1e-9, max_step=24, jac=None)
    return sol.y[-1]


def MSmodel_drug_rev(t: float, y: np.ndarray, ktr: float, fprol: float,
                     kcyc: float, kcyc2: float, gamma: float, c0: float,
                     slope: float, therapies: Tuple[Sequence[float],
                     Sequence[float]]
    ) -> tuple[float, float, float, float, float, float, float]:
    """
    Compute the rate of change for each compartment in the revised
    Mangas-Sanjuan PKPD model.

    This revised version of the model includes:
    - two quiescent compartments (Q1, Q2)
    - an explicit second cycle constant (kcyc2)

    It simulates proliferative, quiescent, transit, and circulating cell
    compartments under drug exposure, using a system of differential
    equations.

    Parameters
    ----------
    t : float
        Current time point.
    y : np.ndarray
        Current state of the system. Contains values for:
        - prol: Proliferating cells
        - q1, q2: Quiescent compartments
        - t1, t2, t3: Transit compartments
        - circ: Circulating cells
    ktr : float
        Rate constant for transitions between compartments.
    fprol : float
        Fraction of cells in the proliferative state.
    kcyc : float
        Cell cycle rate constant.
    kcyc2 : float
        Second cycle constant.
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    slope : float
        Slope of the drug effect function per unit dose.
    therapies : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times
        - an array or list of corresponding dosages

    Returns
    -------
    tuple of float
        Derivatives for each compartment:
        (dprol, dq1, dq2, dt1, dt2, dt3, dcirc)

    References
    ----------
    See [3] for the original model.
    """
    # get vec components
    prol, q1, q2, t1, t2, t3, circ = np.maximum(10**-12 * np.ones(y.shape), y)

    # get Edrug
    Edrug = E_drug_step(t, slope, therapies)
    fb = (c0 / circ) ** gamma
    kprol = ktr * fprol

    # diff eq system
    dprol = kprol * prol * fb * (1. - Edrug) + \
            kcyc2 * kcyc * q2 - ktr * fprol * prol - kcyc * (1. - fprol) * prol
    dq1 = kcyc * (1. - fprol) * prol - kcyc * q1
    dq2 = kcyc * (q1 - kcyc2 * q2)
    dt1 = ktr * fprol * prol - ktr * t1
    dt2 = ktr * t1 - ktr * t2
    dt3 = ktr * t2 - ktr * t3
    dcirc = ktr * t3 - ktr * circ

    return dprol, dq1, dq2, dt1, dt2, dt3, dcirc


def solve_MSmodel_rev(
    gamma: float, c0: float, MTT: float, slope: float, fprol: float,
    kcyc: float, kcyc2: float,
    therapies_tuple: Tuple[Sequence[float], Sequence[float]],
    end_fu: float, t_eval: np.ndarray
    ) -> np.ndarray:
    """
    Solve the revised Mangas-Sanjuan PKPD model for drug-induced effects on
    blood cell dynamics.

    This version incorporates a second quiescent compartment (Q2) and an
    additional cell cycle parameter (`kcyc2`). It returns the simulated
    circulating cell counts over a specified time course under drug
    treatment.

    Parameters
    ----------
    gamma : float
        Power coefficient for the feedback mechanism affecting
        proliferation.
    c0 : float
        Baseline value of circulating cells for feedback control.
    MTT : float
        Mean transit time governing the compartmental delay.
    slope : float
        Slope of the drug effect function per unit dose.
    fprol : float
        Fraction of cells in the proliferative state.
    kcyc : float
        Cell cycle rate constant.
    kcyc2 : float
        Second cycle constant.
    therapies_tuple : tuple of (Sequence[float], Sequence[float])
        Tuple containing:
        - an array or list of therapy start times
        - an array or list of corresponding dosages
    end_fu : float
        Duration of follow-up (in days) after therapy administration.
    t_eval : np.ndarray
        Time points (in days) at which to evaluate the model solution.

    Returns
    -------
    np.ndarray
        Solution for the circulating cells compartment.

    References
    ----------
    See [3] for the original model structure.
    """
    therapies, dosage = therapies_tuple
    # add therapy time points
    therapies = therapies * 24

    # params
    ktr = 4. / MTT
    # steady state
    y0 = c0 * np.ones(7)
    y0[0] = c0 / fprol
    y0[1] = (1 - fprol) * y0[0]
    y0[2] = 1 / kcyc2 * y0[1]

    # time arrs
    tmax = np.max(therapies) + end_fu * 24

    # solve friberg
    sol = solve_ivp(MSmodel_drug_rev, (0, tmax), y0, method='LSODA',
                    t_eval=t_eval * 24,
                    args=(ktr, fprol, kcyc, kcyc2, gamma, c0, slope,
                        (therapies, dosage)),
                    rtol=1e-6, atol=1e-9, max_step=24, jac=None)
    return sol.y[-1]


def poplog_fit_friberg_mse(params: np.ndarray,
    pop_params: np.ndarray,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> float:
    """
    Compute the log-scale mean squared error (MSE) plus parameter loss for
    the Friberg model fit.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0).
    pop_params : np.ndarray
        Prior population-level parameter estimates.
    y_arr : np.ndarray
        Observed data (circulating cells).
    t_arr : np.ndarray
        Time points corresponding to observed data.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    float
        Combined log-MSE and parameter regularization loss.
    """
    gamma, MTT, slope, c0t = params
    c0 = c0t * 10**9
    y_pred = solve_friberg(gamma, c0, MTT, slope, therapies,
                           end_fu=t_arr[-1] +1 - therapies[0][-1],
                           t_eval=t_arr)
    y_finite = np.log10(y_pred)
    mse = np.sum((y_finite[np.isfinite(y_finite)]
                  - np.log10(y_arr)[np.isfinite(y_finite)])**2)/ len(y_arr)
    param_loss = ((np.log(params) - np.log(pop_params))**2) / (5.** 2)
    return mse + np.sum(param_loss)


def poplog_fit_h_mse(
    params: np.ndarray,
    pop_params: np.ndarray,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> float:
    """
    Compute the log-scale MSE and parameter loss for the Henrich model.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0, ftr).
    pop_params : np.ndarray
        Prior population-level parameter estimates.
    y_arr : np.ndarray
        Observed data (circulating cells).
    t_arr : np.ndarray
        Time points corresponding to observed data.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    float
        Combined log-MSE and parameter regularization loss.
    """
    gamma, MTT, slope, c0t, ftr = params
    c0 = c0t * 10**9
    y_pred = solve_henrich(gamma, c0, MTT, slope, ftr, therapies,
                            end_fu = t_arr[-1] +1 - therapies[0][-1], t_eval=t_arr)
    y_finite = np.log10(y_pred)
    mse = np.sum((y_finite[np.isfinite(y_finite)]
                  - np.log10(y_arr)[np.isfinite(y_finite)])**2)/ len(y_arr)
    param_loss = ((np.log(params) - np.log(pop_params))**2) / (5.** 2)
    return mse + np.sum(param_loss)


def poplog_fit_ms_mse(
    params: np.ndarray,
    pop_params: np.ndarray,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> float:
    """
    Compute the log-scale MSE and parameter loss for the Mangas-Sanjuan model.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0, kcyc, fprol).
    pop_params : np.ndarray
        Prior population-level parameter estimates.
    y_arr : np.ndarray
        Observed data (circulating cells).
    t_arr : np.ndarray
        Time points corresponding to observed data.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    float
        Combined log-MSE and parameter regularization loss.
    """
    gamma, MTT, slope, c0t, kcyc, fprol = params
    c0 = c0t * 10**9
    y_pred = solve_MSmodel(gamma, c0, MTT, slope, fprol, kcyc, therapies,
                           end_fu = t_arr[-1] +1 - therapies[0][-1],
                           t_eval=t_arr)
    y_finite = np.log10(y_pred)
    mse = np.sum((y_finite[np.isfinite(y_finite)]
                  - np.log10(y_arr)[np.isfinite(y_finite)])**2)/ len(y_arr)
    param_loss = ((np.log(params) - np.log(pop_params))**2) / (5.** 2)
    return mse + np.sum(param_loss)


def poplog_fit_ms_mse_rev(    params: np.ndarray,
    pop_params: np.ndarray,
    y_arr: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> float:
    """
    Compute the log-scale MSE and parameter loss for the revised
    Mangas-Sanjuan model (with kcyc2).

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0, kcyc, kcyc2, fprol).
    pop_params : np.ndarray
        Prior population-level parameter estimates.
    y_arr : np.ndarray
        Observed data (circulating cells).
    t_arr : np.ndarray
        Time points corresponding to observed data.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    float
        Combined log-MSE and parameter regularization loss.
    """
    gamma, MTT, slope, c0t, kcyc, kcyc2, fprol = params
    #kcyc2 = 1/60 * kcyc
    c0 = c0t * 10**9
    y_pred = solve_MSmodel_rev(gamma, c0, MTT, slope, fprol, kcyc, kcyc2,
                            therapies,
                            end_fu = t_arr[-1] +1 - therapies[0][-1], t_eval=t_arr)
    y_finite = np.log10(y_pred)
    mse = np.sum((y_finite[np.isfinite(y_finite)]
                  - np.log10(y_arr)[np.isfinite(y_finite)])**2)/ len(y_arr)
    param_loss = ((np.log(params) - np.log(pop_params))**2) / (5.** 2)
    return mse + np.sum(param_loss)


def pred_fri(
    params: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> np.ndarray:
    """
    Generate model predictions for circulating cells using the Friberg model.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0).
    t_arr : np.ndarray
        Time points (in days) at which predictions are generated.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    np.ndarray
        Predicted circulating cell counts at each time point.
    """
    gamma, MTT, slope, c0t = params
    c0 = c0t * 10**9
    y_pred = solve_friberg(
        gamma, c0, MTT, slope, therapies, end_fu=t_arr[-1] +1-therapies[0][-1],
        t_eval=t_arr)
    return y_pred


def pred_h(
    params: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> np.ndarray:
    """
    Generate model predictions for circulating cells using the Henrich model.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0, ftr).
    t_arr : np.ndarray
        Time points (in days) at which predictions are generated.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    np.ndarray
        Predicted circulating cell counts at each time point.
    """
    gamma, MTT, slope, c0t, ftr = params
    c0 = c0t * 10**9
    y_pred = solve_henrich(gamma, c0, MTT, slope, ftr, therapies,
                        end_fu = t_arr[-1] +1 - therapies[0][-1], t_eval=t_arr)
    return y_pred


def pred_ms(
    params: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> np.ndarray:
    """
    Generate model predictions for circulating cells using the
    Mangas-Sanjuan model.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0, kcyc, fprol).
    t_arr : np.ndarray
        Time points (in days) at which predictions are generated.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    np.ndarray
        Predicted circulating cell counts at each time point.
    """
    gamma, MTT, slope, c0t,  kcyc, fprol = params
    c0 = c0t * 10**9
    y_pred = solve_MSmodel(gamma, c0, MTT, slope, fprol, kcyc, therapies,
                        end_fu = t_arr[-1] +1 - therapies[0][-1], t_eval=t_arr)
    return y_pred


def pred_ms_rev(
    params: np.ndarray,
    t_arr: np.ndarray,
    therapies: Tuple[Sequence[float], Sequence[float]]
    ) -> np.ndarray:
    """
    Generate model predictions for circulating cells using the revised
    Mangas-Sanjuan model.

    Parameters
    ----------
    params : np.ndarray
        Model parameters: (gamma, MTT, slope, c0, kcyc, kcyc2, fprol).
    t_arr : np.ndarray
        Time points (in days) at which predictions are generated.
    therapies : tuple of (Sequence[float], Sequence[float])
        Therapy start times and corresponding dosages.

    Returns
    -------
    np.ndarray
        Predicted circulating cell counts at each time point.
    """
    gamma, MTT, slope, c0t, kcyc, kcyc2, fprol = params
    c0 = c0t * 10**9
    y_pred = solve_MSmodel_rev(gamma, c0, MTT, slope, fprol, kcyc, kcyc2,
                            therapies,
                        end_fu = t_arr[-1] +1 - therapies[0][-1], t_eval=t_arr)
    return y_pred