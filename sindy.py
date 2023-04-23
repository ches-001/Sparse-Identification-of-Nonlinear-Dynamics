import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import MultiTaskLassoCV, LinearRegression
from sklearn.feature_selection import SelectFromModel
from typing import Tuple, Any, Iterable, Dict, Optional


def lorenz_ode(
        t: float, 
        X: np.ndarray, 
        sigma: float, 
        rho: float, 
        beta: float)->np.ndarray:
    
    r"""
    ODE (Ordinary Differential Equation) of the lorenz
    attractor is given as:

    dX/dt = [dx/dt, dy/dt, dz/dt], where:

    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = (x * y) - (beta * z)

    Parameters
    ----------------
    t: (float)
        continuous time value for the given state.
    X: (ndarray):
        3D state vector
    sigma: (float)
        parameter used to compute dx/dt
    rho: (float)
        parameter used to compute dy/dt
    beta: (float)
        parameter used to compute dz/dt

    Return
    ----------------
    out: (ndarray)
        Array of x, y, z derivatives of a given state
    """
    
    x, y, z = X
    X_deriv = np.zeros_like(X)
    X_deriv[0] = sigma * (y - x)
    X_deriv[1] = x * (rho - z) - y
    X_deriv[2] = (x * y) - (beta * z)

    return X_deriv


def rossler_ode(
        t: float, 
        X: np.ndarray, 
        a: float, 
        b: float, 
        c: float)->np.ndarray:
    
    r"""
    ODE (Ordinary Differential Equation) of the rossler
    attractor is given as:

    dX/dt = [dx/dt, dy/dt, dz/dt], where:

    dx/dt = -y - z
    dy/dt = x + ay
    dz/dt = b + z(x - c)

    Parameters
    ----------------
    t: (float)
        continuous time value for the given state.
    X: (ndarray):
        3D state vector
    a: (float)
        constant parameter used to compute state of attractor
    b: (float)
        constant parameter used to compute state of attractor
    c: (float)
        constant parameter used to compute state of attractor

    Return
    ----------------
    out: (ndarray)
        Array of x, y, z derivatives of a given state
    """
    
    x, y, z = X
    X_deriv = np.zeros_like(X)
    X_deriv[0] = -y - z
    X_deriv[1] = x + (a * y)
    X_deriv[2] = b + (z * (x - c))

    return X_deriv


def runge_kutta_ode(
        func: Any, 
        y: np.ndarray, 
        t: float,
        dt: float, 
        params:np.ndarray)->Tuple[float, np.ndarray]:
    
    r"""
    using Runge Kutta method to integrate the ODE of 
    the lorenz attractor:

    Giving a function y = f(t, x)
    y_t = y_{t-1} + dt/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)
    Where: 
    k_1 = f(t, y_t)
    k_2 = f(t + dt/2, y_t + (dt/2 * k_1))
    k_2 = f(t + dt/2, y_t + (dt/2 * k_2))
    k_3 = f(t + dt, y_t + (dt * k_3))

    Parameter
    ----------------
    func: (Any)
        ODE function to integrate numerncally
    y: (ndarray)
        state vector
    t: (float)
        timestep value for corresponding state vector
    dt: (float):
        time difference between immediate timesteps
    params: (ndarray):
        vector of constant parameters: (sigma, rho, beta).

    Return
    ----------------
    out: Tuple[float, ndarray]
        Tuple of time and corresponding state vector
    """

    k_1 = func(t, y, *params)
    k_2 = func(t + dt/2, y + (dt/2 * k_1), *params)
    k_3 = func(t + dt/2, y + (dt/2 * k_2), *params)
    k_4 = func(t + dt, y + (dt * k_3), *params)

    y = y + (dt/6 * (k_1 + 2*k_2 + 2*k_3 + k_4))

    return t, y


def simulate_3d_attractor(
        ode_func: Any,
        initial_state: np.ndarray, 
        timesteps: np.ndarray, 
        params: np.ndarray)->Tuple[np.ndarray, np.ndarray]:
    
    r"""
    Simulates the Chaotic attractor

    Parameters
    ----------------
    ode_func: (Any)
        Ordinary Differential Equation of Attractor
    initial_state: (ndarray)
        Initial 3D state value vector of the system
    timesteps: (ndarray)
         array of timesteps where each state in the data occured.
    params: (ndarray)
        constant parameters used to simulate the 3d chaotic attractor 
        (param1, param2, param3).

    Return
    ----------------
    out: Tuple[ndarray, ndarray]
        Tuple of time and corresponding state vectors array
    """
    dt = timesteps[1] - timesteps[0]
    states_array, timesteps_array = [], []
    y = initial_state


    for t in timesteps:
        if t == 0:
            states_array.append(y)
            timesteps_array.append(0)
            continue

        t, y = runge_kutta_ode(
            func=ode_func, y=y, t=t, dt=dt, params=params)
        
        states_array.append(y)
        timesteps_array.append(t)

    states_array, timesteps_array = np.array(states_array), np.array(timesteps_array)
    return states_array, timesteps_array


def compute_derivatives(
        data: np.ndarray, 
        timesteps:np.ndarray, 
        h: int=1)->np.ndarray:
    
    r"""
    Numerically compute the time derivative of the system via finite differece
    of neighbouring points: 
    |(x_{t+h} - (x_{t}) - (x_{t}) - x_{t-h})| = |(x_{t+h} - x_{t-h})|
    Parameters

    Parameters
    ----------------
    data: (ndarray)
        states data collected from the simulated attractor.
    timesteps: (ndarray)
        array of timesteps where each state in the data occured.
    h: (int)
        number of leftward and rightward points to use for center
        differencing.

    Return
    ----------------
    out: ndarray
        array of derivative column vectors for x, y and z
    """
    N = len(data)
    derivatives = np.zeros_like(data)
    
    for i in range(0, N):
        if i + h >= N: n = N - i - 1
        elif i - h < 0: n = i
        else: n = h

        if i == 0:
            delta_x = (
                (data[i+h] - data[i]) /
                (timesteps[i+h] - timesteps[i])
            )
            
        elif i == N - 1:
            delta_x = (
                (data[i] - data[i-h]) /
                (timesteps[i] - timesteps[i-h])
            )

        else:
            delta_x = (
                (data[i+n] - data[i-n]) /
                (timesteps[i+n] - timesteps[i-n])
            )

        derivatives[i] = delta_x

    return derivatives


def compute_poly_features(
        data: np.ndarray, 
        column_names: Iterable[str],
        degree: int=2)->pd.DataFrame:
    
    r"""
    Compute all possible polynomial features from the given data
    up to the degree provided, for SINDy (Sparse Identification of 
    Nonlinear Dynamics)

    Parameters
    ----------------
    data: (ndarray)
        Data collected from the simulated lorenz attractor.
    degree: (int)
        Maximum degree of polynomial features.

    Return
    ----------------
    out: DataFrame
        Dataframe containing all polynomial features
    """
    
    data = pd.DataFrame(data=data, columns=column_names)

    polyfitter = PolynomialFeatures(degree=degree, include_bias=True)

    features = polyfitter.fit_transform(data)
    poly_cols = polyfitter.get_feature_names_out(data.columns)
    features = pd.DataFrame(data=features, columns=poly_cols)
    return features


def compute_linear_operator(
        poly_features:np.ndarray,
        derivatives: np.ndarray,
        alpha: float=0.5, 
        max_iter: int=1000, 
        max_features: int=8, 
        n_cv: int=5, 
        threshold: float=-np.inf,
        scaler: Optional[Any]=None

        )->Tuple[np.ndarray, np.ndarray]:
    
    r"""
    Compute the sparse coefficents via sparse 
    linear regression between the polynomial features and the
    derivatives. In this function, the LASSO algorithm is used
    to perform sparse regression. in the lasso algorithm, the
    objective is to minimize: ||y - (X * b)||_2 + alpha ||b||
    Where:

    y is the target vector / array of vectors (the derivatives)
    X is the polynomial features used for sparse regression
    b is the sparse coefficients
    alpha is a hyperparameter for regualarisation

    In this function, the sci-kit-learn's MultitaskLassoCV is used
    and the relevant features are selected via sci-kit-learn's 
    SelectFromModel class

    Parameters
    ----------------
    poly_features: (ndarray)
        Polynomial features for computing coefficients.
    derivatives: (ndarray)
        Maximum degree of polynomial features.
    alpha: (float)
        Hyperparameter used for L1 parameter regularisation
    max_iter: (int)
        Maximun number of optimisation iterations for the 
        sparse linear regression
    max_features: (int)
        maximum number of features to select from the poly_features
    n_cv: (int)
        number of cross validations
    threshold: (int)
        feature selection threshold (feature relevance >= threshold)
    scaler: (Any)
        Sklearn Scaler class with 'fit_transform(...)' method

    Return
    ----------------
    out: Tuple[ndarray, ndarray]
        coefficients of regression and index of selected features
    """
    
    condition = list(
        map(lambda x: isinstance(x, np.ndarray), [poly_features, derivatives]),
    )

    assert condition[0], f"poly_features is expected to be ndarray, {type(poly_features)} given"
    assert condition[1], f"derivatives is expected to be ndarray, {type(derivatives)} given"
    
    X = poly_features
    Y = derivatives

    if scaler:
        X = X[:, 1:]
        X = scaler.fit_transform(X)
        Y = scaler.fit_transform(Y)

        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)

    base_estimator = MultiTaskLassoCV(
        cv=n_cv, 
        alphas=[alpha]*n_cv, 
        max_iter=max_iter
    )

    selector = SelectFromModel(
        base_estimator, 
        threshold=threshold, 
        max_features=max_features
    )

    selector.fit(X, Y)
    feature_mask = selector.get_support()
    selected_features = np.where(feature_mask == True)[0]

    X = selector.transform(X)

    estimator = LinearRegression()
    estimator.fit(X, Y)

    return estimator.coef_, selected_features


def optimize_objective(
        trial: Any, 
        poly_features: np.ndarray,
        derivatives: np.ndarray,
        alpha_space: Tuple[float, float]=(0.1, 0.5),
        max_iter_space: Tuple[int, int]=(1000, 1500),
        max_features_space: Tuple[int, int]=(5, 10), 
        n_cv: int=5, 
        threshold: float=-np.inf,
        scaler: Optional[Any]=None,

        )->float:
    
    r"""
    Objective Function for hyper-parameter tuning

    Parameters
    ----------------
    trial: (Any)
        current optimization trial state
    poly_features: (ndarray)
        Polynomial features for computing coefficients.
    derivatives: (ndarray)
        Maximum degree of polynomial features.
    alpha_space: (Tuple[float, float])
        Search range of alpha hyperparameter used for L1 parameter regularisation
    max_iter_space: (Tuple[int, int])
        Search range of maximun number of optimisation iterations for the 
        sparse linear regression
    max_features_space: (int)
        Search range of maximum number of features to select from the poly_features
    n_cv: (int)
        number of cross validations
    threshold: (int)
        feature selection threshold (feature relevance >= threshold)
    scaler: (Any)
        Sklearn Scaler class with 'fit_transform(...)' method   


    Return
    ---------------- 
    out: (float)
        Objective to maximize of minimize.
        L2 Norm of difference between estimated derivatives and actual dervatives.
    """
    
    alpha = trial.suggest_float("alpha", *alpha_space, log=False)
    max_iter = trial.suggest_int("max_iter", *max_iter_space, log=False)
    max_features = trial.suggest_int("max_features", *max_features_space, log=False)

    sindy_coeffs, selected_features = compute_linear_operator(
        poly_features=poly_features,
        derivatives=derivatives,
        alpha=alpha,
        max_iter=max_iter,
        scaler=scaler,
        max_features=max_features,
        n_cv=n_cv,
        threshold=threshold
    )

    relevant_features = poly_features[:, selected_features]

    if scaler:
        relevant_features = scaler.fit_transform(relevant_features)
        derivatives = scaler.fit_transform(derivatives)
    
    estimates = relevant_features @ sindy_coeffs.T

    norm2 = np.linalg.norm(estimates - derivatives)**2
    return norm2