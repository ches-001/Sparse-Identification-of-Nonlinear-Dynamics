# SINDy (Sparse Identification of Nonlinear Dynamics)

SINDy is a data driven technique that aims to draft out the governing dynamics / vector field of a nonlinear dynamical system, purely from data.

In many real life scenarios, the dynamics of a dynamical system is unknown, the relevance of identifying nonlinear dynamics of a system purely from data stems from the fact that knowing the governing dynamics can enable us to better understand the system, predict future states and even control the systems behaviour to converge at some desired attractor.

## **Section 1:** Steps to Sparse Identification of Nonlinear Dynamics
<br><br>
1.  ### **Data Collection**
The first step in this process is to collect data from a dynamical system, either real or simulated. A dynamical system is any system whose states vary in time, these systems can also be prone to disturbances, and their behaviours can be altered by actuation parameters. Mathematically, a dynamics of system can be expressed as a differential equation or a system of differential equation, like so:
$$X^{\prime} = F(X_t, t, \alpha; \beta)$$

Where:

$X^{\prime}$ is the derivative of the system. 

$X_t$ is the state at time $t$.

$t$ is the time value.

$\alpha$ is the actuation parameter(s).

$\beta$ is environmental factors / parameters outside our control

<br><br>

2. ### **Identifying and Arranging State Components**
In this step, the dataset collected is used to form column state vectors that will then be combined to matrices. For example, if we had a system whose states change in time, where each state represents a vector in 3 dimensions x, y and z, then we can isolate each dimension of the given states into 3 different column vectors $\bar{x}$, $\bar{y}$, $\bar{z}$, where:

$$\bar{x} =
   \begin{bmatrix}
     x_0 \\
     x_1 \\
     x_2 \\
     .\\
     .\\
     .\\
     x_t
   \end{bmatrix}$$

$$\bar{y} =
   \begin{bmatrix}
     y_0 \\
     y_1 \\
     y_2 \\
     .\\
     .\\
     .\\
     y_t
   \end{bmatrix}$$

$$\bar{z} =
   \begin{bmatrix}
     z_0 \\
     z_1 \\
     z_2 \\
     .\\
     .\\
     .\\
     z_t
   \end{bmatrix}$$

<br><br>

3. ### **Compute the State Variational Derivatives from Data**
To perform SINDy, the derivatives of the given states with respect to time needs to be computed. These derivatives are computed numerically from data because the underlying dynamics / equation(s) that describes the system is unknown and hence cannot be differentiated. The derivatives can be computed in various ways, but in this implementation, we considered 3 finite differencing methods, namely: 

1. Forward Differencing.
2. Backward Differencing. 
3. Center Differencing.

For a recap, given a state value $x_t$ at time $t$, in *Forward Differencing*, the derivative at time $t$ is:
$\Delta x = \frac {x_{t+1} - x_{t}} {\Delta t}$. This method is used towards the begining of the datapoints where the number of points prior to a given timestep is 0 (first timestep in the data), but the number of points after it is more than $h$.

In *Backward Differencing*, $\Delta x = \frac {x_{t} - x_{t-1}} {\Delta t}$. This method is used towards the ending of the datapoints, where the number of points prior to a given timestep is greater than $h$, but the number of points after it is 0 (last timestep in the data).

And in *Center differencing*, $\Delta x = \frac {x_{t+h} - x_{t-h}} {\Delta t}$, where $h$ is the number of points prior or after the given timestep. This method is used in everyother position in time in the data

**NOTE:**
*If the number of points prior or after a given timstep is less than h, but still greater than 0, then the value of h is adjusted (by reducing its value) to compute centered differencing with the available datapoints left and right of the center timestep.*

After the derivatives are computed across each dimension (x, y, z), we form derivative vector columns, $\dot{x}$, $\dot{y}$, $\dot{z}$, where:

$$\dot{x} =
   \begin{bmatrix}
     {x_0}^{\prime} \\
     {x_1}^{\prime} \\
     {x_2}^{\prime} \\
     .\\
     .\\
     .\\
     {x_t}^{\prime}
   \end{bmatrix}$$

$$\dot{y} =
   \begin{bmatrix}
     {y_0}^{\prime} \\
     {y_1}^{\prime} \\
     {y_2}^{\prime} \\
     .\\
     .\\
     .\\
     {y_t}^{\prime}
   \end{bmatrix}$$

$$\dot{z} =
   \begin{bmatrix}
     {z_0}^{\prime} \\
     {z_1}^{\prime} \\
     {z_2}^{\prime} \\
     .\\
     .\\
     .\\
     {z_t}^{\prime}
   \end{bmatrix}$$

<br><br>

4. ### **Formulate State and Derivative Matrices**
The column vectors for states ($\bar{x}$, $\bar{y}$, $\bar{z}$) and derivatives ($\dot{x}$, $\dot{y}$, $\dot{z}$) are combined to form matrices $X$ and $Y$, where:


$$X =
   \begin{bmatrix}
     x_0, &y_0, &z_0 \\
     x_1 &y_1, &z_1 \\
     x_2, &y_2, &z_2 \\
     . &. &.\\
     . &. &.\\
     . &. &.\\
     x_t, &y_t, &z_t
   \end{bmatrix}$$

$$Y =
   \begin{bmatrix}
     {x_0}^{\prime}, &{y_0}^{\prime}, &{z_0}^{\prime} \\
     {x_1}^{\prime} &{y_1}^{\prime}, &{z_1}^{\prime} \\
     {x_2}^{\prime}, &{y_2}^{\prime}, &{z_2}^{\prime} \\
     . &. &.\\
     . &. &.\\
     . &. &.\\
     {x_t}^{\prime}, &{y_t}^{\prime}, &{z_t}^{\prime}
   \end{bmatrix}$$

<br><br>

5. ### **Compute Polynomial Features for X**
if we attempt to extimate $Y$ (derivatives matrix) from $X$ (states matrix), then we will do so with some linear operator $\beta$, such that:

$Y = X \beta$.

This is a linear regression problem, and the matrix $X$ is ought to be a matrix of polynomial features computed from the original $X$ matrix formed by the column vectors $\bar{x}$, $\bar{y}$, $\bar{z}$.

In this implementation, we compute the polynomial features of the $X$ matrix from 0 order up to the nth order ($2 \leq n \leq \infty$), to get columns vectors such as: column of 1s, $\bar{x}$, $\bar{y}$, $\bar{z}$, $\bar{x}^2$, $\bar{x} \bar{y}$, $\bar{x} \bar{z}$, $\bar{y} \bar{z}$, $\bar{y}^2$, $\bar{z}^2$, ....$\bar{z}^5$

after computing the polynomial features, the matrix $X$ is overwritten to becomes: 

$$X =
   \begin{bmatrix}
     1, &x_0, &y_0, &z_0, &{x_0}^2, &. &. &. &. &,{z_0}^5 \\
     1, &x_1, &y_1, &z_1, &{x_1}^2, &. &. &. &. &,{z_1}^5\\
     1, &x_2, &y_2, &z_2, &{x_2}^2, &. &. &. &. &,{z_2}^5\\
     . &. &. &. &. &. &. &. &.\\
     . &. &. &. &. &. &. &. &.\\
     . &. &. &. &. &. &. &. &.\\
     1, &x_t, &y_t, &z_t, &{x_t}^2, &. &. &. &. &,{z_t}^5
   \end{bmatrix}$$


<br><br>

6. ### **Estimate the Linear Operator $\beta$ with Sparse Regression**

Given the linear Algebraic Equation $Y = X \beta$, we could easily estimate the best value of the matrix $\beta$, by:
1. computing $X$ transpose ($X^T$) and multiplying it by both sides such that: 
$X^T Y = X^T X \beta$

2. computing the inverse of the square matrix $X^T X$ and multiplying it to both sides, such that:
$(X^T X)^{-1} X^T Y = (X^T X)^{-1} X^T X \beta$. Now, if
$(X^T X)^{-1} X^T = I$, where $I$ is an identity matrix, then:
$\beta = (X^T X)^{-1} X^T Y$

Although this value of $\beta$ is one that best minimizes the least square error between the estimated derivates and the numerically computed derivates, consider a situation where the number of features in the matrix X is much, inotherwords;  if the sizes of matrix $X$ and $Y$ are (n x p) and (n x 3) respectively, we would expect that the size of the linear operator $\beta$ is (p x 3). If p happens to be a large number and the $\beta$ matrix happens to be dense, it would be quite difficult to estimate the governing nonlinear dynamics of the system. What we want is a technique that reduces the density of the linear operator $\beta$, to make it more sparse so as to filter out or remove the irrelevant parameters, and in turn leave only the relevant polynomials that best estimate the numerically computed derivates $Y$. This way, we can better understand the dynamics of the system with as minimum parameters and dependencies possible.

To achieve this sparse linear operator, we utilise a sparse regression algorithm called LASSO (Least Absolute Shrinkage and Selection Operator). This is an iterative regression technique where the objective is:

 $argmin ||Y^{\prime} - X \beta||_2 + \alpha ||\beta||$. 
 
 where: $0 \leq \alpha \leq 1$

 The difference between this and the regular iterative linear regression technique is the regularisation term $\alpha ||\beta||$. This regularisation term helps to ensure that the weighted norm of the linear operator $\beta$ is minimized alongside the "two norm" of the difference between the estimated and the actual derivates. This way, the linear operator becomes a sparse matrix.


<br><br>

6. ### **Identifying The Dynamics**
After computing the sparse linear operator $\beta$, we get the index of the zero rows, we remove these zero rows and correspond these indexes to the columns of the polynomial feature matrix $X$, each column that corresponds to any of the index is removed from matrix $X$, this way the number of features that correspond to a given derivative is reduced.

to Illustrate with a toy example:

let:
$$X =
   \begin{bmatrix}
     x_{00}, &x_{01}, &x_{02}, &x_{03}, &x_{04}\\
     x_{10}, &x_{11}, &x_{12}, &x_{13}, &x_{14}\\
     x_{20}, &x_{21}, &x_{22}, &x_{23}, &x_{24}\\
     x_{30}, &x_{31}, &x_{32}, &x_{33}, &x_{34}\\
   \end{bmatrix}$$

$$Y =
   \begin{bmatrix}
     y_{00}, &y_{01}, &y_{02}\\
     y_{10}, &y_{11}, &y_{12}\\
     y_{20}, &y_{21}, &y_{22}\\
     y_{30}, &y_{31}, &y_{32}\\
   \end{bmatrix}$$

if we computed the sparse linear operator $\beta$ to be:

$$\beta =
   \begin{bmatrix}
     \beta_{00}, &0, &\beta_{02}\\
     \beta_{10}, &\beta_{11}, &0\\
     0, &0, &0\\
     0, &0, &0\\
     0, &\beta_{41}, &0\\
   \end{bmatrix}$$

Then, the indexes corresponding to zero rows are index 2 and index 3 (indexing begins at 0), hence we can remove columns 2 and 3 from matrix $X$, as well as rows 2 and 3 from matrix $\beta$ to form SINDy matrices, such that:

$$X =
   \begin{bmatrix}
     x_{00}, &x_{01}, &x_{04}\\
     x_{10}, &x_{11}, &x_{14}\\
     x_{20}, &x_{21}, &x_{24}\\
     x_{30}, &x_{31}, &x_{34}\\
   \end{bmatrix}$$

and

$$\beta =
   \begin{bmatrix}
     \beta_{00}, &0, &\beta_{02}\\
     \beta_{10}, &\beta_{11}, &0\\
     0, &\beta_{41}, &0\\
   \end{bmatrix}$$

Therefore:

$$
   \begin{bmatrix}
     y_{00}, &y_{01}, &y_{02}\\
     y_{10}, &y_{11}, &y_{12}\\
     y_{20}, &y_{21}, &y_{22}\\
     y_{30}, &y_{31}, &y_{32}\\
   \end{bmatrix} = 
\begin{bmatrix}
     x_{00}, &x_{01}, &x_{04}\\
     x_{10}, &x_{11}, &x_{14}\\
     x_{20}, &x_{21}, &x_{24}\\
     x_{30}, &x_{31}, &x_{34}\\
   \end{bmatrix} * 
   \begin{bmatrix}
     \beta_{00}, &0, &\beta_{02}\\
     \beta_{10}, &\beta_{11}, &0\\
     0, &\beta_{41}, &0\\
   \end{bmatrix}
$$

From the linear matrix combination above, we know that:

$y_{00} = x_{00} \beta_{00} + x_{01} \beta_{10}$

$y_{01} = x_{01} \beta_{11} + x_{04} \beta_{41}$

and

$y_{02} = x_{00} \beta_{02}$

If the columns of these matrices indicated any variable symbols, we can simply infer a relationship between these variables in a similar manner, for example: If columns 0, 1 and 2 of matrix $Y$ correspond to $\dot{x}$, $\dot{y}$, $\dot{z}$ respectively, and if the columns of matrix $X$ correspond to $\bar{x}$, $\bar{y}$, $\bar{z}$ respectively, while the columns of the linear operator $\beta$ correspond to $\eta$, $\psi$ and $\epsilon$ respectively, then we can identify that:

$\dot{x} = \bar{x}\eta + \bar{y}\eta$ => $\dot{x} = \eta(\bar{x} + \bar{y})$

$\dot{y} = \bar{y}\eta + \bar{z}\psi$

$\dot{z} = \bar{x}\psi$


<br>

**Note:** 
*For data collected from a simulated system, we can also introduce an extra dimension corresponding to simulated control input to ensure some type of forcing on the system. The control input can typically be of any type, like a periodic function input or anyother type of function at all, depending on the application*

<br><br>

## **Section 2:**  Code Example
Now we have gone through the steps required for Sparse Identification of Nonlinear Dynamics, let us simply take a code example with helper functions from the [sindy.py](https://github.com/ches-001/Sparse-Identification-of-Nonlinear-Dynamics/blob/main/sindy.py) file from this repository. In this example, we will attempt to identify the nonlinear dynamics of a lorenz attractor with sparse linear regression.

1. ### **Simulating a Lorenz Attractor**

A Lorenz attractor is a dynamical system of three dimensional space $x$, $y$ and $z$. The dynamics of a Lorenz Attractor is given by the ODE:

$\dot{X} = f(t, X)$

 where:
 
$\dot{X}$ is a 3D vector $[\dot{x}, \dot{y}, \dot{z}]$

Where:
$\dot{x} = \sigma(y - x)$

$\dot{y} = x(\rho - z) - y$

$\dot{z} = xy - \beta z$

The following code simulates a lorenz system for 10,000 timesteps from $t=0$ to $t=50$:

```python
import numpy as np
from sindy import simulate_lorenz_attractor

# system parameters
sigma = 10.0
rho = 28.0
beta = 8/3
params = np.array([sigma, rho, beta])

# state => [x, y, z]
X0 = np.array([0, 1, 20])

# timesteps
n_steps = 10000
t = np.linspace(0, 50, n_steps)

# simulated states and timsteps data
X, t = simulate_lorenz_attractor(
    initial_state=X0, 
    timesteps=t, 
    params=params
)
```

In the code snippet above, the `states` variable is a matrix of size (10000 x 3), and the `timesteps` variable is a vector of size (10000, ).

For the simulation to work, we needed to define some initial state, and numerically integrate the lorenz dynamics (ODE) at this initial state, over all given timesteps. To do this we employ the *Runge Kutta mumerical integration method* for integrating ODEs. In this mothod, given a function:

$\dot{X}_t = f(t, X_t)$

The integral of this function can be approximated numerically as:

$Y_t = Y_{t-1} + \frac {\Delta t}6 (k_1 + 2*k_2 + 2*k_3 + k_4)$

Where:

$k_1 = f(t, Y_t)$

$k_2 = f(t + dt/2, Y_t + (dt/2 * k_1))$

$k_2 = f(t + dt/2, Y_t + (dt/2 * k_2))$

$k_3 = f(t + dt, Y_t + (dt * k_3))$

This technique is called the fourth-order Runge Kutta Method. It is called "fourth-order" because this numerical method establishes a fourth order of accuracy in its numerical computation.


<br><br>

2. ### **Computing Derivatives**

To compute the derivatives, we use the functions below:

```python
from sindy import lorenz_ode,  compute_derivatives

# numerically computed derivatives of system
numerical_derivatives = compute_derivatives(
    data=X, 
    timesteps=t,
    h=1
)

# actual derivatives of system
actual_derivatives = np.array([
    lorenz_ode(t, x, sigma, rho, beta) for t, x in zip(t, X)
])
```

The `compute_derivatives` function computes the derivatives using finite differencing as explained in **Section 1**

**Note:** *`h` represents the number of leftward and rightward points to use for center differencing.*

<br><br>

3. ### **Computing Polynomial Features from States Matrix**

The polynomial features are computed as shown in the snippet below:

```python
from sindy import compute_3d_features

X_poly = compute_poly_features(
    ode_func=lorenze_ode,
    data=X_U,
    degree=2, 
    column_names=["X", "Y", "Z"]
)
```
This function computes the polynomial features with the sci-kit-learn's `preprocessing.PolynomialFeatures().fit_transform()` method.

The `X_poly` variable is a dataframe of polynomial features corresponding to multiplicative combinations of $x$, $y$, and $z$ up to 2 degrees. The column names of the dataframe correspond to each polynomial combination like so: ['1', 'X', 'Y', 'Z', 'X^2', 'X Y', 'X Z', . . . . ,Z^2]

<br><br>

4. ### **Computing Sparse Linear Operator ($\beta$)**

The sparse linear operator matrix can be computed as follows

```python
# compute sparse coeffs
from sindy import compute_linear_operator

sindy_coeffs, selected_features = compute_linear_operator(
    poly_features=X_poly.values,
    derivatives=numerical_derivatives,
    alpha=0.2,
    max_iter=1000,
    max_features=8,
    n_cv=n_cv,
    threshold=threshold,
    scaler=scaler
)

BETA = sindy_coeffs.T.round(3)
```

The `compute_linear_operator` function uses the sci-kit-learn's `MultiTaskLassoCV` and the `SelectFromModel` classes to select the relevant polynomial features to compute the relevant coefficients via sci-kit-learn's `LinearRegression` class. You can further decide to introduce sparsity to the optimum linear operator matrix by rounding to the nearest n digits as in the last line of the snippet above, this will round very small numbers up or down to 0 and introduce more sparsity.

**Note:** *Although the `scaler` parameter is not necessary and can be set to `None` in the function above, the linear regression converges faster and better when the dataset is normalised or scaled to within a given range. The scaler parameter if not `None`, is expected to be a scaler class from the sci-kit-learn module, such as `MinMaxScaler` or `StandardScaler`*


The `compute_linear_operator` function also returns an array of integers corresponding to indexes of the relevant polynomial columns, we can filter out only the relevant features with this as shown in the snippet below:

```python
relevant_cols = X_poly.columns[selected_features]
X_prime = X_poly[relevant_cols]
```

<br><br>

5. ### **Use The Computed Linear Operator to Compute to Estimate Derivatives**

After computing the linear operator, you can attempt to estimate the derivate like so:

```python
estimated_derivatives = X_prime @ BETA
```

With this, you can verify that it is an approximate version of the original derivative that was computed numerically.

<br><br>

Check out the  [main.ipynb](https://github.com/ches-001/Sparse-Identification-of-Nonlinear-Dynamics/blob/main/main.ipynb) notebook for further information on selecting the optimum hyper-parameters and adding forcing to the system with periodic control inputs.
