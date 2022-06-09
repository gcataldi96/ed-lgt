import numpy as np

# =====================================================================================
def First_derivative(x,f,dx):
    # CHECK TYPES
    if not isinstance(x, np.ndarray):
        raise TypeError(f'x should be an ndarray, not a {type(x)}')
    if not isinstance(f, np.ndarray):
        raise TypeError(f'f should be an ndarray, not a {type(f)}')
    if not np.isscalar(dx) and not isinstance(dx, float):
        raise TypeError(f'dx should be a SCALAR FLOAT, not a {type(dx)}')
    # COMPUTE THE 1st OR THE 2nd DERIVATIVE
    # f_der OF A FUNCTION f WRT A VARIABLE x
    f_der=np.zeros(x.shape[0]-2)
    # COMPUTE THE 1ST ORDER CENTRAL DERIVATIVE
    for ii in range(f_der.shape[0]):
        jj=ii+1
        f_der[ii]=(f[jj+1]-f[jj-1])/dx
    # USE AN UPDATE VERSION OF X WHERE THE FIRST
    # AND THE LAST ENTRY ARE ELIMINATED IN ORDER
    # TO GET AN ARRAY OF THE SAME DIMENSION OF
    # THE ONE WITH THE DERIVATIVE OF F
    x_copy=np.zeros(f_der.shape[0])
    for ii in range(f_der.shape[0]):
        x_copy[ii]=x[ii+1]
    return x_copy, f_der


# =====================================================================================
def Second_derivative(x,f,dx):
    # CHECK TYPES
    if not isinstance(x, np.ndarray):
        raise TypeError(f'x should be an ndarray, not a {type(x)}')
    if not isinstance(f, np.ndarray):
        raise TypeError(f'f should be an ndarray, not a {type(f)}')
    if not np.isscalar(dx) and not isinstance(dx, float):
        raise TypeError(f'dx should be a SCALAR FLOAT, not a {type(dx)}')
    # COMPUTE THE 1st OR THE 2nd DERIVATIVE
    # f_der OF A FUNCTION f WRT A VARIABLE x
    f_der=np.zeros(x.shape[0]-2)
    # COMPUTE THE 2ND ORDER CENTRAL DERIVATIVE
    for ii in range(f_der.shape[0]):
        jj=ii+1
        f_der[ii]=(f[jj+1]-2*f[jj]+f[jj-1])/(dx**2)
    # USE AN UPDATE VERSION OF X WHERE THE FIRST
    # AND THE LAST ENTRY ARE ELIMINATED IN ORDER
    # TO GET AN ARRAY OF THE SAME DIMENSION OF
    # THE ONE WITH THE DERIVATIVE OF F
    x_copy=np.zeros(f_der.shape[0])
    for ii in range(f_der.shape[0]):
        x_copy[ii]=x[ii+1]
    return x_copy, f_der
