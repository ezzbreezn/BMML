import numpy as np
from scipy.signal import fftconvolve

def calculate_log_probability(X, F, B, s):
    """
    Calculates log p(X_k|d_k,F,B,s) for all images X_k in X and
    all possible displacements d_k.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.

    Returns
    -------
    ll : array, shape(H-h+1, W-w+1, K)
        ll[dh,dw,k] - log-likelihood of observing image X_k given
        that the villain's face F is located at displacement (dh, dw)
    """
    norm_diff = np.sum((X - B[:, :, None]) ** 2, axis=(0, 1), keepdims=True)
    F_norm = np.sum(F ** 2)
    res = fftconvolve(X * B[:, :, None], np.ones((F.shape[0], F.shape[1], 1)), axes=(0, 1), mode='valid')
    res -= fftconvolve(X , np.flip(F, axis=(0, 1)).reshape((F.shape[0], F.shape[1], 1)), axes=(0, 1), mode='valid')
    B_sq = fftconvolve((B ** 2)[:, :, None], np.ones((F.shape[0], F.shape[1], 1)), axes=(0, 1), mode='valid')
    ll = (-1. / (2 * s ** 2)) * (2 * res - B_sq + norm_diff + F_norm)
    ll -= (X.shape[0] * X.shape[1] * 0.5 * np.log(2 * np.pi) + X.shape[0] * X.shape[1] * np.log(s))
    return ll



def calculate_lower_bound(X, F, B, s, A, q, use_MAP=False):
    """
    Calculates the lower bound L(q,F,B,s,A) for the marginal log likelihood.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    L : float
        The lower bound L(q,F,B,s,A) for the marginal log likelihood.
    """
    ll = calculate_log_probability(X, F, B, s)
    ll += np.log(A[:, :, None] + 1e-12)
    L = 0
    if use_MAP:
        for k in range(X.shape[2]):
            L += ll[int(q[0, k]), int(q[1, k]), k]
    else:
        L = np.sum(q * (ll - np.log(q + 1e-12)))
    return L


def run_e_step(X, F, B, s, A, use_MAP=False):
    """
    Given the current esitmate of the parameters, for each image Xk
    esitmates the probability p(d_k|X_k,F,B,s,A).

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    F  : array_like, shape(h, w)
        Estimate of villain's face.
    B : array shape(H, W)
        Estimate of background.
    s : scalar, shape(1, 1)
        Eestimate of standard deviation of Gaussian noise.
    A : array, shape(H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    use_MAP : bool, optional,
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    q : array
        If use_MAP = False: shape (H-h+1, W-w+1, K)
            q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
            of villain's face given image Xk
        If use_MAP = True: shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    """
    numerator = calculate_log_probability(X, F, B, s) 
    numerator += np.log(A[:, :, None] + 1e-12)
    max_val = np.max(numerator, axis=(0, 1), keepdims=True)
    numerator = np.exp(numerator - max_val)
    q = numerator / np.sum(numerator, axis=(0, 1), keepdims=True)
    if use_MAP:
        q_MAP = np.zeros((2, X.shape[2]))
        for k in range(X.shape[2]):
            idx = np.unravel_index(np.argmax(q[:, :, k]), (q.shape[0], q.shape[1]))
            q_MAP[0, k], q_MAP[1, k] = idx[0], idx[1]
        return q_MAP
    else:
        return q


def run_m_step(X, q, h, w, use_MAP=False):
    """
    Estimates F,B,s,A given esitmate of posteriors defined by q.

    Parameters
    ----------
    X : array, shape(H, W, K)
        K images of size H x W.
    q  :
        if use_MAP = False: array, shape (H-h+1, W-w+1, K)
           q[dh,dw,k] - estimate of posterior of displacement (dh,dw)
           of villain's face given image Xk
        if use_MAP = True: array, shape (2, K)
            q[0,k] - MAP estimates of dh for X_k
            q[1,k] - MAP estimates of dw for X_k
    h : int
        Face mask height.
    w : int
        Face mask width.
    use_MAP : bool, optional
        If true then q is a MAP estimates of displacement (dh,dw) of
        villain's face given image Xk.

    Returns
    -------
    F : array, shape (h, w)
        Estimate of villain's face.
    B : array, shape (H, W)
        Estimate of background.
    s : float
        Estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        Estimate of prior on displacement of face in any image.
    """
    H, W, K = X.shape
    #Use full q distribution
    if use_MAP:
        q_full = np.zeros((H - h + 1, W - w + 1, K))
        for k in range(X.shape[2]):
            q_full[int(q[0, k]), int(q[1, k]), k] = 1.
    else:
        q_full = q

    #A calculation:
    A = np.mean(q_full, axis=2)
    #F calculation:
    q_for_F = np.copy(q_full)
    q_for_F = np.flip(q_for_F, axis=(0, 1))
    temp = fftconvolve(X, q_for_F, axes=(0, 1), mode='valid')
    F = np.sum(temp, axis=2) / K
    #B calculation:
    temp = np.ones(X.shape) - fftconvolve(q_full, np.ones((h, w, 1)), axes=(0, 1), mode='full')
    denominator = np.sum(temp, axis=2)
    idx = np.where(denominator != 0)
    B = np.zeros((H, W))
    B[idx] = np.sum(X * temp, axis=2)[idx] / denominator[idx]
    #s calculation
    norm_diff = np.sum((X - B[:, :, None]) ** 2, axis=(0, 1), keepdims=True)
    F_norm = np.sum(F ** 2)
    res = fftconvolve(X * B[:, :, None], np.ones((h, w, 1)), axes=(0, 1), mode='valid')
    res -= fftconvolve(X , np.flip(F, axis=(0, 1)).reshape((F.shape[0], F.shape[1], 1)), axes=(0, 1) ,mode='valid')
    B_sq = fftconvolve((B ** 2)[:, :, None], np.ones((F.shape[0], F.shape[1], 1)), axes=(0, 1), mode='valid')
    s = np.sqrt(np.sum(q_full * (2 * res - B_sq + norm_diff + F_norm)) / (H * W * K) + 1e-12)
    return F, B, s, A
    
        


def run_EM(X, h, w, F=None, B=None, s=None, A=None, tolerance=0.001,
           max_iter=50, use_MAP=False, init_type='norm'):
    """
    Runs EM loop until the likelihood of observing X given current
    estimate of parameters is idempotent as defined by a fixed
    tolerance.

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    F : array, shape (h, w), optional
        Initial estimate of villain's face.
    B : array, shape (H, W), optional
        Initial estimate of background.
    s : float, optional
        Initial estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1), optional
        Initial estimate of prior on displacement of face in any image.
    tolerance : float, optional
        Parameter for stopping criterion.
    max_iter  : int, optional
        Maximum number of iterations.
    use_MAP : bool, optional
        If true then after E-step we take only MAP estimates of displacement
        (dh,dw) of villain's face given image Xk.

    Returns
    -------
    F, B, s, A : trained parameters.
    LL : array, shape(number_of_iters,)
        L(q,F,B,s,A) after each EM iteration (1 iteration = 1 e-step + 1 m-step); 
        number_of_iters is actual number of iterations that was done.
    """
    max_val = np.max(X)
    H, W, K = X.shape

    if F is None:
        if init_type == 'norm':
            F = max_val * np.abs(np.random.randn(h, w))
        elif init_type == 'uniform':
            F = max_val * np.random.uniform(size=(h, w))
    if B is None:
        if init_type == 'norm':
            B = max_val * np.abs(np.random.randn(H, W))
        elif init_type == 'uniform':
            B = max_val * np.random.uniform(size=(H, W))
        elif init_type == 'av_norm':
            B = np.mean(X, axis=2)
            F = np.max(B) * np.abs(np.random.randn(h, w))
        elif init_type == 'av_norm_globF':
            B = np.mean(X, axis=2)
            F = max_val * np.abs(np.random.randn(h, w))
        elif init_type == 'av_uniform':
            B = np.mean(X, axis=2)
            F = np.max(B) * np.random.uniform(size=(h, w))
        elif init_type == 'av_uniform_globF':
            B = np.mean(X, axis=2)
            F = max_val * np.random.uniform(size=(h, w))
    if s is None:
        s = np.mean(np.std(X, axis=(0,1)))
    if A is None:
        A = np.random.uniform(size=(H - h + 1, W - w + 1))
        A /= np.sum(A)
    LL = []
    q = run_e_step(X, F, B, s, A, use_MAP)
    L_t1 = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
    L_t = -np.inf
    iter_num = 0
    while iter_num < max_iter and L_t1 - L_t > tolerance:
        q = run_e_step(X, F, B, s, A, use_MAP)
        F, B, s, A = run_m_step(X, q, h, w, use_MAP)
        L_t = L_t1
        L_t1 = calculate_lower_bound(X, F, B, s, A, q, use_MAP)
        LL.append(L_t1)
        iter_num += 1
    return F, B, s, A, LL
    


def run_EM_with_restarts(X, h, w, tolerance=0.001, max_iter=50, use_MAP=False,
                         n_restarts=10, init_type='norm'):
    """
    Restarts EM several times from different random initializations
    and stores the best estimate of the parameters as measured by
    the L(q,F,B,s,A).

    Parameters
    ----------
    X : array, shape (H, W, K)
        K images of size H x W.
    h : int
        Face mask height.
    w : int
        Face mask width.
    tolerance, max_iter, use_MAP : optional parameters for EM.
    n_restarts : int
        Number of EM runs.

    Returns
    -------
    F : array,  shape (h, w)
        The best estimate of villain's face.
    B : array, shape (H, W)
        The best estimate of background.
    s : float
        The best estimate of standard deviation of Gaussian noise.
    A : array, shape (H-h+1, W-w+1)
        The best estimate of prior on displacement of face in any image.
    L : float
        The best L(q,F,B,s,A).
    """
    L_best = -np.inf
    F_best = None
    B_best = None
    s_best = None
    A_best = None

    for i in range(n_restarts):
        F, B, s, A, LL = run_EM(X, h, w, tolerance=tolerance, max_iter=max_iter, use_MAP=use_MAP, init_type=init_type)
        if LL[-1] > L_best:
            L_best = LL[-1]
            F_best = F
            B_best = B
            s_best = s
            A_best = A

    return F_best, B_best, s_best, A_best, L_best
