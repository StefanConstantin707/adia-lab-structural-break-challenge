import numpy as np

def get_sliding_windows_from_sequence(sequence: np.array, window_size: int, stride: int = 1) -> np.ndarray:
    """
    Generate sliding windows from a 1D sequence.

    Parameters
    ----------
    sequence : array-like
        Input sequence. Must be 1D or have singleton dimensions (e.g., (n,1), (1,n), (1,n,1)), which will be squeezed.
    window_size : int
        Size of each sliding window. Must be positive.
    stride : int, optional
        Step size between the start indices of consecutive windows. Must be positive. Default is 1.

    Returns
    -------
    windows : np.ndarray
        2D array of shape (num_windows, window_size), where num_windows = max((len(sequence) - window_size)//stride + 1, 0).
        If the sequence length is less than window_size, returns an empty array with shape (0, window_size).

    Raises
    ------
    ValueError
        If sequence cannot be squeezed to 1D, or if window_size or stride are not positive integers.
    """
    arr = np.asarray(sequence)

    squeezed = np.squeeze(arr)

    if squeezed.ndim != 1:
        raise ValueError(f"Input sequence must be 1D or have only singleton dimensions, but got shape {arr.shape} after squeezing to {squeezed.shape}.")

    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError(f"window_size must be a positive integer, got {window_size}.")

    if not isinstance(stride, int) or stride <= 0:
        raise ValueError(f"stride must be a positive integer, got {stride}.")

    N = squeezed.shape[0]
    num_windows = (N - window_size) // stride + 1

    if num_windows <= 0:
        return np.empty((0, window_size), dtype=squeezed.dtype)

    start_indices = np.arange(num_windows).reshape(num_windows, 1) * stride

    offsets = np.arange(window_size).reshape(1, -1)
    idx = start_indices + offsets

    return squeezed[idx]
