import numpy as np
import torch


def tensor_to_complex_np(data):
    """
    Converts a complex torch tensor to numpy array.
    Args:
        data (torch.Tensor): Input data to be converted to numpy.
    Returns:
        np.array: Complex numpy version of data
    """
    data = data.numpy()
    return data[..., 0] + 1j * data[..., 1]


def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.

    Args:
        data (np.array): Input numpy array

    Returns:
        torch.Tensor: PyTorch version of data
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)
    return torch.from_numpy(data)


def apply_mask(data, mask_func, seed=None):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    shape = np.array(data.shape)
    shape[:-3] = 1
    mask = mask_func(shape, seed)
    return torch.where(mask == 0, torch.Tensor([0]), data), mask
def complex_abs(data):
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data (torch.Tensor): A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        torch.Tensor: Absolute value of data
    """
    assert data.size(-1) == 2
    return (data ** 2).sum(dim=-1).sqrt()


def root_sum_of_squares(data, dim=0):
    """
    Compute the Root Sum of Squares (RSS) transform along a given dimension of a tensor.

    Args:
        data (torch.Tensor): The input tensor
        dim (int): The dimensions along which to apply the RSS transform

    Returns:
        torch.Tensor: The RSS value
    """
    return torch.sqrt((data ** 2).sum(dim))


def center_crop(data, shape):
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data (torch.Tensor): The input tensor to be center cropped. It should have at
            least 2 dimensions and the cropping is applied along the last two dimensions.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-2]
    assert 0 < shape[1] <= data.shape[-1]
    w_from = (data.shape[-2] - shape[0]) // 2
    h_from = (data.shape[-1] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to]


def complex_center_crop(data, shape):
    """
    Apply a center crop to the input image or batch of complex images.

    Args:
        data (torch.Tensor): The complex input tensor to be center cropped. It should
            have at least 3 dimensions and the cropping is applied along dimensions
            -3 and -2 and the last dimensions should have a size of 2.
        shape (int, int): The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        torch.Tensor: The center cropped image
    """
    assert 0 < shape[0] <= data.shape[-3]
    assert 0 < shape[1] <= data.shape[-2]
    w_from = (data.shape[-3] - shape[0]) // 2
    h_from = (data.shape[-2] - shape[1]) // 2
    w_to = w_from + shape[0]
    h_to = h_from + shape[1]
    return data[..., w_from:w_to, h_from:h_to, :]


def normalize(data, mean, stddev, eps=0.):
    """
    Normalize the given tensor using:
        (data - mean) / (stddev + eps)

    Args:
        data (torch.Tensor): Input data to be normalized
        mean (float): Mean value
        stddev (float): Standard deviation
        eps (float): Added to stddev to prevent dividing by zero

    Returns:
        torch.Tensor: Normalized tensor
    """
    return (data - mean) / (stddev + eps)


def normalize_instance(data, eps=0.):
    """
        Normalize the given tensor using:
            (data - mean) / (stddev + eps)
        where mean and stddev are computed from the data itself.

        Args:
            data (torch.Tensor): Input data to be normalized
            eps (float): Added to stddev to prevent dividing by zero

        Returns:
            torch.Tensor: Normalized tensor
        """
    mean = data.mean()
    std = data.std()
    return normalize(data, mean, std, eps), mean, std


# Helper functions

def roll(x: torch.Tensor, shift: int, dim: int = -1) -> torch.Tensor:
    """Torch-native roll that works on any device."""
    return torch.roll(x, shifts=shift, dims=dim)

def fftshift(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Shift zero-frequency component to center along one dim."""
    return torch.fft.fftshift(x, dim=dim)

def fftshift2(x: torch.Tensor) -> torch.Tensor:
    """2D fftshift over the last two spatial dims (H, W)."""
    return torch.fft.fftshift(x, dim=(-2, -1))

def _to_complex(x_2ch: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,2,H,W) format to (N,H,W) complex tensor.
    For FastMRI knee data: channels 0=real, 1=imaginary
    """
    if x_2ch.dim() != 4 or x_2ch.size(1) != 2:
        raise ValueError(f"Expected (N,2,H,W), got {tuple(x_2ch.shape)}")
    return torch.complex(x_2ch[:, 0], x_2ch[:, 1])

def _to_2ch(x_complex: torch.Tensor) -> torch.Tensor:
    """
    Convert (N,H,W) complex tensor to (N,2,H,W) format.
    Output: channel 0=real, 1=imaginary
    """
    if not torch.is_complex(x_complex):
        raise ValueError("Expected complex tensor")
    return torch.stack([x_complex.real, x_complex.imag], dim=1)

def fft2(input_: torch.Tensor) -> torch.Tensor:
    """
    2D FFT: (N,2,H,W) -> (N,2,H,W)
    Matches GitHub behavior but uses modern torch.fft API
    """
    # Convert to complex, apply FFT, convert back
    x_complex = _to_complex(input_)
    # Use 'backward' norm to match old torch.fft behavior
    k_complex = torch.fft.fft2(x_complex, norm='backward') 
    return _to_2ch(k_complex)

def ifft2(input_: torch.Tensor) -> torch.Tensor:
    """
    2D IFFT: (N,2,H,W) -> (N,2,H,W) 
    Matches GitHub behavior but uses modern torch.fft API
    """
    X_complex = _to_complex(input_)
    # Use 'backward' norm to match old torch.ifft behavior  
    x_complex = torch.fft.ifft2(X_complex, norm='backward')
    return _to_2ch(x_complex)

def fft1(input_: torch.Tensor, axis: int) -> torch.Tensor:
    """
    1D FFT along specified spatial axis.
    axis=0: along W, axis=1: along H
    """
    x_complex = _to_complex(input_)
    if axis == 1:   # along H dimension
        k_complex = torch.fft.fft(x_complex, dim=-2, norm='backward')
    elif axis == 0: # along W dimension  
        k_complex = torch.fft.fft(x_complex, dim=-1, norm='backward')
    else:
        raise ValueError("axis must be 0 (W) or 1 (H)")
    return _to_2ch(k_complex)

def ifft1(input_: torch.Tensor, axis: int) -> torch.Tensor:
    """
    1D IFFT along specified spatial axis.
    axis=0: along W, axis=1: along H
    """
    X_complex = _to_complex(input_)
    if axis == 1:   # along H dimension
        x_complex = torch.fft.ifft(X_complex, dim=-2, norm='backward')  
    elif axis == 0: # along W dimension
        x_complex = torch.fft.ifft(X_complex, dim=-1, norm='backward')
    else:
        raise ValueError("axis must be 0 (W) or 1 (H)")
    return _to_2ch(x_complex)