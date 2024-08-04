import torch

from ..training.soft_dtw_cuda import SoftDTW

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%%

def compute_recon_error_between_two_tensor(x_1 : torch.tensor, x_2 : torch.tensor, device : str = 'cpu', average_channels : bool = False, average_time_samples : bool = False) -> torch.tensor :
    """
    Compute the dtw between two tensor x_1 and x_2 using the softDTW (implemented through https://github.com/Maghoumi/pytorch-softdtw-cuda).
    The two tensors (x_1 and x_2) must both have shape B x 1 x C x T. The dtw will be computed element by element and channel by channel.
    An element is an EEG signal of shape C x T and the number of element is B (i.e. the first dimension).
    For each EEG signal the dtw is computed independently channel by channel. Evenetually, averaged along channels if average_channels is True and over time sample if average_time_sample is True.

    Example :
    If your input x_1 has shape 4 x 1 x 22 x 1000 it means that the tensor contains 4 EEG signals, and each signal has 22 channels and 1000 temporal samples.
    The tensor x_2 must have the same shape. The dtw is computed element by element in the sense that the first element of x_1 is compared with the first element of x_2 etc.
    Then given a pair of EEG signal of shape 22 x 1000, inside the signal the dtw is computed channel by channel (because the dtw compute the difference between 1d signal).
    So for a pair of EEG signal of shape 22 x 1000 the dtw ouput are 22 values, 1 for each channel. If average_channels is True this 22 values will be averaged, otherwise the function will return them separately.
    The final output will have shape 4 x 22 if average_channels is False or 4 if average_channels is True.

    @param x_1: (torch.tensor) First input tensor of shape B x 1 x C x T
    @param x_1: (torch.tensor) Second input tensor of shape B x 1 x C x T
    @param device: (str) String that specify is use cuda (GPU) or cpu for computation. Note that to use the GPU this parameter must have value cuda
    @param average_channels : (bool) If True compute the average of the reconstruction error along the channels
    @param average_time_sample : (bool) If True the recon error of each channel is divided by the number of samples T.

    @return recon_error: (torch.tensor) Tensor of shape B x C (if average_channels is False) or B (if average_channels is True)
    """
    
    # Check input shape
    if x_1.shape != x_2.shape :
        raise ValueError("x_1 and x_2 must have the same shape. Current shape x_1 : {}, x_2 : {}".format(x_1.shape, x_2.shape))
    
    # Create loss function
    use_cuda = True if device == 'cuda' else False
    recon_loss_function = SoftDTW(use_cuda = use_cuda, normalize = False)
    
    # Tensor to save the reconstruction error
    recon_error = torch.zeros(x_1.shape[0], x_1.shape[2])
    
    # Move input tensor to device
    x_1 = x_2.to(device)
    x_2 = x_2.to(device)
    
    # Compute the DTW channel by channels
    for j in range(x_1.shape[2]): # Iterate through EEG Channels
        x_1_ch = x_1[:, :, j, :].swapaxes(1, 2)
        x_2_ch = x_2[:, :, j, :].swapaxes(1, 2)
        # Note that the depth dimension has size 1 for EEG signal. So after selecting the channel x_ch will have size [B x D x T], with D = depth = 1
        # The sdtw want the length of the sequence in the dimension with the index 1 so I swap the depth dimension and the the T dimension
        
        # Compute reconstruction error, and move it to cpu
        recon_error[:, j] = recon_loss_function(x_1_ch, x_2_ch).cpu()

    # (OPTIONAL) Normalizes on the number of samples
    if average_time_samples : recon_error /= x_1.shape[3]
    
    # (OPTIONAL) Average along channels
    if average_channels : recon_error = recon_error.mean(1)

    return recon_error
