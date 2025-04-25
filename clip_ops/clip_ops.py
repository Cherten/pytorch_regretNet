import numpy as np
import torch


def clip_op_01(x):
    """Clip values to [0, 1]"""
    return torch.clamp(x, 0.0, 1.0)


def clip_op_12(x):
    """Clip values to [1, 2]
    For CA format: x has shape [batch_size, num_agents, 3] where 3 is [item1, item2, bundle]
    """
    return torch.clamp(x, 1.0, 2.0)


def clip_op_23(x):
    """Clip values to [2, 3]"""
    return torch.clamp(x, 2.0, 3.0)


def clip_op_12_15(x):
    """Clip first agent's values to [1, 2] and second agent's values to [1, 5]
    For CA format: x has shape [batch_size, num_agents, 3] where 3 is [item1, item2, bundle]
    """
    if x.dim() == 4:  # Для формата ADV [num_misreports, batch_size, num_agents, 3]
        x[:, :, 0, :] = torch.clamp(x[:, :, 0, :], 1.0, 2.0)  # Agent 1: all values [1,2]
        x[:, :, 1, :] = torch.clamp(x[:, :, 1, :], 1.0, 5.0)  # Agent 2: all values [1,5]
    else:  # Для формата X [batch_size, num_agents, 3]
        x[:, 0, :] = torch.clamp(x[:, 0, :], 1.0, 2.0)  # Agent 1: all values [1,2]
        x[:, 1, :] = torch.clamp(x[:, 1, :], 1.0, 5.0)  # Agent 2: all values [1,5]
    return x


def clip_op_416_47(x):
    """Clip first item's values to [4, 16] and second item's values to [4, 7]"""
    if x.dim() == 4:  # Для формата ADV
        x[:, :, :, 0] = torch.clamp(x[:, :, :, 0], 4.0, 16.0)
        x[:, :, :, 1] = torch.clamp(x[:, :, :, 1], 4.0, 7.0)
    else:  # Для формата X
        x[:, :, 0] = torch.clamp(x[:, :, 0], 4.0, 16.0)
        x[:, :, 1] = torch.clamp(x[:, :, 1], 4.0, 7.0)
    return x


def clip_op_04_03(x):
    """Clip first item's values to [0, 4] and second item's values to [0, 3]"""
    if x.dim() == 4:  # Для формата ADV
        x[:, :, :, 0] = torch.clamp(x[:, :, :, 0], 0.0, 4.0)
        x[:, :, :, 1] = torch.clamp(x[:, :, :, 1], 0.0, 3.0)
    else:  # Для формата X
        x[:, :, 0] = torch.clamp(x[:, :, 0], 0.0, 4.0)
        x[:, :, 1] = torch.clamp(x[:, :, 1], 0.0, 3.0)
    return x


def clip_op_triangle_01_numpy(x):
    """
    Clip values to be in the triangle defined by (0,0), (1,0), (0,1)
    """
    # Convert to numpy for processing
    if isinstance(x, torch.Tensor):
        is_tensor = True
        device = x.device
        x_numpy = x.detach().cpu().numpy()
    else:
        is_tensor = False
        x_numpy = x
    
    x_shape = x_numpy.shape
    x_numpy = np.reshape(x_numpy, [-1, 2])
    
    # Find invalid points (outside triangle)
    invalid_idx = np.where((x_numpy[:,0] < 0) | (x_numpy[:,1] < 0) | (x_numpy.sum(-1) >= 1))
    
    if len(invalid_idx[0]) > 0:
        x_invalid = x_numpy[invalid_idx]
        
        p = np.zeros((x_invalid.shape[0], 3, 2))
        d = np.zeros((x_invalid.shape[0], 3))
        t = np.zeros((x_invalid.shape[0], 3))
        
        # Parameters for projecting to triangle edges
        t[:, 0] = (x_invalid[:, 0] - x_invalid[:, 1] + 1.0) / 2.0
        t[:, 1] = (1 - x_invalid[:, 1])
        t[:, 2] = (1 - x_invalid[:, 0])
        t = np.clip(t, 0.0, 1.0)
        
        # Triangle vertices
        A = np.array([[0, 1]]).T
        B = np.array([[1, 0]]).T
        O = np.array([[0, 0]]).T
        pts_x = [A, A, B]
        pts_y = [B, O, O]
        
        # Project to edges
        for i in range(3):
            p[:, i, :] = ((1 - t[:, i]) * pts_x[i] + t[:, i] * pts_y[i]).T
            d[:, i] = np.sum((x_invalid - p[:, i, :]) ** 2, -1)
        
        # Select closest projection
        sel_p = p[np.arange(x_invalid.shape[0]), np.argmin(d, -1), :]
        x_numpy[invalid_idx] = sel_p
    
    # Reshape back
    x_numpy = np.reshape(x_numpy, x_shape)
    
    # Convert back to tensor if needed
    if is_tensor:
        return torch.tensor(x_numpy, device=device)
    else:
        return x_numpy


def clip_op_triangle_01(x):
    """Clip values to be in the triangle defined by (0,0), (1,0), (0,1)"""
    return clip_op_triangle_01_numpy(x) 