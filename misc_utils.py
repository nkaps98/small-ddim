import os
import torch

def load_ddim_sls(i2n_dir, n2i_dir, reqd_strings=None, retrieve_interval=10, store_sls=True, compute_dists=False):
 
    # Initialize lists
    i2n_sls = []
    n2i_sls = []
    i2n_dist = []
    n2i_dist = []
    i2i_dist = []
 
    # Load streamline files
    i2n_files = sorted(os.listdir(i2n_dir))
    n2i_files = sorted(os.listdir(n2i_dir))
    assert len(i2n_files) == len(n2i_files), 'There must be equal amounts of I2N and N2I files.'
    for (i2n_file, n2i_file) in zip(i2n_files, n2i_files):
        if reqd_strings is not None:
            if not any([reqd_strings[i] in i2n_file for i in range(len(reqd_strings))]) or not any([reqd_strings[i] in n2i_file for i in range(len(reqd_strings))]):
                continue
        i2n_name_parts = os.path.splitext(i2n_file)[0].split('_')
        n2i_name_parts = os.path.splitext(n2i_file)[0].split('_')
        assert len(i2n_name_parts) == len(n2i_name_parts), 'I2N and N2I file names must refer to same base file.'
        for i in range(len(i2n_name_parts) - 3):
            assert i2n_name_parts[i] == n2i_name_parts[i], 'I2N and N2I must refer to same base file.'
        i2n_sl = torch.load(os.path.join(i2n_dir, i2n_file))
        n2i_sl = torch.load(os.path.join(n2i_dir, n2i_file))
        # print(i2n_sl[-1], n2i_sl[0])
        assert torch.all(i2n_sl[-1] == n2i_sl[0]), 'End of I2N must be coincident with start of N2I.'
 
        # Get every nth point of streamlines
        assert len(i2n_sl) % retrieve_interval == 1, 'Must retrieve start and end points of SL.'
        assert len(n2i_sl) % retrieve_interval == 1, 'Must retrieve start and end points of SL.'
        i2n_sl = i2n_sl[::retrieve_interval]
        n2i_sl = n2i_sl[::retrieve_interval]
 
        # Store required information
        if store_sls:
            i2n_sls.append(i2n_sl.squeeze().cpu())
            n2i_sls.append(n2i_sl.squeeze().cpu())
        if compute_dists:
            i2n_dist.append(torch.linalg.vector_norm(i2n_sl[-1] - i2n_sl[0], keepdims=False))
            n2i_dist.append(torch.linalg.vector_norm(n2i_sl[-1] - n2i_sl[0], keepdims=False))
            i2i_dist.append(torch.linalg.vector_norm(n2i_sl[-1] - i2n_sl[0], keepdims=False))
    if store_sls:
        i2n_sls = torch.stack(i2n_sls)
        n2i_sls = torch.stack(n2i_sls)
 
    return i2n_sls, n2i_sls, i2n_dist, n2i_dist, i2i_dist