import torch
import pandas as pd
import numpy as np


class NFkBKTR_Dataset(torch.utils.data.Dataset):
    '''
    Create a torch.data.Dataset class for 2 dimension time series sequences of NFkB and KTR 
    Inputs:
        nfkb_path: [...CpG_R1, LPS_R1, ...]
        ktr_path: [...CpG_R1, LPS_R1, ...]
        i.e indices of nfkb and ktr path should have corresponding ligands
    '''
    def __init__(self, nfkb_path, ktr_path, data_path, remove_nans=False):
        assert len(nfkb_path) == len(ktr_path)
        self.data, self.labels = [], []
        
        for label, (i, j) in enumerate(zip(nfkb_path, ktr_path)):
            
            if len(i) == 2: # replicas
                assert len(i) == len(j)
                r1_nfkb, r2_nfkb = np.array(pd.read_csv(data_path + i[0])), np.array(pd.read_csv(data_path + i[1]))
                r1_ktr, r2_ktr = np.array(pd.read_csv(data_path + j[0])), np.array(pd.read_csv(data_path + j[1]))
                nfkb_array = np.concatenate([r1_nfkb, r2_nfkb], axis=0)
                ktr_array = np.concatenate([r1_ktr, r2_ktr], axis=0)
            else:
                nfkb_array = np.array(pd.read_csv(i))
                ktr_array = np.array(pd.read_csv(j))
            
            assert nfkb_array.shape == ktr_array.shape # array shapes should be the same
            if remove_nans:
                # remove the union of nan rows for both arrays
                nfkb_array, ktr_array = nfkb_array[~np.isnan(nfkb_array).any(axis=1)], ktr_array[~np.isnan(nfkb_array).any(axis=1)]
                nfkb_array, ktr_array = nfkb_array[~np.isnan(ktr_array).any(axis=1)], ktr_array[~np.isnan(ktr_array).any(axis=1)]
                
            nfkb_array = nfkb_array.reshape(nfkb_array.shape[0], nfkb_array.shape[1], 1)
            ktr_array = ktr_array.reshape(ktr_array.shape[0], ktr_array.shape[1], 1)
            
            data = np.concatenate([nfkb_array, ktr_array], axis=2) # [num_rows, num_timefeatures, 2]
            labels = np.repeat(label, len(nfkb_array))
            
            self.data.append(data)
            self.labels.append(labels)
            
        self.data = np.concatenate(self.data, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)
        assert len(self.data) == len(self.labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
