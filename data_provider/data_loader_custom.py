import os
import numpy as np
import pandas as pd
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features, numeric_time_features
import warnings

warnings.filterwarnings('ignore')



class Dataset_Custom_jst(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='responder_6', scale=True, timeenc=0, freq='h', time_str='time_id'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_str = time_str #important, time str for each dataset

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler_y_jst = StandardScaler()
        self.scaler_x_jst = StandardScaler()

        df_raw = pd.read_parquet(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['self.time_str', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(self.time_str)
        df_raw = df_raw[[self.time_str] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_stamp = df_raw[[self.time_str]][border1:border2]

        del df_raw#clean up

        #prep x and y target
        #jst x set
        data_x_jst = df_data.drop(self.target, axis=1).copy()
        data_y_jst = df_data[[self.target]].copy()  #ensure it is a df instead of a series

        del df_data 

        if self.scale:         #do something here to separate the target from input x.
            train_data_y = data_y_jst[border1s[0]:border2s[0]]
            self.scaler_y_jst.fit(train_data_y.values)
            data_y_jst = self.scaler_y_jst.transform(data_y_jst.values)


            x_jst_train_data = data_x_jst[border1s[0]:border2s[0]]
            self.scaler_x_jst.fit(x_jst_train_data.values)
            data_x_jst = self.scaler_x_jst.transform(data_x_jst.values)

        else:
            data_x_jst = data_x_jst.values
            data_y_jst = data_y_jst.values


        #time encoding
        data_stamp =  numeric_time_features(df_stamp[self.time_str])
        data_stamp = data_stamp.transpose(1, 0)

        #save scaler y_jst
        

        # on ram
        # self.data_x = data_x_jst[border1:border2]# jst method
        # self.data_y = data_y_jst[border1:border2]
        # self.data_stamp = data_stamp

        #on gpu
        # Store data as tensors and move to the specified device
        self.data_x = torch.tensor(data_x_jst[border1:border2], dtype=torch.float32).to(self.device)
        self.data_y = torch.tensor(data_y_jst[border1:border2], dtype=torch.float32).to(self.device)
        self.data_stamp = torch.tensor(data_stamp, dtype=torch.float32).to(self.device)

        #clean up
        del data_x_jst
        del data_y_jst
        del data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        seq_y = self.data_y[s_begin:s_end]
        #seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]

        seq_y_mark = self.data_stamp[s_begin:s_end]
        #seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_jst_pred(Dataset):
    def __init__(self, scaler_x_path, scaler_y_path, size=None,
                 features='MS', data_path='ETTh1.csv',
                 target='responder_6', scale=True, timeenc=0, freq='h', time_str='time_id'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.scaler_x_path = scaler_x_path
        self.scaler_y_path = scaler_y_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.time_str = time_str #important, time str for each dataset

        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):

        df_raw = pd.read_parquet(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['self.time_str', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove(self.time_str)
        df_raw = df_raw[[self.time_str] + cols + [self.target]]
        # num_train = int(len(df_raw) * 0.7)
        # num_test = int(len(df_raw) * 0.2)
        # num_vali = len(df_raw) - num_train - num_test
        # border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        # border2s = [num_train, num_train + num_vali, len(df_raw)]
        # border1 = border1s[self.set_type]
        # border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        df_stamp = df_raw[[self.time_str]]

        del df_raw#clean up

        #prep x and y target
        #jst x set
        data_x_jst = df_data.drop(self.target, axis=1).copy()
        data_y_jst = df_data[[self.target]].copy()  #ensure it is a df instead of a series

        del df_data 

        if self.scale:         #do something here to separate the target from input x.

            #load scaler 
            self.scaler_y_jst = StandardScaler()
            self.scaler_x_jst = StandardScaler()

            #swap to exisitng scaler
            if self.scaler_x_path != 'None':
                with open(self.scaler_x_path, 'rb') as f:
                    self.scaler_x_jst = pickle.load(f)

            if self.scaler_y_path != 'None':
                with open(self.scaler_y_path, 'rb') as f:
                    self.scaler_y_jst = pickle.load(f)

            data_y_jst = self.scaler_y_jst.transform(data_y_jst.values)
            data_x_jst = self.scaler_x_jst.transform(data_x_jst.values)

        else:
            data_x_jst = data_x_jst.values
            data_y_jst = data_y_jst.values


        #time encoding
        data_stamp =  numeric_time_features(df_stamp[self.time_str])
        data_stamp = data_stamp.transpose(1, 0)

        # on ram
        # self.data_x = data_x_jst# jst method
        # self.data_y = data_y_jst
        # self.data_stamp = data_stamp

        #on gpu
        # Store data as tensors and move to the specified device
        self.data_x = torch.tensor(data_x_jst, dtype=torch.float32).to(self.device)
        self.data_y = torch.tensor(data_y_jst, dtype=torch.float32).to(self.device)
        self.data_stamp = torch.tensor(data_stamp, dtype=torch.float32).to(self.device)

        #clean up
        del data_x_jst
        del data_y_jst
        del data_stamp


    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]

        seq_y = self.data_y[s_begin:s_end]
        #seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]

        seq_y_mark = self.data_stamp[s_begin:s_end]
        #seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        #return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)