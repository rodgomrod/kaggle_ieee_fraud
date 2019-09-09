import numpy as np
import pandas as pd

def reduce_memory(df):
    print("Reduce_memory...");
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    return df


def reduce_memory2(df):
    print("Reduce_memory...");
    dict_types = dict()
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dict_types[col] = np.int8
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dict_types[col] = np.int16
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dict_types[col] = np.int32
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dict_types[col] = np.int64
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dict_types[col] = np.float16
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dict_types[col] = np.float32
                else:
                    dict_types[col] = np.float64
    return dict_types


def freq_encoder(df, label, new_label, min_freq = 0.001):
    rows = df.shape[0]
    n = 0
    dict_fe = dict()
    vc = df[label].value_counts()
    for i, j in zip(vc.index, vc):
        ratio = j/rows
        if ratio > min_freq:
            dict_fe[i] = n
            n += 1
        else:
            dict_fe[i] = n
        
    if n < 2**8:
        _d_type = 'uint8'
    elif n >= 2**8 and n < 8**16:
        _d_type = 'uint16'
    elif n >= 2**16 and n < 8**32:
        _d_type = 'uint32'
    else:
        _d_type = 'uint64'
        
    df[new_label] = df[label].apply(lambda x: dict_fe[x]).astype(_d_type)
    
    n = 0
    dict_fe = dict()
    vc = df[label].value_counts()
    for i, j in zip(vc.index, vc):
        ratio = j/rows
        if ratio > min_freq:
            dict_fe[i] = n
            n += 1
        else:
            dict_fe[i] = n
            
    df[new_label] = df[label].apply(lambda x: dict_fe[x]).astype(_d_type)
    
    return df
