import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from helper_functions import *
import cantools

pd.options.mode.chained_assignment = None  # default='warn'

def preprocessDataset(df):
    '''
    Takes a dataframe with only one column containing a candump log
    Returns a new dataframe with the log formatted into columns as in can-ml
    '''

    df = df[df.columns[0]].str.split(n=4, expand=True)                              # Split by whitespace
    df = df.rename(columns={0: "timestamp", 1: "can0", 2: "frame", 3: "attack"})    # Rename columns
    df.timestamp = df.timestamp.apply(lambda x: float(eval(x)))                     # Format the timestamps
    df[['arbitration_id', 'data_field']] = df['frame'].str.split('#', expand=True)  # Split the frame into AID and data field columns
    df = df.drop(columns=['can0', 'frame'])                                         # Remove extraneous columns
    df = df[['timestamp', 'arbitration_id', 'data_field', 'attack']]                # Rearrange columns
    return df


def printSummary(df):
    '''
    Takes a dataframe formatted as in can-ml  
    Prints summary information like number of frames etc.  
    Returns None  
    '''
    total_frames   = len(df)
    total_aids     = len(df.arbitration_id.unique())
    total_duration = df.timestamp[len(df) - 1] - df.timestamp[0]
    msgs_per_sec   = total_frames / total_duration
    print("Total number of CAN frames       =", total_frames)
    print("Total duration (seconds)         =", total_duration)
    print("Avg. no. of messages per second  =", msgs_per_sec)
    print("Total no. of AIDs                =", total_aids)
    print()

    return total_frames, total_duration


def getTimeIntervals(df):
    ''' 
    Takes a DataFrame with a timestamp and arbitration_id columns
    Returns a the same DataFrame with a new column `time_interval` and `aid_time_interval`
    '''

    # Calculate time_interval
    df['time_interval'] = df['timestamp'].diff()                # Calculate time interval with previous message
    df = df.iloc[1:]                                            # Remove the first row where time_interval is NaN

    # Calculate aid_time_interval
    df = df.sort_values(['arbitration_id', 'timestamp'])        # Sort all messages according to AIDs
    df['aid_time_interval'] = df['timestamp'].diff()            # Get time intervals within each AID group
    df = df[df.arbitration_id == df.arbitration_id.shift(1)]    # Remove first frame of each AID where the interval should be 0
    df = df.sort_values('timestamp').reset_index()              # Reorder according to original timestamps

    # Remove timestamp columns
    df = df.drop(columns=['timestamp'])   

    # Rearrange columns
    df = df[['time_interval', 'aid_time_interval', 'arbitration_id', 'data_field', 'attack']]          

    return df

def getTimeIntervalSummary(df):
    '''
    Takes a DataFrame with aid_time_interval and arbitration_id columns
    Returns a new DataFrame with AID, message count, average time interval, standard deviation, and maximum percent error columns
    '''
    df = df[['aid_time_interval', 'arbitration_id']]

    aids    = list(df.arbitration_id.unique())
    counts  = []
    avgs    = []
    stdDevs = []
    maxErrors = []

    for aid in aids: 
        df_sub = df[df.arbitration_id == aid].reset_index(drop=True)
        
        counts.append(len(df_sub))
        avg = df_sub.aid_time_interval.mean()
        avgs.append(avg)
        stdDevs.append(df_sub.aid_time_interval.std())

        errors = (abs(df_sub.aid_time_interval - avg) / avg) * 100
        maxErrors.append(max(errors))

    new_df = pd.DataFrame({
        "arbitration_id"            : aids,
        "message_count"             : counts,
        "avg_time_interval"         : avgs,
        "std_dev_time_interval"     : stdDevs,
        "max_percent_error_mean"    : maxErrors

    })
    new_df = new_df.sort_values(by='arbitration_id', ignore_index=True)
    return new_df



def stringToByte(x):
    ''' 
    Takes a string representing a CAN frame payload, e.g. "0520ea0a201a007f"  
    Converts the string to a byte object, e.g. b'\x05\x20\xea\x0a\x20\x1a\x00\x7f'  
    Returns a byte object  
    ''' 

    x = [int(x[i:i+2], 16) for i in range(0, len(x), 2)]
    x = bytes(list(x))
    return x


def deserializeCANFrame(arbitration_id, data_field, db):
    '''
    Takes the AID and payload of a single CAN Frame and a DBC 
    Returns a dictionary of signals decoded
    '''
    try: 
        return db.decode_message(int(arbitration_id, 16), stringToByte(data_field), decode_choices=False, allow_truncated=True)

    except KeyError:
        return {'NoSignal': None}
    except Exception as e: 
        raise e


def deserializeCANDataFrame(df, db, attack=False, signals=None):
    ''' 
    Takes a dataframe of CAN frames as in can-ml and a DBC file
    Deserializes the payload of each frame to yield signals using the provided DBC
    Returns a new dataframe with a column for each signal of each AID instead of the data_field column
    '''

    df = df.copy()

    if attack and (signals == None or signals == []):
        raise Exception("No or empty list of valid signals to retain for attack DataFrame")
    
    try:
        df.loc[:, 'signals'] = df.apply(lambda x: deserializeCANFrame(x['arbitration_id'], x['data_field'], db), axis=1)
        
        dfsig_norm = df.apply(lambda x: {x['arbitration_id'] : x['signals']}, axis=1)
        dfsig_norm = pd.json_normalize(dfsig_norm)
        
        df = df.drop(columns=['signals', 'data_field'])
        dfsig_norm = dfsig_norm.drop(columns=[col for col in dfsig_norm.columns if 'NoSignal' in col])  # Remove NoSignals columns from invalid AIDs
        dfsig_norm = dfsig_norm[sorted(dfsig_norm.columns)]                                             # Sort columns 

        if attack:
            dfsig_norm = dfsig_norm.reindex(columns=signals)

        df = pd.concat([df[[col for col in df.columns if col != 'attack']], dfsig_norm, df[['attack']]], axis=1)
        
    except Exception as e:
        raise e
    return df


def getWindowLabel(x):
    x = x.unique()
    if len(x) == 1 and x[0] == 0:
        return 0    
    return [v for v in x if v != 0][0]


def getSignalMinMax(db, cols):
    '''
    Takes a DBC file and a list of columns (each signal column label should be in the form of AID.Signal)
    Grabs the minimum and maximum values of each signal from the DBC
    Returns a DataFrame with the AID.Signal signal names as index and columns minimum and maximum 
    '''

    cols = [x for x in cols if x not in ['timestamp', 'time_interval', 'aid_time_interval', 'arbitration_id', 'attack']]
    result = pd.DataFrame(columns=['minimum', 'maximum'])

    signal  = []
    minimum = []
    maximum = []

    for col in cols: 
        aid = col.split('.', maxsplit=1)[0]
        sig = col.split('.', maxsplit=1)[1]

        msg_rule = db.get_message_by_frame_id(int(aid, 16)).signals

        for sig_rule in msg_rule:
            if sig_rule.name == sig and sig_rule.minimum != None and sig_rule.minimum != None: 
                min = sig_rule.minimum
                max = sig_rule.maximum

                signal.append(col)
                minimum.append(min)
                maximum.append(max)

                break
    
    result = pd.DataFrame({
        'minimum' : minimum,
        'maximum' : maximum 
    },  index = signal)

    return result


def getColumnMinMax(df, db, cols): 
    '''
    Takes a DataFrame, a DBC, and a list of columns (each signal column label should be in the form of AID.Signal)
    Determines the minimum and maximum values for all columns (except attack)
    Returns a DataFrame with the column names as index and columns minimum and maximum 

    This is the equivalent of fitting a minmaxscaler
    '''

    signal_min_max = getSignalMinMax (db=db, cols=cols)     # Get min and max values of signals from DBC
    
    minimum = []
    maximum = []
    remain_cols = [col for col in cols if (col not in signal_min_max.index and col != 'attack')]  
    
    for col in remain_cols:                                 # Get min and max values from training set for time intervals, AIDs, and any signal for which there are no min and max values in DBC
        minimum.append(df[col].min())
        maximum.append(df[col].max())



    result = pd.DataFrame({
        'minimum' : minimum,
        'maximum' : maximum 
    },  index = remain_cols)

    result = pd.concat([signal_min_max, result], axis=0)
    return result


def minMaxScaleWindow(df, minmax_vals):
    '''
    Takes a DataFrame with decoded signal columns and a DataFrame with minimum and maximum values for each column
    Scales the signal columns according to the minimum and maximum values in minmax_vals
    Returns the same DataFrame except with the signal columns scaled

    This is the equivalent of using a fitted minmaxscaler to scale a dataframe
    '''

    try: 
        df = df.astype('float64')
        for col in df.columns:
            if col != 'attack':
                min = minmax_vals['minimum'][col]
                max = minmax_vals['maximum'][col]
                df[col] = (df[col] - min) / (max - min)

        return df

    except KeyError:
        print("Mismatch between column names in DataFrame and in minmax_vals")
        print("Error on", col)


def minMaxScaleWindowModel(df, minmax_vals):
    '''
    Takes a DataFrame with decoded signal columns and a DataFrame with minimum and maximum values for each column
    Scales the signal columns according to the minimum and maximum values in minmax_vals
    Returns the same DataFrame except with the signal columns scaled

    This is almost the same as the minmaxScaleWindow function, except it does not include the attack column in the returned dataframe and vectorizes the scaling
    '''

    ## Exclude attack column
    cols = [x for x in df.columns if x != 'attack']

    # Get min and max values for each column
    minmax_vals = minmax_vals.T
    minmax_vals = minmax_vals.reindex(columns=cols)
    df = df.reindex(columns=cols)

    df = (df - minmax_vals.loc['minimum']) / (minmax_vals.loc['maximum'] - minmax_vals.loc['minimum'])
    return df


def minMaxScale(list_windows, minmax_vals):
    '''
    Takes a list of DataFrames with decoded signal columns and a DataFrame with minimum and maximum values for each column
    Scales the signal columns according to the minimum and maximum values in minmax_vals
    Returns the same DataFrame list except with the signal columns scaled

    '''

    for win_idx in range(len(list_windows)):
        list_windows[win_idx] = minMaxScaleWindow(list_windows[win_idx], minmax_vals)

    return list_windows


