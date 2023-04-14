"""
Created on Wed Nov 16 20:32:48 2016

@author: shayneufeld

This file contains preprocessing functions to create feature dataframes
for our AC209A project modeling the 2-armed bandit task in mice
"""
import numpy as np
import pandas as pd
import os

def add_session(root_dir,record_df, owner_id, imaging=False):
    '''
    This function is for a single session. The root dir here should point
    directly to the folder where trials.csv and parameters.csv are located.
    
    '''
    
    columns = ['Elapsed Time (s)','Since last trial (s)',
    'Trial Duration (s)','Port Poked','Right Reward Prob',
    'Left Reward Prob','Reward Given', 'laser_stim']
    if imaging==True:
        columns.extend(['center_frame', 'decision frame'])
    
    try:  
        for file in os.listdir(root_dir):
            file_name = os.path.join(root_dir,file)

            if 'trials.csv' in file:
                trials = pd.read_csv(file_name,names=columns)
                session_id = file[:file.index('_',9)]

            elif 'parameters.csv' in file:
                params = pd.read_csv(file_name)

        # calculate p(high port)
        high_p_port = np.zeros(trials.shape[0])
        
        for row in trials.iterrows():
          
            i = row[0]
            current_trial = row[1]
                        
            if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
                high_p_port[i] = 1
            elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
                high_p_port[i] = 1
           
        print(np.round(high_p_port.mean(),decimals=2))
        # convert date to datetime object
        if owner_id=='lynne':
            date=session_id[-8:]    
            mouse_id = session_id[:-9]
        else:
            date = session_id[:8]
            print('check date format')
            
            if owner_id=='mike':
                mouse_id = session_id[9:]
            else:
                mouse_id = session_id[:9]
            
        date_str =  np.str(date)
        if (len(date_str) == 7):
            date_str = '0' + date_str
        datetime = pd.to_datetime(date_str,format='%m%d%Y')

        # correct for block range stats if markov task was run
        if hasattr(params, 'ismarkov')==False:
            params['ismarkov']=0
                    
        if params.ismarkov[0]:
            params.loc[:,('blockRangeMin', 'blockRangeMax')] ='NaN'
                        
        # create a dictionary with all information
        record = {
            'Owner': owner_id,
            'Session ID': session_id,
            'Mouse ID': mouse_id,
            'Date': datetime,
            'Markov': params.ismarkov[0],
            'Left Reward Prob': params['leftRewardProb'].values,
            'Right Reward Prob': params['rightRewardProb'].values,
            'Block Range Min': params['blockRangeMin'],
            'Block Range Max': params['blockRangeMax'],
            'No. Trials': trials.shape[0],
            'No. Blocks': np.sum(np.diff(trials['Right Reward Prob'].values) != 0),
            'No. Rewards': np.sum(trials['Reward Given']),
            'p(high Port)': np.round(high_p_port.mean(),decimals=2),
            'Decision Window Duration': params['centerPokeRewardWindow'],
            'Min Inter-trial-interval': params['minInterTrialInterval'],
            'Left Solenoid Duration': params['rewardDurationLeft'],
            'Right Solenoid Duration': params['rewardDurationRight']
                 }
        
        #create DataFrame   
        if len(record_df)>0:
            record_df = record_df.append(pd.DataFrame(data=record,columns=record.keys()),ignore_index=True)
            record_df = record_df.drop_duplicates()
        else:
            record_df = pd.DataFrame(data=record)     
    
    except: 
        pass
    
    return record_df


def create_feature_matrix(trials,n_indi,imaging=False):
    '''
    This function creates the feature matrix we will use!
    
    Inputs:
            trials       :  pandas dataframe returned by extractTrials.m
            n_indi       :  number of past trials to be used in individual trial features
            imaging      : true/false whether is imaging data for this behavior session
    Outputs:
            feature_matrix: pandas dataframe of the features for each trial
        
    '''
    
    feature_names = [
            'Trial',
            'Block Trial',
            '0_ITI',
            '0_trialDuration',
            'Decision',
            'Switch',
            'Higher p port',
            'Reward'               
             ]

    feature_names[5:5] = [item for i in np.arange(1,n_indi+1) for item in [f'{i}_Port', f'{i}_Reward', f'{i}_ITI', f'{i}_trialDuration']]
    
    prob_dict = {1: '100-0',
                 0: '100-0',
                 0.95:'95-05', 
                 0.05:'95-05', 
                 0.1:'90-10', 
                 0.9:'90-10',
                 0.8:'80-20',
                 0.2:'80-20',
                 0.85:'85-15',
                 0.15:'85-15'}
    
    feature_matrix = pd.DataFrame(columns=feature_names)
            
    feature_matrix['Trial'] = trials.iloc[n_indi:].index.values+1       
    feature_matrix['Mouse ID'] = trials.iloc[n_indi:]['Mouse'].values
    feature_matrix['Session ID'] = trials.iloc[n_indi:]['Session'].values
    feature_matrix['Condition'] = prob_dict[trials['Left Reward Prob'][0]]          
    feature_matrix['Reward'] = trials.iloc[n_indi:]['Reward Given'].values
    feature_matrix['0_ITI'] = trials.iloc[n_indi:]['Since last trial (s)'].values
    feature_matrix['0_trialDuration'] = trials.iloc[n_indi:]['Trial Duration (s)'].values
    feature_matrix['Decision'] = trials.iloc[n_indi:]['Port Poked'].values - 1
    feature_matrix['Switch'] = np.abs(np.diff(trials.iloc[n_indi-1:]['Port Poked'].values))
    feature_matrix['laser_stim'] = trials.iloc[n_indi:]['laser_stim'].values

    
    for iBack in np.arange(1,n_indi+1):
        
        feature_matrix[f'{iBack}_Port'] = trials.iloc[n_indi-iBack:-iBack]['Port Poked'].values - 1
        feature_matrix[f'{iBack}_Reward'] = trials.iloc[n_indi-iBack:-iBack]['Reward Given'].values
        feature_matrix[f'{iBack}_ITI'] = trials.iloc[n_indi-iBack:-iBack]['Since last trial (s)'].values
        feature_matrix[f'{iBack}_trialDuration'] = trials.iloc[n_indi-iBack:-iBack]['Trial Duration (s)'].values
     
    n_trials = trials.shape[0] #number of trials in this session
    block_starts = np.zeros(n_trials)
    block_starts[1:] = np.diff(trials['Right Reward Prob'].values) != 0
    
    for j,i in enumerate(np.arange(n_indi,n_trials)):
        
        '''
        Block Trial Number
        '''
        if (j == 0): #first block number will be the sum
            if (np.sum(block_starts[:n_indi]) == 0):
            #then we are still in the first block, and we simply are n_indi trials in
                feature_matrix.loc[j,'Block Trial'] = n_indi+1
            else:
                feature_matrix.loc[j,'Block Trial'] = n_indi - np.where(block_starts[:n_indi]==True)[0][0] + 1
                 
        elif block_starts[i]: #if block_starts[j] == True, start counting from 0
            feature_matrix.loc[j,'Block Trial'] = 0 
        else:
            feature_matrix.loc[j,'Block Trial'] = feature_matrix.loc[j-1,'Block Trial'] + 1
        
        '''
        CURRENT TRIAL
        '''
        current_trial = trials.iloc[i,:]
                
        '''
        p(high) port
        '''
        
        if ((current_trial['Right Reward Prob'] > current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 1)):
            feature_matrix.loc[j,'Higher p port'] = 1
        elif ((current_trial['Right Reward Prob'] < current_trial['Left Reward Prob']) & (current_trial['Port Poked'] == 2)):
            feature_matrix.loc[j,'Higher p port'] = 1
        else:
            feature_matrix.loc[j,'Higher p port'] = 0

        
        '''
        sync to imaging data
        '''
    if imaging is True:
        feature_matrix['center_frame'] = trials.iloc[n_indi:]['center_frame'].values
        feature_matrix['decision_frame'] = trials.iloc[n_indi:]['decision_frame'].values
        
    feature_matrix['Target']=(feature_matrix.Decision * feature_matrix['Higher p port'])\
                                                    + ((1-feature_matrix.Decision) * (1-feature_matrix['Higher p port']))
    
    feature_matrix.loc[feature_matrix['Block Trial']==0, 'Block ID'] = np.arange(2,np.sum(feature_matrix['Block Trial']==0)+2)
    feature_matrix['Block ID'].ffill(inplace=True)
    feature_matrix['Block ID'].fillna(value=1, inplace=True)
    block_lengths = feature_matrix.groupby('Block ID').apply(lambda x: x['Block Trial'].max()).values.astype('int')+1
    feature_matrix['Current Block Length'] = np.repeat(block_lengths, block_lengths)[n_indi+1:]
    feature_matrix['session_pos'] = session_pos = feature_matrix.Trial/feature_matrix.Trial.max()            

    
    return feature_matrix