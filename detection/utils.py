import re
def append_single_utterances(dataset_df): 
    '''
    From a dataframe containing full dialogues, extract all single user utterances and append to the dataframe.
    
    Input: 
        * dataset_df = dataframe containing 'label', 'dia_no', and 'dia'
    
    Usage: 
        ```
        dataset_df = append_single_utterances(dataset_df)
        ```
    '''

    pattern = re.compile(f'^user: ?(.*)\s*(?:\n|$)', re.MULTILINE)

    for i, row in dataset_df.iterrows(): 
        cur_label = row['label']
        cur_dia_no = row['dia_no']
        user_utterances = ['user: '+user_utterance.group(1) for user_utterance in re.finditer(pattern, row['dia']) if user_utterance.group(1).strip()]
        for utterance in user_utterances: 
            dataset_df.loc[dataset_df.shape[0]] = {'label': cur_label, 
                                                    'dia': utterance, 
                                                    'dia_no': cur_dia_no}
    return dataset_df

def append_progressing_utterances(dataset_df): 
    '''
    From a dataframe containing full dialogues, extract all single user utterances and append to the dataframe.
    
    Input: 
        * dataset_df = dataframe containing 'label', 'dia_no', and 'dia'
    
    Usage: 
        ```
        dataset_df = append_single_utterances(dataset_df)
        ```
    '''
    for i, row in dataset_df.iterrows(): 
        cur_label = row['label']
        cur_dia_no = row['dia_no']
        temp = [_.strip() for _ in row['dia'].split("\n")]
        if temp[0].startswith("sys"): temp = temp[1:]
        for i in range(1,len(temp)+1,2): 
            dataset_df.loc[dataset_df.shape[0]] = {'label': cur_label, 
                                                    'dia': '\n'.join(temp[:i]), 
                                                    'dia_no': cur_dia_no}
    return dataset_df