import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functools

####### STUDENTS FILL THIS OUT ######

#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    # Rename columns in ndc_df
    ndc_df = ndc_df.rename(columns={'Non-proprietary Name': 'generic_drug_name',
                                    'NDC_Code': 'ndc_code'})
    
    # Merge dataframes on df
    df = pd.merge(df,
                  ndc_df[['ndc_code', 'generic_drug_name']],
                  how='left',
                  on='ndc_code')
    return df


#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    
    # Sort encounters, group by patient and select first encounter with head
    first_encounter_df = df.sort_values(['encounter_id'], ascending=True).groupby('patient_nbr').head(1)
    
    return first_encounter_df


#Question 6
def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id
    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''

    # Proportions train, validation, test
    p = [0.8, 0.2, 0.2]

    # Shuffle and unique values
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    
    # Split df into train and the rest
    sample_size = round(total_values * (1 - p[1] - p[2]))
    train = df[df[patient_key].isin(unique_values[:sample_size])].reset_index(drop=True)
    rest = df[df[patient_key].isin(unique_values[sample_size:])].reset_index(drop=True)
    
    # Split rest into validation and test
    p_val = p[1] / (p[1] + p[2])
    sample_size = round(len(rest) * p_val)
    validation = rest.iloc[:sample_size].reset_index(drop=True)
    test = rest.iloc[sample_size:].reset_index(drop=True)

    return train, validation, test


#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''

        # Count number of lines (unique values)
        with open(vocab_file_path, 'r') as fp:
            nlines = len(fp.readlines())

        # Define categorical column from file
        cat_column_ffile = tf.feature_column.categorical_column_with_vocabulary_file(key=c,
            vocabulary_file=vocab_file_path, vocabulary_size=nlines, num_oov_buckets=1)

        # Use embedding column if there's more than 18 unique values. This is the case of 'generic_drug_name'
        # and 'primary_diagnosis_code'. The rest of cases it uses indicator column (one-hot encoding)
        if nlines > 20:
            tf_categorical_feature_column = tf.feature_column.embedding_column(cat_column_ffile, dimension=300)
            print('Feature: {} - Unique values: {} - Embedding column'.format(c, nlines))
        else:
            tf_categorical_feature_column = tf.feature_column.indicator_column(cat_column_ffile)
            print('Feature: {} - Unique values: {} - One-hot column'.format(c, nlines))

        # Append column
        output_tf_list.append(tf_categorical_feature_column)

    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''

    # Normalizer with zscore
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD)

    # Define tf_numeric_feature as in the example
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)

    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
def get_student_binary_prediction(df, col):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[col].apply(lambda x: 1 if x >=5 else 0)
    return student_binary_prediction

