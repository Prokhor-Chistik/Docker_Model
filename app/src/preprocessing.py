# Import libraries
import pandas as pd


def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file)

    return input_df


# Main preprocessing function
def run_preproc(input_df):

    output_df = input_df.drop(['client_id', 'mrg_'], axis=1)
    if 'binary_target' in output_df:
        output_df = output_df.drop(['binary_target'], axis=1)
    
    output_df[['регион', 'использование', 'pack']] = output_df[['регион', 'использование', 'pack']].fillna('Nan')

    # Return resulting dataset
    return output_df