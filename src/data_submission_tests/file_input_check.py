import os
import pandas as pd
import fcsparser
import fcswrite
from pathlib import Path

csv_file_path = input(str('Enter path to .csv of metadata: '))
directory_path = input(str('Enter path to directory containing .fcs files: '))
directory_files = os.listdir(directory_path)


def test_file_size(directory_files, directory_path):
    '''Tests file size of .fcs files in dir - if less that 1kb test fails (indicates .fcs corruption)'''
    file_size_dict = {}
    for file in directory_files:
        if file.endswith('.fcs'):
            fcs_file_path = os.path.join(directory_path, file)
            fcs_file_size = os.path.getsize(fcs_file_path)
            file_size_dict[fcs_file_path] = fcs_file_size
    fcs_file_errored = [key for key, size in file_size_dict.items() if size <= 1]
    if fcs_file_errored:
        print("FILE SUBMISSION FAILED SIZE TEST")
        print("The following file(s) do not meet size criteria: ", fcs_file_errored)
    else:
        print("FILE SUBMISSION SUCCEEDED SIZE TEST")


def test_file_vs_meta(csv_file_path, directory_files):
    '''Reads in the names of the .fcs file names and compare to meta_data file entries to ensure matching entries'''
    df = pd.read_csv(csv_file_path, encoding='utf-8-sig')
    csv_filenames = df['Filename'].tolist()
    
    common_files = [file for file in csv_filenames if file in directory_files]
    missing_in_dir = [file for file in csv_filenames if file not in directory_files]
    extra_in_dir = [file for file in directory_files if file not in csv_filenames]

    if missing_in_dir or extra_in_dir:
        print("FILE SUBMISSION FAILED META VS DIRECTORY NAME TEST")
        print("Missing in directory:", missing_in_dir)
        print('#'*20)
        print('#'*20)
        print('#'*20)
        print("Not in CSV:", extra_in_dir)
    elif common_files is None:
        print("FILE SUBMISSION FAILED META VS DIRECTORY NAME TEST")
        print("No common files found, filenames do not match between os files and metadata list")
    else:
        print("FILE SUBMISSION SUCCEEDED META VS DIRECTORY NAME TEST")
    return missing_in_dir, extra_in_dir


def fcs_colnames_in_dir_to_df(directory_files, directory_path):
    '''Gets column names of all fcs files in fcs file dir - holds data in dataframe'''
    marker_names_dict = {}
    for file in directory_files:
        if file.endswith('.fcs'):
            file_path = os.path.join(directory_path, file)
            meta, data = fcsparser.parse(file_path, reformat_meta=True)
            fcs_file_dataframe = pd.DataFrame(data)
            marker_names_dict[file] = fcs_file_dataframe.columns.tolist()
    column_names_df = pd.DataFrame.from_dict(marker_names_dict, orient='index') #df containing column names of each fcs file with the file name as the key and thus index
    return column_names_df, marker_names_dict


def test_shared_fcs_colnames_entry(column_names_df):
    '''Finds most_common_marker_name per column in marker_names_dict - finds the specific entries where entry !=most_common_marker_name and reports to user'''
    most_common_marker_name = {} # most_common_marker_name dict: key = col index, value = most common colname (via mode), each dict entry = iterated across each column
    bad_marker_entries = {} # bad_marker_entries dict: key = col index, value = the bad_entry - can use the most_common_marker_name to understand what it should be.
    bad_marker_fcs = {}

    for col_index in range(len(column_names_df.columns)):
        most_common = column_names_df.iloc[:, col_index].mode()[0] 
        most_common_marker_name[col_index] = most_common

        mismatches = column_names_df[column_names_df.iloc[:, col_index] != most_common]
        if not mismatches.empty:
            bad_marker_entries[col_index] = mismatches.index.tolist()
            mismatch_store = mismatches
    for idx, row in mismatch_store.iterrows():
        bad_marker_fcs[idx] = row.tolist()
    bad_marker_fcs 
    
    '''Report to user what most common marker name is per column, and what fcs files don't meet this standard'''
    if bad_marker_entries:
        bad_marker_entries_dict = {most_common_marker_name[key]: value for key, value in bad_marker_entries.items()}
        print("FILE SUBMISSION FAILED MARKER NAMING TEST")
        print("Please ensure all flourophore markers are *exactly* the same")
        print("Accepted naming scheme:", most_common_marker_name)
        print('Please find .csv of incorrect markers in the working directory')
        export = pd.DataFrame.from_dict(bad_marker_entries_dict, orient='index') #df containing column names of each fcs file with the file name as the key and thus index
        export.to_csv('bad_marker_entries.csv')
    else:
        print("FILE SUBMISSION SUCCEEDED MARKER NAME TEST")
    return most_common_marker_name, bad_marker_fcs
        
        
def fix_bad_marker_entries(most_common_marker_name, bad_marker_fcs):
    '''Finds the bad marker .fcs files in dir, takes the mode .fcs marker entries and applies them to a perfect_channel_name array, and applies that to the bad_marker .fcs files '''
    bad_marker_held_fcs_list = list(bad_marker_fcs.keys())
    print(f'Bad markers in .fcs files:{bad_marker_held_fcs_list}')
    bad_marker_fcs_in_dir = list(set(bad_marker_held_fcs_list) & set(directory_files))
    
    for fcs_file in bad_marker_fcs_in_dir:
        file_path = os.path.join(directory_path, fcs_file)
        meta, data = fcsparser.parse(file_path, reformat_meta=True)
        bad_marker_fcs_dataframe = pd.DataFrame(data)
        bad_marker_fcs_dataframe.rename(columns={bad_marker_fcs_dataframe.columns[i]: new_name for i, new_name in most_common_marker_name.items()}, inplace=True)
        fcswrite.write_fcs(file_path, most_common_marker_name, bad_marker_fcs_dataframe)
    print('Completed overwrite of markers in .fcs')
    
    
# Run code
test_file_size(directory_files, directory_path)
missing_in_dir, extra_in_dir = test_file_vs_meta(csv_file_path, directory_files)
#column_names_df, marker_names_dict = fcs_colnames_in_dir_to_df(directory_files, directory_path)
#most_common_marker_name, bad_marker_fcs = test_shared_fcs_colnames_entry(column_names_df)
#print('#'*20)
#print('#'*20)
#fix_bad_marker_entries(most_common_marker_name, bad_marker_fcs)
missing_in_dir_df = pd.DataFrame(missing_in_dir, columns=['missing in dir'])
extra_in_dir_df = pd.DataFrame(extra_in_dir, columns=['extra in dir'])

missing_in_dir_df.to_csv('missing_in_dir_df.csv')
extra_in_dir_df.to_csv('extra_in_dir_df.csv')

