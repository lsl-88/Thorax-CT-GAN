# Import the libraries
from Dataset import LTRC, LTRC_ARDS, UMM, UKSH
from Processing import PreProcess, PostProcess
from Summary import DataSummary

import time
from utils import *

# Set the directories
# data_dir = '/home/ubuntu/sl_root/Data'
data_dir = '/home/sailam/Desktop/MSNE/Thesis/Data'
# save_root_dir = '/home/ubuntu/sl_root/Processed_data'
save_root_dir = '/home/sailam/Desktop/MSNE/Thesis/Processed_data'

# Set the name of the Run
name = 'Run_17'

# LTRC (Healthy)

# Obtain the summary of LTRC (Healthy) dataset
data_summary = DataSummary(name)
LTRC_healthy_df = data_summary.LTRC_healthy_preprocess(data_dir=data_dir, save=True)
patient_list = LTRC_healthy_df['Patient ID'].to_list()

# Check if the patient has been processed
patient_list = check_existing_data(patient_list, save_root_dir)

# Initialize empty dictionary
process_cache = {}

for single_pat in patient_list:

    # Initialize the patient and empty dictionary
    subject = LTRC(single_pat, data_dir)
    patient_summary = {}

    # Initialize the processed date, time and start time
    processed_date = time.strftime('%d-%m-%Y')
    processed_time = time.strftime('%H:%M')
    start_time = time.process_time()

    # Load the patient dataset
    subject = subject.load(data_type='all', verbose=True)

    # Print the stats of data
    subject.print_stats

    # Initialize the pre-processing parameters
    preprocess_params = PreProcess(emphysema_va1=50, ventilated_val=300, poorly_vent_val=700, atelectatic_val=1000)

    # Print the preprocessed parameters
    preprocess_params.print_parameters

    # Generate the preprocessed data
    preprocess_params.full_label_map(subject)

    # Initialize the postprocessing parameters
    postprocess_params = PostProcess(save_root_dir, merge_channel=False)
    postprocess_params.remove_slices(subject, percentage=0.05)

    # Generate the postprocessed data
    postprocess_params.normalization(subject)

    # Save the postprocessed data
    postprocess_params.save(subject)
    end_time = time.process_time() - start_time

    # Cache the process data into nested dictionary
    patient_summary['Processed Date'] = processed_date
    patient_summary['Processed Time'] = processed_time
    patient_summary['Time Taken'] = round(end_time, 2)
    process_cache[single_pat] = patient_summary

    # Send email
    send_email(single_pat, info_type='data_processing')
    del subject

# Generate the process summary
data_summary.process_summary(patient_list, process_cache, dataset='LTRC_Healthy', save=True)

# LTRC (ARDS)

# Obtain the summary of LTRC (ARDS) dataset
data_summary = DataSummary(name)
LTRC_ARDS_df = data_summary.LTRC_ARDS_preprocess(data_dir=data_dir, save=True)
patient_list = LTRC_ARDS_df['Patient ID'].to_list()

# Check if the patient has been processed
patient_list = check_existing_data(patient_list, save_root_dir)

# Initialize empty dictionary
process_cache = {}

for single_pat in patient_list:

    # Initialize the patient and empty dictionary
    subject = LTRC_ARDS(single_pat, data_dir)
    patient_summary = {}

    # Initialize the processed date, time and start time
    processed_date = time.strftime('%d-%m-%Y')
    processed_time = time.strftime('%H:%M')
    start_time = time.process_time()

    # Load the patient dataset
    subject = subject.load(data_type='all', verbose=True)

    # Print the stats of data
    subject.print_stats

    # Initialize the preprocessing parameters
    preprocess_params = PreProcess(emphysema_va1=50, ventilated_val=300, poorly_vent_val=700, atelectatic_val=1000)

    # Print the preprocessed parameters
    preprocess_params.print_parameters

    # Generate the preprocessed data
    preprocess_params.full_label_map(subject)

    # Initialize the postprocessing parameters
    postprocess_params = PostProcess(save_root_dir, merge_channel=False)

    # Generate the postprocessed data
    postprocess_params.normalization(subject)

    # Save the postprocessed data
    postprocess_params.save(subject)
    end_time = time.process_time() - start_time

    # Cache the process data into nested dictionary
    patient_summary['Processed Date'] = processed_date
    patient_summary['Processed Time'] = processed_time
    patient_summary['Time Taken'] = round(end_time, 2)
    process_cache[single_pat] = patient_summary

    # Send email
    send_email(single_pat, info_type='data_processing')
    del subject

# Generate the process summary
data_summary.process_summary(patient_list, process_cache, dataset='LTRC_ARDS', save=True)

# UMM

# Obtain the summary of UMM dataset
data_summary = DataSummary(name)
UMM_df = data_summary.UMM_preprocess(data_dir, save=True)
patient_list = UMM_df['Patient ID'].to_list()

# Check if the patient has been processed
patient_list = check_existing_data(patient_list, save_root_dir)

# Initialize empty dictionary
process_cache = {}

for single_pat in patient_list:

    # Initialize the patient and empty dictionary
    subject = UMM(single_pat, data_dir)
    patient_summary = {}

    # Initialize the processed date, time and start time
    processed_date = time.strftime('%d-%m-%Y')
    processed_time = time.strftime('%H:%M')
    start_time = time.process_time()

    # Load the patient dataset
    subject = subject.load(data_type='all', verbose=True)

    # Print the stats of data
    subject.print_stats

    # Initialize the preprocessing parameters
    preprocess_params = PreProcess(emphysema_va1=50, ventilated_val=300, poorly_vent_val=700, atelectatic_val=1000)

    # Print the preprocessed parameters
    preprocess_params.print_parameters

    # Generate the preprocessed data
    preprocess_params.full_label_map(subject)

    # Initialize the postprocessing parameters
    postprocess_params = PostProcess(save_root_dir, merge_channel=False)

    # Generate the postprocessed data
    postprocess_params.normalization(subject)

    # Save the postprocessed data
    postprocess_params.save(subject)
    end_time = time.process_time() - start_time

    # Cache the process data into nested dictionary
    patient_summary['Processed Date'] = processed_date
    patient_summary['Processed Time'] = processed_time
    patient_summary['Time Taken'] = round(end_time, 2)
    process_cache[single_pat] = patient_summary

    # Send email
    send_email(single_pat, info_type='data_processing')
    del subject

# Generate the process summary
data_summary.process_summary(patient_list, process_cache, dataset='UMM', save=True)

# UKSH

# Obtain the summary of UKSH dataset
data_summary = DataSummary(name)
UKSH_df = data_summary.UKSH_preprocess(data_dir, save=True)
patient_list = UKSH_df['Patient ID'].to_list()

# Check if the patient has been processed
patient_list = check_existing_data(patient_list, save_root_dir)

# Remove patients without segmentation map
try:
    patient_list.remove('A19')
except:
    pass
try:
    patient_list.remove('A18')
except:
    pass

# Initialize empty dictionary
process_cache = {}

for single_pat in patient_list:

    # Initialize the patient and empty dictionary
    subject = UKSH(single_pat, data_dir)
    patient_summary = {}

    # Initialize the processed date, time and start time
    processed_date = time.strftime('%d-%m-%Y')
    processed_time = time.strftime('%H:%M')
    start_time = time.process_time()

    # Load the patient dataset
    subject = subject.load(data_type='all', verbose=True)

    # Print the stats of data
    subject.print_stats

    # Initialize the preprocessing parameters
    preprocess_params = PreProcess(emphysema_va1=50, ventilated_val=300, poorly_vent_val=700, atelectatic_val=1000)

    # Print the preprocessed parameters
    preprocess_params.print_parameters

    # Generate the preprocessed data
    preprocess_params.full_label_map(subject)

    # Initialize the postprocessing parameters
    postprocess_params = PostProcess(save_root_dir, merge_channel=False)

    # Generate the postprocessed data
    postprocess_params.normalization(subject)
    postprocess_params.remove_slices(subject, percentage=0.05)

    # Save the postprocessed data
    postprocess_params.save(subject)
    end_time = time.process_time() - start_time

    # Cache the process data into nested dictionary
    patient_summary['Processed Date'] = processed_date
    patient_summary['Processed Time'] = processed_time
    patient_summary['Time Taken'] = round(end_time, 2)
    process_cache[single_pat] = patient_summary

    # Send email
    send_email(single_pat, info_type='data_processing')
    del subject

# Generate the process summary
data_summary.process_summary(patient_list, process_cache, dataset='UKSH', save=True)