import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MaxNLocator,FuncFormatter
import sys

sys.path.append('/lets_talk_about_knn_code')

print("Using GPU" if tf.config.list_physical_devices('GPU') else "Using CPU")

#from configurations_functions_knn import (remapBldDB, remapFloorDB)#knn_positioning,, datarepNewNullDB, calculate_3d_positioning_error_org

# Define functions for remapping IDs
def remapBldDB(database, origBlds, newBlds):
    mapping = dict(zip(origBlds, newBlds))
    for key in ['trncrd', 'tstcrd']:
        database[key][:, 4] = np.array([mapping.get(bld, bld) for bld in database[key][:, 4]])
    return database

def remapFloorDB(database, origFloors, newFloors):
    mapping = dict(zip(origFloors, newFloors))
    for key in ['trncrd', 'tstcrd']:
        database[key][:, 3] = np.array([mapping.get(floor, floor) for floor in database[key][:, 3]])
    return database

def calculate_3d_positioning_error(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

# Set print options to display the entire array, comment it if dont want to see whole array in ouput
#np.set_printoptions(threshold=np.inf) # is variable threshold=1000

def compute_distances(test_sample, train_rssi, distance_metric='cityblock', alpha=None):
    # Placeholder for distance computation
    if distance_metric == 'cityblock':
        return np.sum(np.abs(train_rssi - test_sample), axis=1)
    elif distance_metric == 'euclidean':
        return np.sqrt(np.sum((train_rssi - test_sample) ** 2, axis=1))
    elif distance_metric == 'minkowski' and alpha is not None:
        return np.sum(np.abs(train_rssi - test_sample) ** alpha, axis=1) ** (1 / alpha)
    else:
        raise ValueError("Unsupported distance metric or missing alpha for Minkowski distance.")


def compute_weighted_centroid(nearest_positions, nearest_distances, strategy='unweighted'):
    if strategy == 'unweighted':
        return np.mean(nearest_positions, axis=0)
    elif strategy == 'weighted':
        weights = 1 / (nearest_distances + 1e-12)  # Adding a small value to avoid division by zero
        return np.average(nearest_positions, axis=0, weights=weights)
    else:
        raise ValueError("Unsupported strategy. Choose 'unweighted' or 'weighted'.")

def knn_positioning(train_rssi, train_coords, test_rssi, k, strategy='unweighted', distance_metric='cityblock', alpha=None):
    estimated_positions = []
    for test_sample in test_rssi:
        # Compute distances between the test sample and all training samples
        distances = compute_distances(test_sample, train_rssi, distance_metric, alpha=alpha)
        sorted_indices = np.argsort(distances)  # Indices of distances in ascending order
        
        n_candidates = k
        while n_candidates < len(sorted_indices) and abs(distances[sorted_indices[n_candidates]] - distances[sorted_indices[n_candidates - 1]]) < 1e-12:
            n_candidates += 1
        
        # Get all indices up to n_candidates
        all_nearest_indices = sorted_indices[:n_candidates]
        nearest_positions = train_coords[all_nearest_indices]
        nearest_distances = distances[all_nearest_indices]
        
        # Compute the estimated position using the weighted centroid method
        estimated_position = compute_weighted_centroid(nearest_positions, nearest_distances, strategy)
        estimated_positions.append(estimated_position)
    
    return np.array(estimated_positions)

def replace_non_detected_values(database, default_value, new_value):
    database['trnrss'][database['trnrss'] == default_value] = new_value
    database['tstrss'][database['tstrss'] == default_value] = new_value
    return database
    
def data_rep_positive(database):
    min_rssi = min(train_df_rssi.min().min(), test_df_rssi.min().min())
    shift_value = max(0, -min_rssi)
    database['trnrss'] += shift_value
    database['tstrss'] += shift_value
    return database

# Define file directory
data_directory = '/lets_talk_about_knn_code/dataset'
results_directory = '/lets_talk_about_knn_code'
results_directory = os.path.join(results_directory,'Results and analysis', 'Results_pos_err', 'knn_plain2024', 'C1_test')

# Ensure results directory exists
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

mean_errors_list = []

# Iterate over all base names in the directory
for base_name in ['TUT3']:#['DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'SAH1', 'SIM001', 'TIE1', 'TUT1','TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7', 'UJI1', 'UTS1']:  # Add more base names as needed
    print(f"Processing dataset: {base_name}")
    
    train_coord_file = os.path.join(data_directory, f"{base_name}_trncrd.csv")
    train_rssi_file = os.path.join(data_directory, f"{base_name}_trnrss.csv")
    test_coord_file = os.path.join(data_directory, f"{base_name}_tstcrd.csv")
    test_rssi_file = os.path.join(data_directory, f"{base_name}_tstrss.csv")
    
    # Check if all required files exist
    if not (os.path.exists(train_coord_file) and os.path.exists(train_rssi_file) and os.path.exists(test_coord_file) and os.path.exists(test_rssi_file)):
        print(f"Missing files for {base_name}, skipping...")
        continue
    
    # Load coordinate data
    coord_columns = ['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']
    train_df_coord = pd.read_csv(train_coord_file, header=None, names=coord_columns)
    test_df_coord = pd.read_csv(test_coord_file, header=None, names=coord_columns)
    
    # Load RSSI signal data
    train_df_rssi = pd.read_csv(train_rssi_file, header=None)
    test_df_rssi = pd.read_csv(test_rssi_file, header=None)


####################################################################
#added code to handle missing data from the sensors and handling the foolr and building

    # Integrate database handling
    database_orig = {
        'trncrd': train_df_coord[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']].values,
        'tstcrd': test_df_coord[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']].values,
        'trnrss': train_df_rssi.values,
        'tstrss': test_df_rssi.values
    }

    # Remap building and floor IDs
    origBlds = np.unique(database_orig['trncrd'][:, 4])
    nblds = len(origBlds)
    database0 = remapBldDB(database_orig, origBlds, np.arange(1, nblds + 1))

    origFloors = np.unique(database_orig['trncrd'][:, 3])
    nfloors = len(origFloors)
    database0 = remapFloorDB(database_orig, origFloors, np.arange(1, nfloors + 1))
    

    # Define non-detected values
    defaultNonDetectedValue = 100
    #defaultNonDetectedValue = np.array([100])
        
    # Handle non-detected RSSI values
    minValueDetected = min(np.min(database0['trnrss']), np.min(database0['tstrss']))
    newNonDetectedValue = []

    if len(newNonDetectedValue) == 0:
        newNonDetectedValue = minValueDetected -1

    #Manual fix for WGS84 datasets and UEXBx datasets
    if np.min(database0['trnrss']) == -200:
        defaultNonDetectedValue = -200
        newNonDetectedValue = -200

    if np.min(database0['trnrss']) == -110 and np.max(database0['trnrss']) < 0:
        idxT = database0['trnrss'] <= -109
        idxV = database0['tstrss'] <= -109

        database_orig['trnrss'][idxT] = -110
        database_orig['tstrss'][idxV] = -110

        database0['trnrss'][idxT] = -110
        database0['tstrss'][idxV] = -110

        defaultNonDetectedValue = -110
        newNonDetectedValue = -110

    if np.min(database0['trnrss']) == -109 and np.max(database0['trnrss']) < 0:
        idxT = database0['trnrss'] <= -108
        idxV = database0['tstrss'] <= -108

        database_orig['trnrss'][idxT] = -109
        database_orig['tstrss'][idxV] = -109

        database0['trnrss'][idxT] = -109
        database0['tstrss'][idxV] = -109

        defaultNonDetectedValue = -109
        newNonDetectedValue = -109
        
    #Handling Non detected values
    
    if defaultNonDetectedValue != 0: #removed .size
        database0 = replace_non_detected_values(database0, defaultNonDetectedValue, newNonDetectedValue)
     
    # Processing data to make it positive
    database = data_rep_positive(database0)

    database_cleaned = {
        'trncrd': np.array(database['trncrd']),
        'tstcrd': np.array(database['tstcrd']),
        'trnrss': np.array(database['trnrss']),
        'tstrss': np.array(database['tstrss'])
    }

    # Create boolean arrays to indicate valid Mac addresses (APs)
    database_cleaned['trainingValidMacs'] = (database_cleaned['trnrss'] != defaultNonDetectedValue)
    database_cleaned['testValidMacs'] = (database_cleaned['tstrss'] != defaultNonDetectedValue)
   
    vecidxmacs = np.arange(database_cleaned['trnrss'].shape[1])
    vecidxTsamples = np.arange(database_cleaned['trnrss'].shape[0])
    vecidxVsamples = np.arange(database_cleaned['tstrss'].shape[0])

    validMacs = vecidxmacs[np.sum(database_cleaned['trainingValidMacs'], axis=0) > 0]

    # Keep only the valid Mac addresses
    database_cleaned['trnrsss'] = database_cleaned['trnrss'][:, validMacs]
    database_cleaned['trainingValidMacs'] = database_cleaned['trainingValidMacs'][:, validMacs]
    database_cleaned['tstrss'] = database_cleaned['tstrss'][:, validMacs]
    database_cleaned['testValidMacs'] = database_cleaned['testValidMacs'][:, validMacs]

    # Clean void fingerprints
    validTSamples = vecidxTsamples[np.sum(database_cleaned['trainingValidMacs'], axis=1) > 0]
    database_cleaned['trnrss'] = database_cleaned['trnrss'][validTSamples, :]
    database_cleaned['trainingValidMacs'] = database_cleaned['trainingValidMacs'][validTSamples, :]
    database_cleaned['trncrds'] = database_cleaned['trncrd'][validTSamples, :]

    validVSamples = vecidxVsamples[np.sum(database_cleaned['testValidMacs'], axis=1) > 0]
    database_cleaned['tstrss'] = database_cleaned['tstrss'][validVSamples, :]
    database_cleaned['testValidMacs'] = database_cleaned['testValidMacs'][validVSamples, :]
    database_cleaned['tstcrd'] = database_cleaned['tstcrd'][validVSamples, :]

    # Convert cleaned dataframes to DataFrames if needed
    train_df_rssi_cleaned = pd.DataFrame(database_cleaned['trnrss'])
    test_df_rssi_cleaned = pd.DataFrame(database_cleaned['tstrss'])

    # Convert cleaned coordinates to DataFrames if needed
    train_df_coord_cleaned = pd.DataFrame(database_cleaned['trncrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID'])
    test_df_coord_cleaned = pd.DataFrame(database_cleaned['tstcrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID'])

    rsamples = database_cleaned['trnrss'].shape[0]
    osamples = database_cleaned['tstrss'].shape[0]
    nmacs = database_cleaned['tstrss'].shape[1]

    # Combine training data
    train_df_combined = pd.concat([train_df_coord_cleaned[['Latitude', 'Longitude', 'Altitude']], train_df_rssi_cleaned], axis=1)

    X_train = train_df_combined.iloc[:, 3:]
    y_train = train_df_combined[['Latitude', 'Longitude', 'Altitude']]
    
    # Combine test data
    test_df_combined = pd.concat([test_df_coord_cleaned[['Latitude', 'Longitude', 'Altitude']], test_df_rssi_cleaned], axis=1)
  
    # Set specific values for k, strategy, and distance_metric
    k = 1
    strategy = 'unweighted'
    distance_metric = 'cityblock'
    alpha= 0.1

    print(f"Running k={k}, strategy={strategy}, distance_metric={distance_metric}")
    
    # Perform K-NN positioning and error calculation
    y_test_pred = knn_positioning(X_train.values, y_train.values, test_df_combined.iloc[:, 3:].values, k, strategy, distance_metric, alpha)
    test_errors = calculate_3d_positioning_error(test_df_coord_cleaned[['Latitude', 'Longitude', 'Altitude']].values, y_test_pred)
    mean_error = np.mean(test_errors)
    mean_errors_list.append({'Dataset Name': base_name, 'k': k, 'Strategy': strategy, 'Distance Metric': distance_metric, 'Mean Error': mean_error})
    print(f'{base_name} Test Mean 3D Positioning Error: {np.round(mean_error, 2)}')

    # Create a DataFrame to store errors, actual coordinates, and predicted coordinates
    results_df = pd.DataFrame({
        'Latitude_pred': y_test_pred[:, 0],
        'Longitude_pred': y_test_pred[:, 1],
        'Altitude_pred': y_test_pred[:, 2],
    })

    print(f'\nRunning the algorithm with')
    print(f'    database features      : [{rsamples},{osamples},{nmacs}]')
    print(f'    k                      : {k}')
    #print(f'    datarep                : {datarep}')
    print(f'    minValueDetected       : {minValueDetected }')
    print(f'    defaultNonDetectedValue: {defaultNonDetectedValue}')
    print(f'    newNonDetectedValue    : {newNonDetectedValue}')
    print(f'    distanceMetric         : {distance_metric}')
    

    # results = {
    #     'error': {np.round(mean_error, 2)},
    #     'prediction': np.zeros((database_cleaned['tstrss'].shape[0], 5)),
    #     'targets': np.zeros((database_cleaned['tstrss'].shape[0], 5)),
    #     'candidates': np.zeros((database_cleaned['tstrss'].shape[0], 1)),
    #     'distances': np.zeros((database_cleaned['tstrss'].shape[0], 1)),
    #     'timesample': np.zeros((database_cleaned['tstrss'].shape[0], 5)),
    #     'considered': np.zeros((database_cleaned['tstrss'].shape[0], 5))
    # }

    # Create subfolder for the dataset within "results"
    dataset_results_directory = os.path.join(results_directory, base_name, f"positive_distance_{distance_metric}_k{k:03d}")#_k{k:03d}_alpha{alpha} if using alpha list

    if not os.path.exists(dataset_results_directory):
        os.makedirs(dataset_results_directory)

    # Save the predictions to CSV files
    predictions_file = os.path.join(dataset_results_directory, f"predictions_{base_name}_k{k:03d}_{distance_metric}.csv")#_{distance_metric}_alpha{alpha}.csv
    results_df.to_csv(predictions_file, index=False, header=None)
    error_df = pd.DataFrame({'Error': np.round(test_errors, 2)})  # Round errors to 2 decimal places

    # Save the errors to CSV without index
    error_file = os.path.join(dataset_results_directory, f"errors_{base_name}_k{k:03d}_{distance_metric}.csv")#_{distance_metric}_alpha{alpha}.csv
    error_df.to_csv(error_file, index=False, header=None)
    mean_errors_df = pd.DataFrame(mean_errors_list)

    # Save the mean errors to CSV file
    mean_errors_summary_file = os.path.join(results_directory, f"mean_errors_summary{k}.csv")
    mean_errors_df.to_csv(mean_errors_summary_file, index=False)
    print(f'Saved mean errors summary to {mean_errors_summary_file}')
