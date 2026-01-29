import os
import time
import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MaxNLocator
from scipy.spatial.distance import cdist
import sys

sys.path.append('../lets_talk_about_knn_code')

from configurations_functions_knn import (remap_vector,compute_weighted_centroid,remapBldDB, remapFloorDB, datarepNewNull, datarepNewNullDB,calculate_3d_positioning_error,replace_non_detected_values, data_rep_positive,compute_distances)
# List all GPUs
print("Using GPU" if tf.config.list_physical_devices('GPU') else "Using CPU")

# Define file directory
data_directory = '../lets_talk_about_knn_code/dataset'
results_directory = '../lets_talk_about_knn_code'
results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'knn_plain2024', 'C3')

# Ensure results directory exists
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

datarep = 'positive'
k_values = list(range(1, 21)) + list(range(21, 53, 2))
distance_metrics = ['cityblock', 'euclidean', 'minkowski3', 'cosine', 'sorensen', 'neyman', 'neyman2', 'lgd', 'plgd10','plgd40']
strategy = 'unweighted'
alpha = 0.0001

# Initialize the list for mean errors
mean_errors_list = []
best_configs_dict = {}  # dictionary to keep track of the best configuration for each dataset

# Iterate over all base names in the directory
for base_name in ['DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'SAH1', 'SIM001', 'TIE1', 'TUT1','TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7', 'UJI1', 'UTS1', 'GPR00' , 'GPR01' , 'GPR02', 'GPR03', 'GPR04', 'GPR05', 'GPR06', 'GPR07', 'GPR08', 'GPR09', 'GPR10', 'GPR11', 'GPR12', 'GPR13' ,'SOD01', 'SOD02', 'SOD03', 'SOD04', 'SOD05', 'SOD06', 'SOD07', 'SOD08', 'SOD09', 'UEXB1', 'UEXB2', 'UEXB3', 'UJIB1', 'UJIB2']:  # Add more base names as needed
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
    # added code to handle missing data from the sensors and handling the floor and building

    # Integrate database handling
    database_main = {
        'trncrd': train_df_coord[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']].values,
        'tstcrd': test_df_coord[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']].values,
        'trnrss': train_df_rssi.values,
        'tstrss': test_df_rssi.values
    }
    
    # Create an independent deep copy of the original database
    database_orig = copy.deepcopy(database_main)

    # Remap building and floor IDs
    origBlds = np.unique(database_orig['trncrd'][:, 4])
    nblds = len(origBlds)
    database0 = remapBldDB(database_orig, origBlds, np.arange(1, nblds + 1))
    
    origFloors = np.unique(database_orig['trncrd'][:, 3])
    nfloors = len(origFloors)
    database0 = remapFloorDB(database0, origFloors, np.arange(1, nfloors + 1))
    
    # Define non-detected values
    defaultNonDetectedValue = 100
    newNonDetectedValue = [] 
    
    # Handle non-detected RSSI values
    minValueDetected = min(np.min(database0['trnrss']), np.min(database0['tstrss']))
        
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
    if defaultNonDetectedValue != 0: 
        database0 = replace_non_detected_values(database0, defaultNonDetectedValue, newNonDetectedValue)

    # Check if datarep is 'positive'
    if datarep == 'positive':
        database = data_rep_positive(database0)
        if 'plgd10' in distance_metrics or 'plgd40' in distance_metrics:
            additionalparams =  -85 - newNonDetectedValue
        else:
            additionalparams = 0
    
    rsamples1 = database['trnrss'].shape[0]
    osamples1 = database['tstrss'].shape[0]
    nmacs1 = database['tstrss'].shape[1]
    
    # Create boolean arrays to indicate valid Mac addresses (APs)
    database['trainingValidMacs'] = (database_main['trnrss'] != defaultNonDetectedValue) 
    database['testValidMacs'] = (database_main['tstrss'] != defaultNonDetectedValue)
    
    # Count and get the total number of valid access points in both training and test data
    trainingValidMacs = (database['trainingValidMacs'].sum(axis=0) > 0).sum()
    testValidMacs = (database['testValidMacs'].sum(axis=0) > 0).sum()
    
    # Print the results
    print(f'Total valid access points in training data: {trainingValidMacs}')
    print(f'Total valid access points in test data: {testValidMacs}')
    
    vecidxmacs = np.arange(nmacs1)
    vecidxTsamples = np.arange(rsamples1)
    vecidxVsamples = np.arange(osamples1)
    
    validMacs = vecidxmacs[np.sum(database['trainingValidMacs'], axis=0) > 0]
    
    # Keep only the valid Mac addresses
    database['trnrss'] = database['trnrss'][:, validMacs]
    database['trainingValidMacs'] = database['trainingValidMacs'][:, validMacs]
    database['tstrss'] = database['tstrss'][:, validMacs]
    database['testValidMacs'] = database['testValidMacs'][:, validMacs]

    # Clean void fingerprints
    validTSamples = vecidxTsamples[np.sum(database['trainingValidMacs'], axis=1) > 0]
    #print(f'Total valid samples in training: {(np.sum(database["trainingValidMacs"], axis=1) > 0).sum()}')
    
    database['trnrss'] = database['trnrss'][validTSamples, :]
    database['trainingValidMacs'] = database['trainingValidMacs'][validTSamples, :]
    database['trncrd'] = database['trncrd'][validTSamples, :]


    validVSamples = vecidxVsamples[np.sum(database['testValidMacs'], axis=1) > 0]
    #print(f'Total valid fingerprints in testing: {(np.sum(database["testValidMacs"], axis=1) > 0).sum()}')

    database['tstrss'] = database['tstrss'][validVSamples, :]
    database['testValidMacs'] = database['testValidMacs'][validVSamples, :]
    database['tstcrd'] = database['tstcrd'][validVSamples, :]
   
    assert database['trnrss'].shape[0] == database['trncrd'].shape[0], "Mismatch in number of training samples."
    assert database['tstrss'].shape[0] == database['tstcrd'].shape[0], "Mismatch in number of test samples."

    trainingMacs = database['trnrss']
    testMacs = database['trnrss']
    trainingLabels = database['tstcrd']
    
    # Creating dataframe
    train_df_rssi_cleaned = pd.DataFrame(database['trnrss'])
    test_df_rssi_cleaned = pd.DataFrame(database['tstrss'])
    train_df_coord_cleaned = pd.DataFrame(database['trncrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID'])
    test_df_coord_cleaned = pd.DataFrame(database['tstcrd'], columns=['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID'])

    rsamples = database['trnrss'].shape[0]
    osamples = database['tstrss'].shape[0]
    nmacs = database['tstrss'].shape[1]

    # Combine training data
    train_df_combined = pd.concat([train_df_coord_cleaned[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']], train_df_rssi_cleaned], axis=1)
    X_train = train_df_combined.iloc[:, 5:]
    y_train = train_df_combined[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']]
    
    test_rssi = test_df_rssi_cleaned.iloc[:, 5:]# .iloc[50:101, 10:21] first is for row range and 2nd is for column range
    test_df_combined = pd.concat([test_df_coord_cleaned[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']], test_df_rssi_cleaned], axis=1)  
       
    # Perform K-NN positioning and error calculation
    start_time = time.perf_counter()
    best_errors = {}
    # Iterate over distance metrics
    for k in k_values:
        for distance_metric in distance_metrics:
            print(f"Running k={k}, strategy={strategy}, distance_metric={distance_metric}")
            
            estimated_positions = []
            estimated_buildings = []
            estimated_floors = []
            points_list = []
            for i in range(test_rssi.shape[0]):
                    
                # Step 1: Extract valid MAC addresses for the current test sample
                ofp_v = database['testValidMacs'][i, :]  # Valid MACs for the current test sample
                
                # Step 2: Filter training samples based on valid MACs
                valid_indices = np.sum(database['trainingValidMacs'][:, ofp_v], axis=1) > 0
                samples = vecidxTsamples[:len(valid_indices)][valid_indices]     
                #samples = np.where(valid_indices)[0]# give indics of the valid samples

                trainingMacs = database['trnrss'][samples, :]
                trainingValidMacs = database['trainingValidMacs'][samples, :]  
                trainingLabels = database['trncrd'][samples, :]

                rsamplesreduced = database['trnrss'].shape[0]  # Number of reduced samples
                
                # Step 3: Extract the current test sample's RSSI vector
                ofp = database['tstrss'][i, :]
                opf1 = np.tile(ofp, (rsamplesreduced, 1))
        
                # Step 4: Compute distances between the test sample and filtered training samples
                distances = compute_distances(trainingMacs, ofp, distance_metric,additionalparams)# alpha=alpha)

                # Step 5: Sort distances and find k nearest neighbors
                sorted_indices = np.argsort(distances)  # Sort by distance
                distancessort = distances[sorted_indices]  # Sorted distances
                candidates = sorted_indices  # Candidate indices in sorted order

                # Step 6: Determine the number of candidates (k)
                n_candidates = min(k, len(distances))

                # Check for ties in distances and expand the number of candidates if needed
                while n_candidates < len(distancessort) - 1:
                    if abs(distancessort[n_candidates - 1] - distancessort[n_candidates]) < 1e-12:
                        n_candidates += 1
                    else:
                        break
                
                # Step 7: Get the nearest positions, buildings, and floors
                nearest_indices = candidates[:n_candidates]
                nearest_positions = trainingLabels[nearest_indices, :3]
                nearest_buildings = trainingLabels[nearest_indices, 4]
                nearest_floors = trainingLabels[nearest_indices, 3]
                
                # Step 8: Estimate the building, floor, and position
                estimated_building = np.round(np.mean(nearest_buildings))
                estimated_floor = np.round(np.mean(nearest_floors))
                estimated_position = compute_weighted_centroid(nearest_positions, distancessort[:n_candidates], strategy)

                # Append results
                estimated_positions.append(estimated_position)
                estimated_buildings.append(estimated_building)
                estimated_floors.append(estimated_floor)
                points_list.append(n_candidates)

            # Step 7: Remap buildings and floors to real-world values
            real_world_buildings = remap_vector(np.array(estimated_buildings), np.arange(1, nblds + 1), origBlds)
            real_world_floors = remap_vector(np.array(estimated_floors), np.arange(1, nfloors + 1), origFloors)

            # After the loop, create a dictionary to store the results
            y_test_pred = {
                'positions': np.array(estimated_positions),
                'buildings': np.array(estimated_buildings),
                'floors': np.array(estimated_floors),
                'points': np.array(points_list)
            }

            # Calculate prediction errors for each sample
            test_errors = calculate_3d_positioning_error(test_df_coord_cleaned[['Latitude', 'Longitude', 'Altitude']].values, y_test_pred['positions'])
            mean_error = np.mean(test_errors)
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            
            #mean_errors_list.append({'Dataset Name': base_name, 'k': k, 'Strategy': strategy, 'Distance Metric': distance_metric, 'Mean Error': np.round(mean_error, 2)})
            mean_errors_list.append({'Dataset Name': base_name, 'Mean Error': np.round(mean_error, 2)})
            #print('mean error',mean_error)
            
            # Update best configuration for each dataset and distance metric
            config_key = (base_name, distance_metric)
            if config_key not in best_errors or best_errors[config_key]['Mean Error'] > mean_error:
                best_errors[config_key] = {'k': k, 'Strategy': strategy, 'Distance Metric': distance_metric, 'Mean Error': mean_error}

            # Create a DataFrame to store errors, actual coordinates, and predicted coordinates
            results_df = pd.DataFrame({
                'Latitude_pred': y_test_pred['positions'][:, 0],
                'Longitude_pred': y_test_pred['positions'][:, 1],
                'Altitude_pred': y_test_pred['positions'][:, 2],
            })

            print(f'\nRunning the algorithm with {base_name}')
            print(f'    database features pre  : [{rsamples1},{osamples1},{nmacs1}]')
            print(f'    database New features  : [{rsamples},{osamples},{nmacs}]')
            print(f'    k                      : {k}')
            #print(f'    datarep                : {datarep}')
            print(f'    minValueDetected       : {minValueDetected }')
            print(f'    defaultNonDetectedValue: {defaultNonDetectedValue}')
            print(f'    newNonDetectedValue    : {newNonDetectedValue}')
            print(f'    distanceMetric         : {distance_metric}')
            print(f'    Data Set used          : {base_name} ')
            print(f'    3D Positioning Error   : {np.round(mean_error, 2)} m')

            # Create subfolder for the dataset within "results"
            dataset_results_directory = os.path.join(results_directory, base_name, f"distance_{distance_metric}_k{k:03d}")
            if not os.path.exists(dataset_results_directory):
                os.makedirs(dataset_results_directory)

            # Save the predictions to CSV files
            predictions_file = os.path.join(dataset_results_directory, f"predictions_{base_name}_k{k:03d}_{distance_metric}.csv")
            results_df.to_csv(predictions_file, index=False, header=None)
            #print(f'Saved predicted coordinates to {predictions_file}')

            # Create DataFrame for errors with rounded Error values
            error_df = pd.DataFrame({'Error': np.round(test_errors, 2)})  # Round errors to 7 decimal places
            error_file = os.path.join(dataset_results_directory, f"errors_{base_name}_k{k:03d}_{distance_metric}.csv")
            error_df.to_csv(error_file, index=False, header=None)
            #print(f'Saved prediction errors to {error_file}')
                    
            # Update the best configuration dictionary
            for key, value in best_errors.items():
                base_name, distance_metric = key
                if base_name not in best_configs_dict or best_configs_dict[base_name]['Mean Error'] > value['Mean Error']:
                    best_configs_dict[base_name] = value

            # # Plotting the actual vs predicted coordinates for test set
            # plt.figure(figsize=(10, 8))
            # sc = plt.scatter(test_df_coord_cleaned['Longitude'], test_df_coord_cleaned['Latitude'], c=test_df_coord_cleaned['Altitude'], cmap='viridis', label='Actual', alpha=0.6)
            # plt.scatter(y_test_pred[:, 1], y_test_pred[:, 0], c=y_test_pred[:, 2], cmap='coolwarm', label='Predicted', alpha=0.6)
            # plt.colorbar(sc, label='Altitude')
            # plt.title(f'Test: Actual vs Predicted Coordinates (k={k}, {base_name}, {strategy}, {distance_metric})')
            # plt.xlabel('Longitude')
            # plt.ylabel('Latitude')
            # plt.legend()
            # plt.savefig(os.path.join(dataset_results_directory, f"actual_vs_predicted_{base_name}_k{k:03d}_{distance_metric}.png"))
            # plt.close()

            # # Visualize the prediction errors for test set
            # plt.figure(figsize=(10, 8))
            # plt.quiver(test_df_coord_cleaned['Longitude'], test_df_coord_cleaned['Latitude'], y_test_pred[:, 1] - test_df_coord_cleaned['Longitude'], y_test_pred[:, 0] - test_df_coord_cleaned['Latitude'],
            #         angles='xy', scale_units='xy', scale=1, color='red', alpha=0.6)
            # plt.title(f'Test: Prediction Errors (Longitude and Latitude) (k={k}, {base_name}, {strategy}, {distance_metric})')
            # plt.xlabel('Longitude')
            # plt.ylabel('Latitude')
            # plt.savefig(os.path.join(dataset_results_directory, f"prediction_errors_{base_name}_k{k:03d}_{distance_metric}.png"))
            # plt.close()


# Create a DataFrame for mean errors
mean_errors_df = pd.DataFrame(mean_errors_list)
mean_errors_csv_path = os.path.join(results_directory, 'mean_errors.csv')
mean_errors_df.to_csv(mean_errors_csv_path, index=False)


# Convert best configurations to DataFrame and save
best_configs_df = pd.DataFrame.from_dict(best_configs_dict, orient='index')
best_configs_csv_path = os.path.join(results_directory, 'best_configs.csv')
best_configs_df.to_csv(best_configs_csv_path, index=True)
