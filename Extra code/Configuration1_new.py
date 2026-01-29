import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from matplotlib.ticker import MaxNLocator, FuncFormatter
import argparse
import sys

def main(data_directory, results_directory, k, strategy, distance_metric, alpha):
    print("Using GPU" if tf.config.list_physical_devices('GPU') else "Using CPU")

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

    def compute_distances(test_sample, train_rssi, distance_metric='cityblock', alpha=None):
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
            weights = 1 / (nearest_distances + 1e-12)
            return np.average(nearest_positions, axis=0, weights=weights)
        else:
            raise ValueError("Unsupported strategy. Choose 'unweighted' or 'weighted'.")

    def knn_positioning(train_rssi, train_coords, test_rssi, k, strategy='unweighted', distance_metric='cityblock', alpha=None):
        estimated_positions = []
        for test_sample in test_rssi:
            distances = compute_distances(test_sample, train_rssi, distance_metric, alpha=alpha)
            sorted_indices = np.argsort(distances)
            
            n_candidates = k
            while n_candidates < len(sorted_indices) and abs(distances[sorted_indices[n_candidates]] - distances[sorted_indices[n_candidates - 1]]) < 1e-12:
                n_candidates += 1
            
            all_nearest_indices = sorted_indices[:n_candidates]
            nearest_positions = train_coords[all_nearest_indices]
            nearest_distances = distances[all_nearest_indices]
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

    # Ensure results directory exists
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    mean_errors_list = []

    for base_name in ['TUT3']:
        print(f"Processing dataset: {base_name}")
        
        train_coord_file = os.path.join(data_directory, f"{base_name}_trncrd.csv")
        train_rssi_file = os.path.join(data_directory, f"{base_name}_trnrss.csv")
        test_coord_file = os.path.join(data_directory, f"{base_name}_tstcrd.csv")
        test_rssi_file = os.path.join(data_directory, f"{base_name}_tstrss.csv")
        
        if not (os.path.exists(train_coord_file) and os.path.exists(train_rssi_file) and os.path.exists(test_coord_file) and os.path.exists(test_rssi_file)):
            print(f"Missing files for {base_name}, skipping...")
            continue
        
        coord_columns = ['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']
        train_df_coord = pd.read_csv(train_coord_file, header=None, names=coord_columns)
        test_df_coord = pd.read_csv(test_coord_file, header=None, names=coord_columns)
        train_df_rssi = pd.read_csv(train_rssi_file, header=None)
        test_df_rssi = pd.read_csv(test_rssi_file, header=None)

        database_orig = {
            'trncrd': train_df_coord[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']].values,
            'tstcrd': test_df_coord[['Latitude', 'Longitude', 'Altitude', 'FloorID', 'BuildingID']].values,
            'trnrss': train_df_rssi.values,
            'tstrss': test_df_rssi.values
        }

        origBlds = np.unique(database_orig['trncrd'][:, 4])
        nblds = len(origBlds)
        database0 = remapBldDB(database_orig, origBlds, np.arange(1, nblds + 1))

        origFloors = np.unique(database_orig['trncrd'][:, 3])
        nfloors = len(origFloors)
        database0 = remapFloorDB(database_orig, origFloors, np.arange(1, nfloors + 1))

        defaultNonDetectedValue = 100
        minValueDetected = min(np.min(database0['trnrss']), np.min(database0['tstrss']))
        newNonDetectedValue = []

        if len(newNonDetectedValue) == 0:
            newNonDetectedValue = minValueDetected - 1

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
        
        if defaultNonDetectedValue != 0:
            database0 = replace_non_detected_values(database0, defaultNonDetectedValue, newNonDetectedValue)

        database = data_rep_positive(database0)

        database_cleaned = {
            'trncrd': np.array(database['trncrd']),
            'tstcrd': np.array(database['tstcrd']),
            'trnrss': np.array(database['trnrss']),
            'tstrss': np.array(database['tstrss'])
        }

        database_cleaned['trainingValidMacs'] = (database_cleaned['trnrss'] != defaultNonDetectedValue)
        database_cleaned['testValidMacs'] = (database_cleaned['tstrss'] != defaultNonDetectedValue)
       
        vecidxmacs = np.arange(database_cleaned['trnrss'].shape[1])
        vecidxTsamples = np.arange(database_cleaned['trnrss'].shape[0])
        vecidxVsamples = np.arange(database_cleaned['tstrss'].shape[0])

        validMacs = vecidxmacs[np.sum(database_cleaned['trainingValidMacs'], axis=0) > 0]

        database_cleaned['trnrsss'] = database_cleaned['trnrss'][:, validMacs]
        database_cleaned['trainingValidMacs'] = database_cleaned['trainingValidMacs'][:, validMacs]
        database_cleaned['tstrsss'] = database_cleaned['tstrss'][:, validMacs]
        database_cleaned['testValidMacs'] = database_cleaned['testValidMacs'][:, validMacs]

        train_rssi = database_cleaned['trnrsss']
        train_coords = database_cleaned['trncrd']
        test_rssi = database_cleaned['tstrsss']
        test_coords = database_cleaned['tstcrd']

        estimated_positions = knn_positioning(train_rssi, train_coords, test_rssi, k, strategy, distance_metric, alpha)

        mse = mean_squared_error(test_coords, estimated_positions)
        rmse = np.sqrt(mse)
        print(f"RMSE for {base_name}: {rmse}")

        mean_errors_list.append(rmse)

        plt.figure()
        plt.scatter(test_coords[:, 0], test_coords[:, 1], label='True Position')
        plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], label='Estimated Position')
        plt.xlabel('Latitude')
        plt.ylabel('Longitude')
        plt.title(f'{base_name} - True vs Estimated Positions')
        plt.legend()
        plt.savefig(os.path.join(results_directory, f"{base_name}_positions.png"))
        plt.close()

    print(f"Mean RMSE across datasets: {np.mean(mean_errors_list)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN-based indoor positioning system.')
    parser.add_argument('--data_directory', type=str, required=True, help='Directory containing the dataset files.')
    parser.add_argument('--results_directory', type=str, required=True, help='Directory to save the results.')
    parser.add_argument('--k', type=int, default=1, help='Number of neighbors for KNN.')
    parser.add_argument('--strategy', type=str, default='unweighted', choices=['unweighted', 'weighted'], help='KNN weighting strategy.')
    parser.add_argument('--distance_metric', type=str, default='cityblock', choices=['cityblock', 'euclidean', 'minkowski'], help='Distance metric for KNN.')
    parser.add_argument('--alpha', type=float, default=None, help='Alpha parameter for Minkowski distance. Required if distance_metric is minkowski.')

    args = parser.parse_args()
    main(args.data_directory, args.results_directory, args.k, args.strategy, args.distance_metric, args.alpha)
