import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def calculate_3d_positioning_error_org(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))

# Define a similarity function (example: cosine similarity)
def my_similarity_function(test_sample, train_samples):
    test_norm = np.linalg.norm(test_sample)
    train_norm = np.linalg.norm(train_samples, axis=1, keepdims=True)
    dot_product = np.dot(train_samples, test_sample)
    similarity_values = dot_product / (train_norm * test_norm)
    return similarity_values

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

# Define the function to replace old null values with new null values
def datarepNewNull(arr, old_null, new_null):
    return np.where(arr == old_null, new_null, arr)

def datarepNewNullDB(db0, old_null, new_null):
    db1 = {}
    db1['trnrss'] = datarepNewNull(db0['trnrss'], old_null, new_null)
    db1['tstrss'] = datarepNewNull(db0['tstrss'], old_null, new_null)
    db1['trncrd'] = db0['trncrd']
    db1['tstcrd'] = db0['tstcrd']
    return db1

def compute_distances(test_sample, train_samples, distance_metric='cityblock', similarity_function=None, alpha=None):
    epsilon = 1e-10  # Define epsilon inside the function for cosine distance
    if similarity_function is None:
        if distance_metric == 'cityblock':
            return np.sum(np.abs(train_samples - test_sample), axis=1)
        elif distance_metric == 'euclidean':
            return np.sqrt(np.sum((train_samples - test_sample) ** 2, axis=1))
        elif distance_metric == 'minkowski3':
            return np.sum(np.abs(train_samples - test_sample) ** 3, axis=1) ** (1/3)
        elif distance_metric == 'cosine':
            train_magnitude = np.sqrt(np.sum(train_samples ** 2, axis=1))
            test_magnitude = np.sqrt(np.sum(test_sample ** 2))
            train_magnitude = np.clip(train_magnitude, epsilon, None)
            test_magnitude = np.clip(test_magnitude, epsilon, None)
            cosine_similarity = np.sum(train_samples * test_sample, axis=1) / (train_magnitude * test_magnitude)
            return 1 - cosine_similarity
        elif distance_metric == 'sorensen':
            denom = np.sum(train_samples + test_sample, axis=1)
            denom = np.clip(denom, epsilon, None)  # Avoid division by zero
            return np.sum(np.abs(train_samples - test_sample), axis=1) / denom
        elif distance_metric == 'neyman':
            factorzero = 0.000001 if np.max(train_samples) <= 1.0001 and np.max(test_sample) <= 1.0001 else 0.0001
            divisor = test_sample + (factorzero * (test_sample == 0))
            return np.sum(((train_samples - test_sample) ** 2) / divisor, axis=1)
        elif distance_metric == 'neyman2':
            factorzero = 0.000001 if np.max(train_samples) <= 1.0001 and np.max(test_sample) <= 1.0001 else 0.0001
            divisor = train_samples + (factorzero * (train_samples == 0))
            return np.sum(((train_samples - test_sample) ** 2) / divisor, axis=1)
        elif distance_metric == 'lgd':
            sigma = 5
            threshold = 0.0001
            numerator = -((train_samples - test_sample) ** 2)
            denominator = 2 * (sigma ** 2)
            differences = (1 / np.sqrt(2 * np.pi * sigma ** 2)) * np.exp(numerator / denominator)
            nonzero = (train_samples * test_sample) != 0
            return -np.sum(np.log(differences * nonzero + threshold * (1 - nonzero)), axis=1)
        elif distance_metric == 'plgd':
            threshold = alpha  # Assuming 'alpha' is used as the threshold
            p1 = np.sum((train_samples - threshold) * (train_samples >= threshold) * (test_sample == 0), axis=1)
            p2 = np.sum((test_sample - threshold) * (test_sample >= threshold) * (train_samples == 0), axis=1)
            d_lgd = compute_distances(test_sample, train_samples, distance_metric='lgd')
            return d_lgd + (1 / 10) * (p1 + p2)  # Change 1/10 to 1/40 for PLGD40 if needed
        else:
            raise ValueError("Unknown distance metric.")
    else:
        similarity_values = similarity_function(test_sample, train_samples)
        if np.max(similarity_values) <= 1.0:
            distances = 1.0 - similarity_values
        else:
            epsilon = 1e-8
            distances = 1.0 / (similarity_values + epsilon)
        return distances

def compute_weighted_centroid(positions, distances, strategy='unweighted'):
    epsilon = 1e-8
    if strategy == 'unweighted':
        weights = np.ones(len(distances))
    elif strategy == 'inverse_distance':
        weights = 1 / (distances + epsilon)
    elif strategy == 'squared_inverse_distance':
        weights = 1 / (distances ** 2 + epsilon)
    else:
        raise ValueError("Unknown strategy.")
    weights /= weights.sum()
    weighted_positions = np.dot(weights, positions)
    return weighted_positions



def knn_positioning(train_rssi, train_coords, test_rssi, k, strategy='unweighted', distance_metric='cityblock', alpha=None):
    all_distances = np.apply_along_axis(compute_distances, 1, test_rssi, train_samples=train_rssi, distance_metric=distance_metric, alpha=alpha)
    sorted_indices = np.argsort(all_distances, axis=1)
    
    estimated_positions = []
    for i in range(test_rssi.shape[0]):
        distances = all_distances[i]
        sorted_indices = np.argsort(distances)  # Ascending order of the distance values
        nearest_indices = sorted_indices[:k]  # Indices of the first k smallest distances

        # Handle Ties in Distances
        k_distance = distances[nearest_indices[-1]]  # Distance of the k-th nearest neighbor
        additional_indices = np.where(distances == k_distance)[0]  # Identifies additional indices with the same distance as the k-th nearest neighbor
        all_nearest_indices = np.unique(np.concatenate((nearest_indices, additional_indices)))  # Combines the indices of the k-nearest neighbors and any additional indices to handle ties
        
        nearest_positions = train_coords[all_nearest_indices]
        nearest_distances = distances[all_nearest_indices]  # Retrieves the coordinates and distances of all nearest neighbors, including any tied distances
        estimated_position = compute_weighted_centroid(nearest_positions, nearest_distances, strategy)
        estimated_positions.append(estimated_position)
    
    return np.array(estimated_positions)


# Assuming compute_distances and compute_weighted_centroid are defined elsewhere


# List all GPUs
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPUs are available:")
    for gpu in gpus:
        print(f"- {gpu}")
else:
    print("No GPUs found. Using CPU.")

# Define file directory
data_directory = '../lets_talk_about_knn_code/dataset'
results_directory = '../lets_talk_about_knn_code'
results_directory = os.path.join(results_directory, 'Results and analysis', 'Results_pos_err', 'knn_plain2024', 'C2')

# Ensure results directory exists
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# Initialize the list for mean errors
mean_errors_list = []
best_configs_dict = {}  # Dictionary to keep track of the best configuration for each dataset

# Iterate over all base names in the directory
for base_name in ['DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'SAH1', 'SIM001', 'TIE1', 'TUT1', 'TUT2', 'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7', 'UJI1', 'UTS1']:  # Add more base names as needed
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
    # Handle missing data from the sensors and handling the floor and building

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
    database_orig = remapBldDB(database_orig, origBlds, np.arange(1, nblds + 1))

    origFloors = np.unique(database_orig['trncrd'][:, 3])
    nfloors = len(origFloors)
    database_orig = remapFloorDB(database_orig, origFloors, np.arange(1, nfloors + 1))

    # Define non-detected values and thresholds
    non_detected_values = [100, -200, -110, -109, -102, -104]
    max_threshold = 100

    # Replace non-detected values
    database_orig['trnrss'] = np.where(np.isin(database_orig['trnrss'], non_detected_values), -100, database_orig['trnrss'])
    database_orig['tstrss'] = np.where(np.isin(database_orig['tstrss'], non_detected_values), -100, database_orig['tstrss'])

    # Clip values greater than max_threshold to max_threshold
    database_orig['trnrss'] = np.clip(database_orig['trnrss'], None, max_threshold)
    database_orig['tstrss'] = np.clip(database_orig['tstrss'], None, max_threshold)

    # Handle non-detected RSSI values
    minValueDetected = min(np.min(database_orig['trnrss']), np.min(database_orig['tstrss']))
    newNonDetectedValue = None  # Default value if not handled specifically

    if newNonDetectedValue is None:
        newNonDetectedValue = minValueDetected - 1
        
    if np.min(database_orig['trnrss']) == -200:
        defaultNonDetectedValue = -200
        newNonDetectedValue = -200

    if np.min(database_orig['trnrss']) == -110 and np.max(database_orig['trnrss']) < 0:
        idxT = database_orig['trnrss'] <= -109
        idxV = database_orig['tstrss'] <= -109

        database_orig['trnrss'][idxT] = -110
        database_orig['tstrss'][idxV] = -110

        defaultNonDetectedValue = -110
        newNonDetectedValue = -110

    if np.min(database_orig['trnrss']) == -109 and np.max(database_orig['trnrss']) < 0:
        idxT = database_orig['trnrss'] <= -108
        idxV = database_orig['tstrss'] <= -108

        database_orig['trnrss'][idxT] = -109
        database_orig['tstrss'][idxV] = -109

        defaultNonDetectedValue = -109
        newNonDetectedValue = -109

    # Replace NonDetectedValue using the defined function
    if 'defaultNonDetectedValue' in locals():
        # Prepare the database structure for replacement
        db0 = {
            'trnrss': database_orig['trnrss'],
            'tstrss': database_orig['tstrss'],
            'trncrd': database_orig.get('trncrd', np.zeros((0, 3))),  # Use .get() with a default value
            'tstcrd': database_orig.get('tstcrd', np.zeros((0, 3)))   # Use .get() with a default value
        }
        database_orig = datarepNewNullDB(db0, defaultNonDetectedValue, newNonDetectedValue)

    # Handle non-detected RSSI values again if necessary
    train_df_rssi = pd.DataFrame(database_orig['trnrss'])
    test_df_rssi = pd.DataFrame(database_orig['tstrss'])
    ####################################################################

    min_rssi_value = min(train_df_rssi.min().min(), test_df_rssi.min().min())
    print(f"min_rssi_value : {min_rssi_value}")

    # Determine the shift value to make all RSSI values positive but not exceeding 100
    shift_value = 0
    if min_rssi_value < 0:
        shift_value = abs(min_rssi_value)
        if shift_value > 100:
            shift_value = 100  # Cap the shift value to 100 to avoid values exceeding 100
        train_df_rssi_positive = np.clip(train_df_rssi + shift_value, 0, 100)
        test_df_rssi_positive = np.clip(test_df_rssi + shift_value, 0, 100)
    else:
        train_df_rssi_positive = train_df_rssi
        test_df_rssi_positive = test_df_rssi
        
    # Integrate the logic for removing unnecessary APs
    database_cleaned = {
        'trncrd': database_orig['trncrd'],
        'tstcrd': database_orig['tstcrd'],
        'trnrss': train_df_rssi_positive.values,
        'tstrss': test_df_rssi_positive.values
    }

    # Determine valid APs
    valid_aps = np.sum(database_cleaned['trnrss'] != -100, axis=0) > 0
    
    # Remove unnecessary APs
    database_cleaned['trnrss'] = database_cleaned['trnrss'][:, valid_aps]
    database_cleaned['tstrss'] = database_cleaned['tstrss'][:, valid_aps]
    
    # Filter coordinate data according to valid APs
    database_cleaned['trncrd'] = database_cleaned['trncrd'][np.sum(database_cleaned['trnrss'] != -100, axis=1) > 0]
    database_cleaned['tstcrd'] = database_cleaned['tstcrd'][np.sum(database_cleaned['tstrss'] != -100, axis=1) > 0]

    # Remove void fingerprints
    # Define the indices
    vecidxTsamples = np.arange(database_cleaned['trnrss'].shape[0])
    vecidxVsamples = np.arange(database_cleaned['tstrss'].shape[0])

    # Valid training samples
    validTSamples = vecidxTsamples[np.sum(database_cleaned['trnrss'] != -100, axis=1) > 0]
    database_cleaned['trncrd'] = database_cleaned['trncrd'][validTSamples, :]
    database_cleaned['trnrss'] = database_cleaned['trnrss'][validTSamples, :]

    validVSamples = vecidxVsamples[np.sum(database_cleaned['tstrss'] != -100, axis=1) > 0]
    database_cleaned['tstcrd'] = database_cleaned['tstcrd'][validVSamples, :]
    database_cleaned['tstrss'] = database_cleaned['tstrss'][validVSamples, :]

    # Convert cleaned dataframes to NumPy arrays
    train_df_rssi_cleaned = np.array(database_cleaned['trnrss'])
    test_df_rssi_cleaned = np.array(database_cleaned['tstrss'])

    # Convert cleaned coordinates to NumPy arrays
    train_df_coord_cleaned = np.array(database_cleaned['trncrd'])
    test_df_coord_cleaned = np.array(database_cleaned['tstcrd'])

 
    all_pos_errors = []
    best_mean_error = float('inf')
    best_config = None  # Placeholder for best configuration details

    # List of distance metrics
    distance_metrics = ['cityblock', 'euclidean', 'minkowski3', 'cosine', 'sorensen', 'neyman', 'neyman2', 'lgd', 'plgd']
    k = 1
    strategy = 'unweighted'
    alpha = 0.1
   
    for distance_metric in distance_metrics:
        print(f"Running k={k}, strategy={strategy}, distance_metric={distance_metric}")
        if distance_metric == 'plgd':
            for alpha in [10, 40]:
                estimated_positions = knn_positioning(train_df_rssi_cleaned, train_df_coord_cleaned, test_df_rssi_cleaned, k, strategy, distance_metric, alpha=alpha)
                pos_errors = calculate_3d_positioning_error_org(test_df_coord_cleaned, estimated_positions)
                mean_error = np.mean(pos_errors)
                all_pos_errors.append((base_name, k, strategy, distance_metric, alpha, mean_error))
                if mean_error < best_mean_error:
                    best_mean_error = mean_error
                    best_config = (k, strategy, distance_metric, alpha)
        else:
            estimated_positions = knn_positioning(train_df_rssi_cleaned, train_df_coord_cleaned, test_df_rssi_cleaned, k, strategy, distance_metric)
            pos_errors = calculate_3d_positioning_error_org(test_df_coord_cleaned, estimated_positions)
            mean_error = np.mean(pos_errors)
            all_pos_errors.append((base_name, k, strategy, distance_metric, None, mean_error))
            if mean_error < best_mean_error:
                best_mean_error = mean_error
                best_config = (k, strategy, distance_metric, None)

    mean_errors_list.extend(all_pos_errors)
    best_configs_dict[base_name] = best_config

    # Save the position errors to a file for each dataset
    results_file_path = os.path.join(results_directory, f"{base_name}_pos_errors.csv")
    pd.DataFrame(all_pos_errors, columns=['Dataset', 'k', 'Strategy', 'DistanceMetric', 'Alpha', 'MeanError']).to_csv(results_file_path, index=False)

# Save the best configuration details for each dataset
best_configs_file_path = os.path.join(results_directory, "best_configs.csv")
pd.DataFrame.from_dict(best_configs_dict, orient='index', columns=['k', 'Strategy', 'DistanceMetric', 'Alpha']).to_csv(best_configs_file_path)

# Save the mean errors across all datasets to a CSV file
mean_errors_file_path = os.path.join(results_directory, "mean_errors.csv")
pd.DataFrame(mean_errors_list, columns=['Dataset', 'k', 'Strategy', 'DistanceMetric', 'Alpha', 'MeanError']).to_csv(mean_errors_file_path, index=False)
