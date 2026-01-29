import os
import numpy as np
import pandas as pd

# Parameters
databases = [
    'DSI1', 'DSI2', 'LIB1', 'LIB2', 'MAN1', 'MAN2', 'MINT1', 'SAH1', 'TIE1', 'TUT1', 'TUT2', 
    'TUT3', 'TUT4', 'TUT5', 'TUT6', 'TUT7', 'UJI1', 'UJI2', 'UTS1', 'OFIN1', 'OFINB1', 'OFINB2', 
    'OFINB3', 'OFINB4', 'UEXB1', 'UEXB2', 'UEXB3', 'UJIB1', 'UJIB2', 'GPR00', 'GPR01', 'GPR02', 
    'GPR03', 'GPR04', 'GPR05', 'GPR06', 'GPR07', 'GPR08', 'GPR09', 'GPR10', 'GPR11', 'GPR12', 
    'GPR13', 'SOD01', 'SOD02', 'SOD03', 'SOD04', 'SOD05', 'SOD06', 'SOD07', 'SOD08', 'SOD09', 
    'KIOS0', 'KIOS1', 'KIOS2', 'KIOS3', 'KIOS4', 'KIOS5', 'EEIL01', 'EEIL02', 'HDB11', 'HDB12', 
    'HDB13', 'HDB21', 'HDB22', 'HDB23', 'HDB31', 'HDB32', 'HDB33'
]
fpmethods = ['knn_plain2022', 'wknnid_plain2022', 'wknnsid_plain2022']
datareprs = ['positive']
distances = [
    'distancem_cityblock', 'distancem_euclidean', 'distancem_minkowsky3', 'distancem_sorensen',
    'distancem_cosine', 'distancem_LGD', 'distancem_PLGD10', 'distancem_PLGD40', 
    'distancem_neyman', 'distancem_neyman2'
]
kvalues = list(range(1, 22)) + list(range(23, 52, 2))

# Initialize storage for errors and labels
num_combinations = len(fpmethods) * len(datareprs) * len(distances) * len(kvalues)
error_data = np.zeros((num_combinations, len(databases)))
label_data = np.empty((num_combinations, len(databases)), dtype=object)

idx = 0

# Loop through each database
for db_idx, database in enumerate(databases):
    print(f"Processing database: {database}")

    # Loop through fingerprinting methods, data representations, and distances
    for fpmethod in fpmethods:
        for datarepr in datareprs:
            for distance in distances:
                for k in kvalues:
                    
                    # Construct file path and read the errors CSV
                    filename = os.path.join('Results_pos_err', database, fpmethod, 
                                            f"{datarepr}_{distance}_k{str(k).zfill(3)}", 
                                            'errors_rep001.csv')
                    
                    try:
                        errors = pd.read_csv(filename, header=None).to_numpy()
                        mean_error = np.mean(errors)

                        # Save results to error_data and label_data
                        error_data[idx, db_idx] = mean_error
                        label_data[idx, db_idx] = filename

                    except FileNotFoundError:
                        print(f"File not found: {filename}")
                        error_data[idx, db_idx] = np.nan
                        label_data[idx, db_idx] = None

                    idx += 1

# Save results to a .npz file
np.savez('getBestParams_experiment1.npz', error=error_data, label=label_data)
