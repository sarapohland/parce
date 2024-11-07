import os
import pickle

# Create dictionary of image properties and factors
factors = {'lunar':
                {
                    'brightness': [0.5, 0.75, 1.25, 1.5, 2.0, 3.0], 
                    'contrast': [0.25, 0.5, 1.5, 2.0, 3.0, 4.0], 
                    'saturation': [0.0, 5.0, 7.0, 10.0, 15.0, 20.0], 
                    'noise': [20, 40, 60, 80, 100, 120], 
                    'pixelate': [10, 20, 24, 30, 36, 40], 
                },
            'speed':
                {
                    'brightness': [], 
                    'contrast': [], 
                    'saturation': [], 
                    'noise': [], 
                    'pixelate': [], 
                },
            'pavilion':
                {
                    'brightness': [0.0, 0.5, 1.5, 2.0, 2.5, 3.0], 
                    'contrast': [0.0, 0.5, 0.75, 1.5, 1.75, 2.0], 
                    'saturation': [0.0, 2.0, 4.0, 6.0, 8.0, 10.0], 
                    'noise': [20, 40, 60, 80, 100, 120], 
                    'pixelate': [10, 20, 24, 30, 36, 40], 
                }
    }

def load_results_by_property(folder, values=None):
    # Iterate through all files in the directory
    factors, datas = [], []
    for file_name in os.listdir(folder):
        if file_name.endswith('.p'):
            # Extract the factor from the filename
            factor_str = file_name.split('.p')[0]
            factor = float(factor_str)
            if values is not None:
                if not factor in values:
                    continue
            factors.append(factor)
            
            # Load the saved data
            file_path = os.path.join(folder, file_name)
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
            datas.append(data)
    return factors, datas
