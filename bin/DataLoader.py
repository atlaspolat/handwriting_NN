import pickletools
import pickle
import gzip
import numpy as np

def see_the_human_content_of_pickle_file(file_path, max_bytes=1024):
    if file_path.endswith(".pkl"):
        with open(file_path, 'rb') as f:
            partial_content = f.read(max_bytes)
            try:
                    pickletools.dis(partial_content)
            except pickle.UnpicklingError:
                    print("The file is not a pickle file and cannot be opened")
            except Exception as e:
                    print(f"An error occurred: {e}")

    elif file_path.endswith(".gz"):
            with gzip.open(file_path, 'rb') as f:
                partial_content = f.read(max_bytes)
                try:
                    pickletools.dis(partial_content)
                except pickle.UnpicklingError:
                    print("The file is not a pickle file and cannot be opened")
                except Exception as e:
                    print(f"An error occurred: {e}")
                     
    else:
         print("The file is not a pickle file and cannot be opened")
                

def load_data(file_path):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data."""
     
    if file_path.endswith(".gz"):
        with gzip.open(file_path, 'rb') as f:
            training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
            return (training_data, validation_data, test_data)
    elif file_path.endswith(".pkl"):
        with open(file_path, 'rb') as f:
            training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
            return (training_data, validation_data, test_data)
    else:
        raise ValueError("Unsupported file format. Please provide a .gz or .pkl file.")
        
def load_data_wrapper(file_path):
    tr_d , va_d, te_d = load_data(file_path)
    training_inputs = [np.reshape(train_example, (784, 1)) for train_example in tr_d[0]]
    training_results = [vectorize_result(train_result) for train_result in tr_d[1]]
    vectorized_training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(val_example, (784, 1)) for val_example in va_d[0]]
    vectorized_validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(test_example, (784, 1)) for test_example in te_d[0]]
    vectorized_test_data = list(zip(test_inputs, te_d[1]))
    return (vectorized_training_data, vectorized_validation_data, vectorized_test_data)

    

def vectorize_result(result: int):
    """Return a 10-dimensional unit vector with a 1.0 in the jth position and zeroes elsewhere.
    This is used to convert a digit (0...9) into a corresponding desired output from the neural network."""
    result_vector = np.zeros((10, 1))
    result_vector[result] = 1.0
    return result_vector