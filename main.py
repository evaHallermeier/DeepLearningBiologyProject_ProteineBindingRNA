import re
import numpy as np
from keras import layers, initializers, Sequential
from keras.optimizers import Adam
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import time
import multiprocessing
import zipfile
import os
import sys

# Constants
IS_TRAIN = False  # Set to False for test submission
TAKE_LOWEST = False

# Hyperparameters
epochs = 5
batch_size = 2048
learning_rate = 0.001
decay_rate = 1e-9
desired_lines = 55000
max_size = 41
opt = Adam(learning_rate=learning_rate, decay=decay_rate)

num_processes = multiprocessing.cpu_count()

def get_model():
    model = Sequential([
        layers.Conv1D(filters=512, kernel_size=8, activation='relu', input_shape=(max_size, 4),
                      kernel_initializer=initializers.GlorotUniform()),
        layers.Conv1D(filters=256, kernel_size=7, activation='relu',
                      kernel_initializer=initializers.GlorotUniform()),
        layers.Conv1D(filters=128, kernel_size=6, activation='relu',
                      kernel_initializer=initializers.GlorotUniform()),
        layers.MaxPooling1D(pool_size=8, strides=8),
        layers.MaxPooling1D(pool_size=2, strides=2),
        layers.Flatten(),
        layers.Dense(16, activation='relu', kernel_initializer=initializers.GlorotUniform()),
        layers.Dense(32, activation='relu', kernel_initializer=initializers.GlorotUniform()),
        layers.Dense(1, activation='sigmoid', kernel_initializer=initializers.GlorotUniform())
    ])
    return model


def compile_model(loss):
    model = get_model()
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model


def train(model):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15, shuffle=True,
                        workers=num_processes)
    return history, model


def extract_concentration(filename):
    match = re.search(r'(\d+)nM', filename)
    return int(match.group(1)) if match else -1


def select_input_and_highest_concentration(rbp_list):
    concentration_list = [(extract_concentration(filename), filename) for filename in rbp_list]
    concentration_list.sort(key=lambda x: x[0])

    if TAKE_LOWEST:
        concentration = concentration_list[1][1]
    else:
        concentration = concentration_list[-1][1]

    return concentration_list[0][1], concentration


def get_files_class(args):
    if len(args) < 7:
        print("arguments are missing")
        exit(1)

    rbns_filenames = args[3:]
    return select_input_and_highest_concentration(rbns_filenames)



def one_hot_sequence(sequences):
    sequences_encoded = []
    for seq in sequences:
        sequences_encoded.append(one_hot(seq))
    return np.array(sequences_encoded)

def build_df(class_file):
    sequences = open_file(class_file)
    return one_hot_sequence(sequences)

def open_file(filename):
    zip_filename = "RBNS_training.zip" if IS_TRAIN else "RBNS_testing.zip"

    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return read_file(file)
    else:
        with zipfile.ZipFile(zip_filename, 'r') as zf:
            if filename in zf.namelist():
                with zf.open(filename) as file:
                    return read_file(file)

            else:
                print("File", filename, "does not exist")
                exit(0)


def read_file(file):
    sequences = []

    for _, line in zip(range(desired_lines), file):
        sequence = line.decode('utf-8').split('\t')[0]
        sequences.append(sequence)

    return sequences


def one_hot(seq):
    mapping = {"A": [1., 0., 0., 0.], "C": [0., 1., 0., 0.],
               "G": [0., 0., 1., 0.], 'N': [0.25] * 4,
               "T": [0., 0., 0., 1.], "U": [0., 0., 0., 1.],
               }

    seq2 = [mapping.get(i, [0.25, 0.25, 0.25, 0.25]) for i in seq]
    seq2.extend([[0.25, 0.25, 0.25, 0.25]] * (max_size - len(seq)))

    return np.array(seq2)

def test(model, out, true_values):
    pred = model.predict(x_test).flatten()

    if not IS_TRAIN:
        with open(out, "w") as file:
            file.writelines(f"{value:.6f}\n" for value in pred)
        return
    else:
        mean = np.mean(pred)
        std = np.std(pred)
        min_value, max_value = np.min(pred), np.max(pred)
        first_predictions = pred[:10]

        print("Mean of pred:", mean)
        print("Standard Deviation of pred:", std)
        print("Range of pred:", min_value, '-', max_value)
        print("First predictions:", first_predictions)

        # Convert pred and true_values to numpy arrays of float32
        pred = np.array(pred, dtype=np.float32)
        true_values = np.array(true_values, dtype=np.float32)

        correlation_coefficient, p_value = pearsonr(pred, true_values)
        print("Correlation coefficient:", correlation_coefficient)
        print("P-value:", p_value)

        f = "results.txt"
        create_and_write_file(f, correlation_coefficient)
        return correlation_coefficient


def preprocess_binary_data(negative_class, positive_class):
    # Preprocess and encode data
    neg_data_encoded = build_df(negative_class)
    pos_data_encoded = build_df(positive_class)

    # Create labels
    class_0_labels = np.zeros((neg_data_encoded.shape[0], 1), dtype=int)
    class_1_labels = np.ones((pos_data_encoded.shape[0], 1), dtype=int)

    # Combine datasets and labels
    combined_data = np.vstack((neg_data_encoded, pos_data_encoded))
    combined_labels = np.vstack((class_0_labels, class_1_labels))

    # Shuffle
    combined_data, combined_labels = shuffle(combined_data, combined_labels, random_state=42)

    x_train = combined_data
    y_train = combined_labels

    return x_train, y_train


def build_test_data(file_path):
    sequences = []

    # Read sequences from the file and store them in the 'sequences' list
    with open(file_path, 'r') as file:
        sequences = [line.strip() for line in file]

    # One-hot encode sequences
    sequences_encoded = [one_hot(s) for s in sequences]

    x_test = np.array(sequences_encoded)
    return x_test


def read_scores_to_predict(scores_file):
    true_values = []
    with open(scores_file, 'r') as file:
        for line in file:
            true_values.append(line.strip())
    return true_values


def create_and_write_file(filename, value):
    with open(filename, 'a') as file:
        file.write(str(value) + '\n')


if __name__ == '__main__':
    start_time = time.time()

    # Example of command-line arguments: "output.txt RNAcompete_sequences.txt RBP1.txt RBP2.txt ..."
    output_file = sys.argv[1]
    rna_cmpt_file = sys.argv[2]
    file_neg_class, file_pos_class = get_files_class(sys.argv)

    # Read true values for training if applicable
    true_values = read_scores_to_predict(output_file) if IS_TRAIN else []

    # Preprocess data
    x_train, y_train = preprocess_binary_data(file_neg_class, file_pos_class)
    x_test = build_test_data(rna_cmpt_file)

    # Compile and train the model
    model = compile_model("binary_crossentropy")
    history, model = train(model)

    # Test the model and write results
    test(model, output_file, true_values)

    end_time = time.time()

    runtime = end_time - start_time
    print(f"Runtime of the code: {runtime:.6f} seconds")
