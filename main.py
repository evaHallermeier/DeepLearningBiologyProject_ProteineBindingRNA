import re
import sys
import time
import numpy as np
from keras import layers, initializers, Sequential
from keras.optimizers import Adam
from scipy.stats import pearsonr
from sklearn.utils import shuffle
import os
import zipfile

IS_TRAIN = True # if False : submit result of test

# hyperparameters
epoch = 5
b = 2048
lr = 0.001
opt = Adam(learning_rate=lr, decay=0.00001)
DESIRED_LINES = 50000

MAX_SIZE = 41

def get_model():
    model = Sequential([
        layers.Conv1D(filters=128, kernel_size=6, activation='relu', input_shape=(MAX_SIZE, 4),
                      kernel_initializer=initializers.GlorotUniform()),
        layers.MaxPooling1D(pool_size=6, strides=1),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_initializer=initializers.GlorotUniform()),
        layers.Dense(32, activation='relu', kernel_initializer=initializers.GlorotUniform()),
        layers.Dense(32, activation='relu', kernel_initializer=initializers.GlorotUniform()),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def compile_model(loss, optimizer):
    model = get_model()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

def train(model, epochs, batch_size):
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.15, shuffle=True)
    return history, model

def select_input_and_lowest_concentration(rbp_list):
    concentration_list = []
    for filename in rbp_list:
        match = re.search(r'(\d+)nM', filename)
        if match and "input" not in filename:
            concentration = int(match.group(1))
            concentration_list.append((concentration, filename))
        else:
            concentration_list.append((-1, filename))
    concentration_list.sort(key=lambda l: l[0])
    return concentration_list[0][1], concentration_list[1][1]

def select_input_and_highest_concentration(rbp_list):
    concentration_list = []
    for filename in rbp_list:
        match = re.search(r'(\d+)nM', filename)
        if match and "input" not in filename:
            concentration = int(match.group(1))
            concentration_list.append((concentration, filename))
        else:
            concentration_list.append((-1, filename))
    concentration_list.sort(key=lambda l: l[0])
    return concentration_list[0][1], concentration_list[-1][1]

def get_files_class(args):
    if len(args) < 7:
        print("arguments are missing")
        exit(1)
    rbns_filenames = args[3:]
    neg_file, pos_file = select_input_and_lowest_concentration(rbns_filenames) #select_input_and_highest_concentration(rbns_filenames)
    return neg_file, pos_file

def one_hot_sequence(sequences):
    sequences_encoded = []
    for seq in sequences:
        sequences_encoded.append(one_hot(seq))
    return np.array(sequences_encoded)

def build_df(class_file, desired_lines=None):
    sequences = open_file(class_file, desired_lines)
    return one_hot_sequence(sequences)

def open_file(filename):
    if IS_TRAIN:
        zip_filename = "RBNS_training.zip"
    else:
        zip_filename = "RBNS_testing.zip"

    with zipfile.ZipFile(zip_filename, 'r') as zf:
        if filename in zf.namelist():
            with zf.open(filename) as file:
              sequences = read_file(file)
            return sequences
        else:
            if os.path.exists(filename):
                with open(filename, 'r') as file:
                    sequences = read_file(file)
            else:
                print("file ", filename ,"do est n exists")
                exit(0)

    return sequences

def read_file(file):
    lines_read = 0
    sequences = []
    for line in file:
        sequence = line.strip().split('\t')[0]  # Extract RNA sequence before the tab
        sequences.append(sequence)
        lines_read += 1
        if DESIRED_LINES is not None and lines_read >= DESIRED_LINES:
            break
    return sequences

def one_hot(seq):
    length_seq = len(seq)
    seq2 = list()
    mapping = {"A": [1., 0., 0., 0.], "C": [0., 1., 0., 0.],
               "G": [0., 0., 1., 0.], 'N': np.array([0.25] * 4),
               "T": [0., 0., 0., 1.], "U": [0., 0., 0., 1.],
               }
    for i in seq:
        seq2.append(mapping[i] if i in mapping.keys() else [0., 0., 0., 0.])
    if length_seq < MAX_SIZE:
        j = MAX_SIZE - length_seq
        for k in range(j):
            seq2.append([0.25, 0.25, 0.25, 0.25])
    return np.array(seq2)

def test(model, out):
    pred = model.predict(x_test)
    pred = [item[0] for item in pred]

    # write resul in output file
    if not IS_TRAIN:
        with open(out, "w") as file:
            for value in pred:
                file.write(str(value) + "\n")
        return
    else: # TRAIN CASE
        mean = np.mean(pred)
        std = np.std(pred)

        print("Mean of pred:", mean)
        print("Standard Deviation of pred:", std)
        print("range of pred ", np.min(pred), '-', np.max(pred))
        print("first prediction: ", pred[0:10])

        correlation_coefficient, p_value = pearsonr(pred, true_values)
        print("correlation is ", correlation_coefficient)
        print("p_value is ", p_value)

        f = "results.txt"
        create_and_write_file(f,correlation_coefficient)
        return correlation_coefficient

def resume(epochs, learning_rate, batch_size, corr):
    print("\n\n")
    print("EXPERIENCE:")
    print("num epochs: ", epochs)
    print("learning rate: ", learning_rate)
    print("batch size ", batch_size)
    print("correlation is ", corr)

def preprocess_binary_data(negative_class, positive_class):
    #   data
    neg_data_encoded = build_df(negative_class, DESIRED_LINES) # and encode
    pos_data_encoded = build_df(positive_class, DESIRED_LINES) # and encode

    #labels
    class_0_labels = np.zeros((neg_data_encoded.shape[0], 1), dtype=int)
    class_1_labels = np.ones((pos_data_encoded.shape[0], 1), dtype=int)

    #combine 2 dataset
    combined_data = np.concatenate((neg_data_encoded, pos_data_encoded), axis=0)

    #combine labels
    combined_labels = np.concatenate((class_0_labels, class_1_labels), axis=0)

    #shuffle
    combined_data, combined_labels = shuffle(combined_data, combined_labels, random_state=42)

    x_train = combined_data
    y_train = combined_labels

    return x_train, y_train

def buildTestData(file):
    file_path = file  # Replace with the actual file path
    sequences = []
    sequences_encoded = []

    # Read sequences from the file "RNAcompete_sequences.txt" and store them in the 'sequences' list
    with open(file_path, 'r') as file:
        for line in file:
            sequence = line.strip()
            sequences.append(sequence)

    # one hot encoding
    for s in sequences:
        one_hot_encoded_dna = one_hot(s)
        sequences_encoded.append(one_hot_encoded_dna)

    rna_cmpt_encoded = sequences_encoded

    x_test = np.array(rna_cmpt_encoded)
    return x_test

# def create_negative_seqs(lines_from_all_files):
#     """For each seq create a shuffled version"""
#     negative_seqs = []
#     for seq in lines_from_all_files:
#         seq_list = [ch for ch in seq]
#         np.random.shuffle(seq_list)
#         negative_seqs.append(''.join(seq_list))
#     return negative_seqs

def read_scores_to_predict(scores_file):
    true_values = []
    with open(scores_file, 'r') as file:
        for line in file:
            true_values.append(line.strip())

def create_and_write_file(filename, value):
    with open(filename, 'a') as file:
        file.write(str(value) + '\n')

if __name__ == '__main__':
    start_time = time.time()

    #exemple of arg:  "RBP1.txt RNAcompete_sequences.txt RBP1_input.seq RBP1_5nM.seq RBP1_20nM.seq RBP1_80nM.seq RBP1_320nM.seq RBP1_1300nM.seq",
    output_file = sys.argv[1] # values of y_test for training or name of file to write reesults if training
    rna_cmpt_file = sys.argv[2] #name of file with RNA compete sequences as input for our predictions
    # binary classification : class 0 : low binding intensity , class 1 : hight binding intensity
    file_neg_class, file_pos_class = get_files_class(sys.argv)

    # get true_values
    if IS_TRAIN:
        true_values = read_scores_to_predict(output_file)
    # true values are lis t of score of binding intensity we need to predict
        true_values = [np.float32(item) for item in true_values]

    x_train, y_train = preprocess_binary_data(file_neg_class, file_pos_class)
    x_test = buildTestData(rna_cmpt_file)

    m = compile_model("binary_crossentropy", opt)
    h, m = train(m, epoch, b)
    test(m, output_file)

    end_time = time.time()

    runtime = end_time - start_time
    print(f"Runtime of the code: {runtime:.6f} seconds")