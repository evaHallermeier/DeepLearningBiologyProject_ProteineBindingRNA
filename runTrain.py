import subprocess
import os

def compute_mean_from_file(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        values = [float(line.strip()) for line in lines]
        mean = sum(values) / len(values)
        print("mean correlation for all protein training for actual model is")
        print(mean)


if __name__ == '__main__':
    if os.path.exists("results.txt"):
        os.remove("results.txt")
    file_list = [
        "RBP1.txt RNAcompete_sequences.txt RBP1_input.seq RBP1_5nM.seq RBP1_20nM.seq RBP1_80nM.seq RBP1_320nM.seq RBP1_1300nM.seq",
        "RBP2.txt RNAcompete_sequences.txt RBP2_input.seq RBP2_5nM.seq RBP2_20nM.seq RBP2_80nM.seq RBP2_320nM.seq RBP2_1300nM.seq",
        "RBP3.txt RNAcompete_sequences.txt RBP3_input.seq RBP3_5nM.seq RBP3_20nM.seq RBP3_80nM.seq RBP3_320nM.seq RBP3_1300nM.seq",
        "RBP4.txt RNAcompete_sequences.txt RBP4_input.seq RBP4_5nM.seq RBP4_20nM.seq RBP4_80nM.seq RBP4_320nM.seq RBP4_1300nM.seq",
        "RBP5.txt RNAcompete_sequences.txt RBP5_input.seq RBP5_5nM.seq RBP5_20nM.seq RBP5_80nM.seq RBP5_320nM.seq RBP5_1300nM.seq",
        "RBP6.txt RNAcompete_sequences.txt RBP6_input.seq RBP6_5nM.seq RBP6_20nM.seq RBP6_80nM.seq RBP6_320nM.seq RBP6_1300nM.seq",
        "RBP7.txt RNAcompete_sequences.txt RBP7_input.seq RBP7_5nM.seq RBP7_20nM.seq RBP7_80nM.seq RBP7_320nM.seq RBP7_1300nM.seq",
        "RBP8.txt RNAcompete_sequences.txt RBP8_input.seq RBP8_5nM.seq RBP8_20nM.seq RBP8_80nM.seq RBP8_320nM.seq RBP8_1300nM.seq",
        "RBP9.txt RNAcompete_sequences.txt RBP9_input.seq RBP9_5nM.seq RBP9_20nM.seq RBP9_80nM.seq RBP9_320nM.seq RBP9_1300nM.seq",
        "RBP10.txt RNAcompete_sequences.txt RBP10_input.seq RBP10_5nM.seq RBP10_20nM.seq RBP10_80nM.seq RBP10_320nM.seq RBP10_1300nM.seq",
        "RBP11.txt RNAcompete_sequences.txt RBP11_input.seq RBP11_1nM.seq RBP11_4nM.seq RBP11_13nM.seq RBP11_40nM.seq RBP11_121nM.seq RBP11_365nM.seq RBP11_1090nM.seq RBP11_3280nM.seq RBP11_9800nM.seq",
        "RBP12.txt RNAcompete_sequences.txt RBP12_input.seq RBP12_5nM.seq RBP12_20nM.seq RBP12_80nM.seq RBP12_320nM.seq RBP12_1300nM.seq",
        "RBP13.txt RNAcompete_sequences.txt RBP13_input.seq RBP13_5nM.seq RBP13_20nM.seq RBP13_80nM.seq RBP13_320nM.seq RBP13_1300nM.seq",
        "RBP14.txt RNAcompete_sequences.txt RBP14_input.seq RBP14_5nM.seq RBP14_20nM.seq RBP14_80nM.seq RBP14_320nM.seq RBP14_1300nM.seq",
        "RBP15.txt RNAcompete_sequences.txt RBP15_input.seq RBP15_5nM.seq RBP15_20nM.seq RBP15_80nM.seq RBP15_320nM.seq RBP15_1300nM.seq",
        "RBP16.txt RNAcompete_sequences.txt RBP16_input.seq RBP16_5nM.seq RBP16_20nM.seq RBP16_80nM.seq RBP16_320nM.seq RBP16_1300nM.seq",
    ]

    j = 0

    for i in range(1, 17):
      #  subprocess.run(["python", "main.py", ] + file_list[j].split())
        subprocess.run(["python", "main.py", ] + file_list[j].split())
        print("Predict protein " + str(i) + " done")
        j += 1
    compute_mean_from_file("results.txt")

