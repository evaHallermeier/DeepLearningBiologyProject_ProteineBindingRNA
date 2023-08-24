import subprocess
import time

if __name__ == '__main__':
    file_list = [
        "RBP17.txt RNAcompete_sequences.txt RBP17_input.seq RBP17_5nM.seq RBP17_20nM.seq RBP17_80nM.seq RBP17_320nM.seq RBP17_1300nM.seq",
        "RBP18.txt RNAcompete_sequences.txt RBP18_input.seq RBP18_5nM.seq RBP18_20nM.seq RBP18_80nM.seq RBP18_320nM.seq RBP18_1300nM.seq",
        "RBP19.txt RNAcompete_sequences.txt RBP19_input.seq RBP19_5nM.seq RBP19_20nM.seq RBP19_80nM.seq RBP19_320nM.seq RBP19_1300nM.seq",
        "RBP20.txt RNAcompete_sequences.txt RBP20_input.seq RBP20_5nM.seq RBP20_20nM.seq RBP20_80nM.seq RBP20_320nM.seq RBP20_1300nM.seq",
        "RBP21.txt RNAcompete_sequences.txt RBP21_input.seq RBP21_5nM.seq RBP21_20nM.seq RBP21_80nM.seq RBP21_320nM.seq RBP21_1300nM.seq",
        "RBP22.txt RNAcompete_sequences.txt RBP22_input.seq RBP22_5nM.seq RBP22_20nM.seq RBP22_80nM.seq RBP22_1300nM.seq",
        "RBP23.txt RNAcompete_sequences.txt RBP23_input.seq RBP23_5nM.seq RBP23_20nM.seq RBP23_80nM.seq RBP23_320nM.seq RBP23_1300nM.seq",
        "RBP24.txt RNAcompete_sequences.txt RBP24_input.seq RBP24_5nM.seq RBP24_20nM.seq RBP24_80nM.seq RBP24_320nM.seq RBP24_1300nM.seq",
        "RBP25.txt RNAcompete_sequences.txt RBP25_input.seq RBP25_5nM.seq RBP25_20nM.seq RBP25_80nM.seq RBP25_320nM.seq RBP25_1300nM.seq",
        "RBP26.txt RNAcompete_sequences.txt RBP26_input.seq RBP26_5nM.seq RBP26_20nM.seq RBP26_80nM.seq RBP26_320nM.seq RBP26_1300nM.seq",
        "RBP27.txt RNAcompete_sequences.txt RBP27_input.seq RBP27_5nM.seq RBP27_80nM.seq RBP27_320nM.seq RBP27_1300nM.seq",
        "RBP28.txt RNAcompete_sequences.txt RBP28_input.seq RBP28_5nM.seq RBP28_20nM.seq RBP28_80nM.seq RBP28_320nM.seq RBP28_1300nM.seq",
        "RBP29.txt RNAcompete_sequences.txt RBP29_input.seq RBP29_80nM.seq RBP29_320nM.seq RBP29_20nM.seq RBP29_1300nM.seq",
        "RBP30.txt RNAcompete_sequences.txt RBP30_input.seq RBP30_5nM.seq RBP30_20nM.seq RBP30_80nM.seq RBP30_320nM.seq RBP30_1300nM.seq",
        "RBP31.txt RNAcompete_sequences.txt RBP31_input.seq RBP31_2nM.seq RBP31_8nM.seq RBP31_16nM.seq RBP31_64nM.seq RBP31_256nM.seq RBP31_500pM.seq RBP31_1000nM.seq"
    ]
    start_time = time.time()
    j = 0
    for i in range(17, 32):
        subprocess.run(["python", "main.py", ] + file_list[j].split())
        print("Predict protein " + str(i) + " done")
        j += 1
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime of all proteins took: {runtime:.6f} seconds")
