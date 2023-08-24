# 🧬 RNA Binding with Deep Learning
## 🚀 Goal 
Create a binding model for each RBP and use it to predict RNA binding intensity for each RNAcompete probe and submit a list of binding intensities

##  🗄️ Data
### 🏋️ Training
set of RNA Bind-n-Seq data of different RBPs for the learning phase with RBNS data (4-6 sequence files) of one RBP and a list of RNAcompete probes (1 sequence file). We have 16 training sets (RBNS data + RNAcompete probes with binding intensities).
### 📈 Check performance
RNA compete probes for the prediction and rncmpt files with scores 
### 🧪 Testing
15 test sets (RBNS data + RNAcompete probes). You have to assign a binding intensity to each RNAcompete probe.

## 🧮 Metric for performance: 
We checked the performance of the model by measuring the pearson correlation between our prediction and the scores
(true values) given for each protein and the mean of each correlation of each protein we got.

## 🏆 Challenges
Prediction performance and training runtime

### 🤔 Data choice
Which data use (which files and which concentration)?
With the idea of a binary classification (low or high score of intensity),
we needed to choose two files for the training of each protein.
We tried a few combinations and we at the end chose to use only the input file (class 0 or negative class)
and the file with the highest concentration (class 1 or positive class)

### 🗂️  Data size
We have limited resources for computation and limited amount of time so the challenge is to find a good compromise
between reasonable amount of training time and good results.
We cannot use the entire file (more than a million samples).
We tried a few different sizes of data and we at the end chose to read the first 55000 lines of files with concentration
or input file.

### ⚙️ Preprocessing
- We decided to use only the files input and the one with the highest concentration of a given protein,
- We read only the DNA sequences (without the count) that we encode using one hot encoding.
- We encode every nucleotide as a one-hot vector of dimension 4. An additional nucleotide N represents an unknown base and is encoded by {0.25, 0.25, 0.25, 0.25}. 
- In addition, we add padding to each sequence to have a single length of 41 for all sequence. 
- We created labels and we shuffled the all data. 
- In addition, for the prediction we read all the RNA compete file, and we encoded with the same one hot encoding method  and defined it as our x_test data.

## 🛠️ Our final implementation
We built a deep neural network that was training on a binary classification problem (low or high binding intensity score). 

![alt text](https://ibb.co/nLNFn8h)

We found that using more filters provides the opportunity to find different motifs or different variants of the same motif. The purpose of the global max-pooling is to determine if a motif exists in the input sequence or not.
We found that different kernel sizes perform differently on different proteins. This can be explained by the fact that motifs can be of varying length. To overcome this, we decided to use three different kernel sizes. 

## 🎛️ Hyper parameters
- nb epochs: 5 
- batch size: 2048 
- optimizer: Adam 
- learning rate: 0.001 
- loss function: binary cross entropy

![alt text](http://url/to/img.png)


## 🧾 Results of the model

![alt text](http://url/to/img.png)


- Mean correlation of all proteins: 0.191
- Average of training time of a single protein: 78-88 seconds
- Using a M2 Pro processor: 32 GB RAM, 10 CPU


## Code explanation
### main.py
Contains training pipeline and make predictions for rna compete.
The code contains a variable called IS_TRAIN : if True, we compute correlation and if False we print predictions in file.
We can run it with the arguments asked.

For example:

`python main.py RBP23.txt RNAcompete_sequences.txt RBP23_input.seq RBP23_5nM.seq RBP23_20nM.seq RBP23_80nM.seq RBP23_320nM.seq RBP23_1300nM.seq`

### runTrain.py
Script that run the main file for each protein of training and compute mean correlation (correlation list in results.txt file).
### runTest.py
Script that run the main file for each protein of testing and permits to create scores files for all proteins of testing phase.