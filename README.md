[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/wT71nrpQ)
# Music Genre Recognition

In the lastest years there has been a notable increase in the research and investiagtion of improvement for deep learning approaches and methods. Therefore, some models/architectures seem to behave better given specific tasks/conditions.

Our aim is to evaluate music genre recognition by implementing and training deep convolutional neural networks and compare different approaches. 

Hence, after reading many papers and exploring existing work (see References), we based our approach on 
the implementation of a Convolutional Recurrent Neural Network, that combines a set of convolutional layers and Long Short Term Memory (LSTM). In this project we've based the main approach on this model, but have as well tried other similar implementations for further comparison. 

## Data

Data was obtained from https://github.com/mdeff/fma [[1]](https://github.com/mdeff/fma). This data consists on 8000 30s-long audio files and their perspective Metadata, which provides information about the songs, such as the genre, the album it belongs to or the artist.

From this metadata we are focused on the genres, which will be our target. For this task there are 8 genres, and 1000 tracks for each genre. 
<ul>
<li>Electronic
 <li>Experimental 
 <li>Folk 
 <li>Hip-Hop 
 <li>Instrumental 
 <li>International 
 <li>Pop 
 <li>Rock
</ul>

## Before running the code

### Data Download
To download the raw audio files and the metadata, run this commands on the terminal and store the downloaded files into the *data* folder.

This downloads the zip files into the data folder
```
cd data

curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
```
This code is to to verify the integrity of the files. If sha1sum doesn't work try with shasum instead.
```
echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
```
Finally, unzip the files. 
```
unzip fma_metadata.zip
unzip fma_small.zip
```
If unzip doesn't work try installing and using 7zip instead.

```
sudo apt install p7zip-full p7zip-rar

7z x fma_metadata.zip
7z x fma_small.zip
```

After this, create a ``.env`` file at repository's root with:

```
AUDIO_DIR = data/fma_small/
```

However, if you only want to run the model implementation and training code, you can directly download the .npz files containing the training, validation and testing sets in the following [Google Drive link](https://drive.google.com/drive/u/0/folders/1-PTQBiz6E53uUa9LebHjds_ZQesRHEqx).

### Environment

Before running the code you have to create a local environment with conda and activate it. The provided [environment.yml](https://github.com/DCC-UAB/dlnn-project_ia-group_8/blob/main/environment.yml) file has all the required dependencies. Run the following command to create a conda environment with all the required dependencies and then activate it:
```
conda env create --file environment.yml 

conda activate musicrecognition
```


## Code structure

``Data_Exploration.ipynb``: Understand and explore the data. 
This code is an adaptation of the ``usage`` notebook from the FMA repository [[1]](https://github.com/mdeff/fma)

``convert_to_npz.ipynb``: Convert spectrogram data into arrays.
This code is an adaptation of the ``convert_to_npz`` notebook from the Music Genre Classification repository by Priya Dwived [[4]](https://github.com/priya-dwivedi/Music_Genre_Classification)

``CRNN_Models.ipynb``: Implementation of CRNN, CNN and RNN models using spectogram data

``MLP.ipynb``: Implementation of MLP models using audio signal MFCCs extrated features.

## Code Usage and Explanation

### To use CRNN, CNN or RNN models

1. ``convert_to_npz``

Go through this notebook, which will take the small fma dataset of raw audio tracks, compute their respective spectrogram and store the data into test, train and validation arrays.

2. ``CRNN_models``

This notebook contains the implementation of four models that take the spectogram data as input and give as output the logits to be passed through a Softmax activation for multiclass classification:

- __GenreClassifier__: A CRNN that combines 1D Convolutional, Recurrent (LSTM) and linear layers. An architecture inspired by [[2]](https://github.com/ksanjeevan/crnn-audio-classification)
- __SimpleClassifier__: A simpler version of GenreClassifier.
- __SimpleClassifier_RNN__: A Recurrent Neural Network that uses LSTM followed by Linear Layers.
- __SimpleClassifier_CNN__ : A Convolutional Neural Network that uses 1D convolutions followed by Linear Layers.

There are two options for training and validating them:

- __Using Weights and Biases__: You'll need to add your API Key and change the configuration as desired: choose between the four possible models (CNN, RNN, CRNN and SimpleCRNN), set number of epochs, filters (in case of convolutional layers), learninig rate...

- __Without Weights and Biases__: Set hyperparameters and run the code.

### To use MLP models

Use only ``MLP`` as this notebook has the data obtetion and preparation completely in there. In this notebook there are two different MLP architectures, one with 4 hidden layers, batch normalisation, dropout and ReLU activations; and a simpler one with just one hidden layer and a ReLU activation. Each of them take as input the MFCCs extrated features from the metadata and give as output the logits to be passed through a Softmax Activation to perform multiclass classification.

### Results and Conclusions

After experimenting with various configurations, it became apparent that our initial model implementation, the CRNN, fell short for this particular task. Despite our efforts, the highest accuracy metric achieved was a modest 35%, reflecting limited performance. 

Confronted with these findings, we recognized the need to explore alternative architectures and configurations in pursuit of improved results. Consequently, we made the decision to embark on the development of three distinct models: a Convolutional Neural Network (CNN), a Recurrent Neural Network (RNN), and a simplified version of the previously complex Convolutional Recurrent Neural Network (CRNN).

As well as tacke the task from another perspective and use different input data and architectures, MLPs that take as input the MFCCs extrated features from the metadata. 

Despite extensive parameter tuning and modification attempts, as it can be seen in the notebooks, none of the other models' accuracies reached the desired level of performance.

Further investigation and experimentation should be performed to address the persistent underperformance and achieve the desired accuracy without overfitting or underfitting.

## Suggested Further Work

Next step was to combine the approaches used and to create a model that took both the spectrograms and metadata, and  the spectrograms went through a LSTM o CNN + LSTM and then after a resize, this was concatenated with the metadata before passing through the Linear Layers, at first with the same metadata used for the MLP, and then when this model worked try to augment the data and use categorical data such as the Artist or the Album, which should be LabelEncoded and Embedded. Unfortunately, this implementation hasn't got to be finished yet.

## References

1. [FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma)


2. [Kiran Sanjeevan Cabeza, UrbanSound classification using Convolutional Recurrent Networks in PyTorch](https://github.com/ksanjeevan/crnn-audio-classification)

3. [Priya Dwived, <i>Using CNNs and RNNs for Music Genre Recognition</i>](https://towardsdatascience.com/using-cnns-and-rnns-for-music-genre-recognition-2435fb2ed6af)

4. [Priya Dwived, <i>Music Genre Classification</i>](https://github.com/priya-dwivedi/Music_Genre_Classification)

5. [Aäron van den Oord, Sander Dieleman, Benjamin Schrauwen, Deep content-based music recommendation](https://papers.nips.cc/paper_files/paper/2013/file/b3ba8f1bee1238a2f37603d90b58898d-Paper.pdf)


## Contributors
Marta Llopart - marta.llopart@autonoma.cat

Laia Vilardell - laia.vilardell@autonoma.cat

Pol Vierge - pol.vierge@autonoma.cat

Neural Networks and Deep Learning, Artificial Intelligence Degree, UAB, 2023