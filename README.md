<div style="text-align: center;">
    <h1>NLTM - National Institute of Technology GOA</h1>
</div>
  
<!-- ABOUT THE PROJECT -->


## Dialect Identification for Indian Languages

<p align="center">
  <img src="static/assets/img/bg-masthead.jpeg" alt="Screenshot">
</p>


### Introduction

<p style="text-align: justify;">
The focus of the work at <a href="https://www.nitgoa.ac.in/">National Institute of Technology Goa</a> is on Spoken Language Identification with a focus on development of Dialect Identification System. The task of spoken language identification (LID) involves automatically identifying the language in which a given speech utterance was spoken. An important aspect of a spoken language is its dialects. Dialects of a given language are differences in speaking style of the first language or native language (L1) because of geographical and ethnic differences.
</p>

### Mission

<p style="text-align: justify;">
Dialect of a speech utterance acts as a virtual geo-tag for the utterance that helps in predicting the geographical location to which a speaker belongs. The dialectal variations of a spoken Indian language are a matter of concern for any automatic processing of speech utterances from that language. The project aims at addressing the issues of identifying the dialect in the conversational speech of the Indian languages. Dialect identification is difficult when compared to LID due to high interclass similarity among the dialects of a language. We also propose to explore deep learning methods for the dialect identification task.
</p>

##  Indian Language Dialect Identification Models

### Architecture

We use two models in unison for Dialect Identification. A feature extractor and then our predictor model.

Our feature Extractor is a wav2vec2 model hosted on HuggingFace and developed by Facebook [Link.](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) <br>
Predictor: U-vector model as defined in Sean Monteiro, Ananya Angra, Muralikrishna H, Veena Thenkanidiyoor, and A. D. Dileep, Exploring the Impact of Different Approaches for Spoken Dialect Identification of Konkani Language, in Proceedings of 25th International Conference on Speech and Computer (SPECOM-2023), November-December 2023 [Link.](https://dblp.org/db/conf/specom/specom2023-2.html)


### Weights

The model weights are stored as a .pth file, accessible via the link provided: [DID Model](). These weights represent the learned parameters of the language model based on the dialects of numerous Indian languages like Marathi, Konkani, Tamil, Kannada etc.

### Benchmark details

Our model's validity is assessed using a range of metrics. For more information, refer to the document linked here: [Benchmark]().

## Demo 
### Command Line Interface (CLI) 
<hr>

#### Overview
The Dialect Identification system comprises a Graphical User Interface (GUI) website and a Command Line Interface (CLI) designed for identifying the Dialect of spoken audio utilizing a u-vector model. The GUI facilitates various tasks such as audio recording, playback of saved audio files, and Dialect identification of spoken audio. On the other hand, the CLI enables users to predict the Dialect of individual audio files or directories containing multiple audio files.


#### **Setup**
Clone the Repository: Obtain the system source code by cloning the repository from GitHub:

```sh
git clone https://github.com/NLTM-NITG/Dialect-Identification
```
    
Navigate to the CLI Directory: Access the Command Line Interface directory within the cloned repository:

```sh
cd Dialect-Identification/CLI
```
<p align="center">
  <span style="background-color: white; padding: 0 10px;">(OR)</span>
</p>

Download the script present in the folder titled Demo.py.

 
#### **Running the CLI Scripts**

To predict the Dialect of audio files using the Command Line Interface, execute the following command:
   
```sh
python demo.py <path/to/audio_file_or_directory>
```
   
Replace `<path/to/audio_file_or_directory>` with the path to the audio file or directory containing audio files with the '.wav' extension.


#### **Output**

- If a directory is specified, the predicted languages for each audio file will be displayed in the console.
- A CSV file named `Predicted_Dialect.csv` will be generated in the current directory, containing the audio filename and predicted language for each audio file.

<hr>

####  **Performance Metrics Calculation**

To calculate performance metrics such as Confusion Matrix, Accuracy, True Positive Rate (Recall), False Positive Rate, False Negative Rate, Precision, F1 Score, and Equal Error Rate (EER), follow these steps:


Add Actual Language Labels: Ensure that the actual language labels of each file are included in the `Predicted_Dialect.csv` file under the column named `Dialect`.


Run the Performance Metrics Script: Execute the `Prediction_Metrics.py` script with the following command:

```sh
python Prediction_Metrics.py --CSV_File_Name <path/to/Predicted_Dialect.csv>
```

Replace `<path/to/Predicted_Dialect.csv>` with the path to the `Predicted_Dialect.csv` file containing the predicted language labels.


#### **Output**

The script will produce the calculated performance metrics, which will be displayed in the console.

<hr>

### GUI Website 

<strong>[View Deployment](https://nltm-nitg.github.io/Dialect-Identification/)</strong>

For more information : [NIT Goa - Dialect Identification Website](https://github.com/NLTM-NITG/Dialect-Identification/blob/main/GUI%20Website.md#gui-websites-screenshots)
>[!NOTE]
>#### 🛠Built With
>
>-   🌐 &nbsp; Frontend </br>
>    ![HTML5](https://img.shields.io/badge/-HTML5-333333?style=flat&logo=HTML5)
>    ![CSS](https://img.shields.io/badge/-CSS-333333?style=flat&logo=CSS3&logoColor=1572B6)
>    ![JAVASCRIPT](https://img.shields.io/badge/-JS-333333?style=flat&logo=javascript)
>-   🧾&nbsp; Backend </br>
>    ![Python 3](https://img.shields.io/badge/-Python-333333?style=flat&logo=Python)
>    ![PyTorch](https://img.shields.io/badge/-PyTorch-333333?style=flat&logo=pytorch)
>    ![Flask](https://img.shields.io/badge/-Flask-333333?style=flat&logo=flask)
>    ![NumPy](https://img.shields.io/badge/-NumPy-333333?style=flat&logo=numpy)
>    ![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)

<hr>

   
### Limitation
These models are trained are trained on a number of Indian Languages for a variety of dialects. Therefore, these models may fail in the following conditions.

- Presence of unfamiliar dialects and accents.
- Presence of high domain mismatch.
- Contains too much noise and unclear speech.
- Does not belong to the dialects of Puneri or Marawadi in Marathi Language.
- Minimum duration of audio is not around 2-4 seconds for accurate Dialect identification.


<!-- CONTACT -->
## License

This project is licensed under the [NLTM Creative Commons CC-BY-4 LICENSE](LICENSE) file for details.

## Contact

- [Dr.Veena T.](https://www.nitgoa.ac.in/)
- [EvanderDS](https://www.linkedin.com/in/evanderds/)
 
## Acknowledgement

This work is performed with the support of the project named "Speech Technologies In Indian Languages". It is part of the NLTM (National Language Technology Mission) consortium project which is sponsored by Meity (Ministry of Electronics and Information Technology), India.

<p align="right">(<a href="#top">⬆️</a>)</p>


