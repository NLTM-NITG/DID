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

### Website 

>[!Note]
> Requires server for backend services

<strong>[View Deployment](https://nltm-nitg.github.io/Dialect-Identification/)</strong>

### Demo screenshots

> Landing Page
  
<p align="center">
  <img src="Demo Pictures/Landing Page.png" alt="Screenshot">
</p>

> Predicting Dialect
  
 <p align="center">
  <img src="Demo Pictures/Main Demo Page.png" alt="Screenshot">
</p>

> Before predicting Dialect

 <p align="center">
  <img src="Demo Pictures/Main demo page before prediction.png" alt="Screenshot">
</p>
  
> After predicting Dialect
  
 <p align="center">
  <img src="Demo Pictures/After prediction.png" alt="Screenshot">
</p>
  
> Publications
  
 <p align="center">
  <img src="Demo Pictures/Publications Page.png" alt="Screenshot">
</p>

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


## Tutorials

<!-- GETTING STARTED -->

### Getting Started

You can just visit [NIT_GOA](#) page for using this app.

### Prerequisites

There is no such complicated prerequisites for using this app except for `using modern browsers`. But if you are using it locally you will need `python3` and `nodejs>=16.0` and you will have to install some packages.

### ⚙Installation

If you want to get a local copy of this app.

1. Clone the repo
    ```sh
    git clone https://github.com/NLTM-NITG/DID.git
    ```
2. Navigate to the folder `NLTM_NIT_GOA`
    ```sh
    cd Dialect_Identification
    ```
3. To host the Backend, we need to install some packages via `pip`. Hence, run the following command. (Note: use `Python 3` only)

    ```sh
    # Installing backend dependencies
    pip install numpy pandas torch librosa ipython sklearn flask flask_cors
    ```
5. To host the website, there are three steps:
- **Step 1:** Begin by installing all necessary dependencies outlined in Step 3.
- **Step 2:** Proceed to install the required dependencies according to the specifications outlined in `requirements.txt`.
- **Step 3:** Initiate the server by executing the relevant command.

6. Download the wave2vec2-ASR Large Model.
  
7. Run the following command to start the website with backend capabilities
    ```sh
    # Run the server
    python app.py
    ```

8. The site is hosted now on specified port as per Flask cnofig.
    
### Limitation
These models are trained are trained on a number of Indian Languages for a variety of dialects. Therefore, these models may fail in the following conditions.

- Presence of unfamiliar dialects and accents.
- Presence of high domain mismatch.
- Contains too much noise and unclear speech.
- Does not belong to the dialects of Puneri or Marawadi in Marathi Language.
- Note: Please follow the corresponding Readme.md file for more details.

### Fine tuning and inference pipeline

> You can use the inference pipeline for [wav2vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) through this link.

1. Please download the model onto your local systeam. place the CSNET, LSTM and U-vec model code into your .py file.
2. Extract the features using wave2vece facebook model found here.
3. Pass feaatures through this model and use it.

For finetuning
>[!WARNING]
>Yet to add the script

<!-- CONTACT -->

## License

This project is licensed under the [NLTM Creative Commons CC-BY-4 LICENSE](LICENSE) file for details.

## Contact

- [Dr.Veena T.](https://www.nitgoa.ac.in/)
- [EvanderDS](https://www.linkedin.com/in/evanderds/)
 
## Acknowledgement

This work is performed with the support of the project named "Speech Technologies In Indian Languages". It is part of the NLTM (National Language Technology Mission) consortium project which is sponsored by Meity (Ministry of Electronics and Information Technology), India.

<p align="right">(<a href="#top">⬆️</a>)</p>


