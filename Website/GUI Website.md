# NLTM - National Institute of Technology Goa | Dialect Identification

## Screenshots

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

<!-- GETTING STARTED -->

## Tutorials

### Getting Started

You can just visit [NIT_GOA](https://nltm-nitg.github.io/Dialect-Identification/) page for using this app.

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


### ⚙Prerequisites

1. There is no such complicated prerequisites for using this app except for `using modern browsers`. But if you are using it locally you will need `python3` and `nodejs>=16.0` and you will have to install some packages.
2. To host the Backend, we need to install some packages via `pip`. Hence, run the following command. (Note: use `Python 3` only)
   
```sh
pip install -r requirements.txt
```

### Installation

If you want to get a local copy of this app.

1. Clone the repo
    ```sh
    git clone https://github.com/NLTM-NITG/Dialect-Identification.git
    ```
2. Navigate to the folder `Website`
    ```sh
    cd Dialect_Identification/Website
    ```
3. Download the models [wave2vec2-ASR](https://github.com/NLTM-NITG/Dialect-Identification/blob/main/wav2vec2_model.pth) and [Model_Marathi](https://github.com/NLTM-NITG/Dialect-Identification/blob/main/Model_Marathi.pth) and move then to the same directory as '''app.py'''

4. Run the following command to start the website with backend capabilities
   
    ```sh
    # Run the server
    python app.py
    ```

8. The site is hosted now on specified port as per Flask config.

### Fine-tuning and Inference pipeline

## Inference
You can use the pipeline for [wav2vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) feature extraction or follow the [CLI instructions](https://github.com/NLTM-NITG/Dialect-Identification?tab=readme-ov-file#command-line-interface-cli).

## Fine tuning for other Dialects
1. Please download the website onto your local system.
2. Change the label values pertaining to the dialects to be used in app.py.
3. Retrain the model based in CSNET, LSTM and U-vec model on your training data.
4. Place this new model link into the Model path.
5. For testing locally run the app.py or CLI instructions.
   
