### GUI Websites screenshots

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

### Fine-tuning and inference pipeline

> You can use the inference pipeline for [wav2vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) feature extraction through this link.

1. Please download the model onto your local systeam. place the CSNET, LSTM and U-vec model code into your .py file.
2. Extract the features using wave2vece facebook model found here.
3. Pass features through this model and use it.
