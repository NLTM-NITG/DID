<div id="top"></div>
<div align="center">
<h3 align="center"> NLTM-DID - NIT GOA</h3>
  <p align="center">
    Dialect Identification for Indian Languages<br />
    <br /><br />
    <strong>
    <a href="#">View Deployment</a>
    </strong>
  </p>
</div>

<!-- ABOUT THE PROJECT -->

## 📝About The Project

![Screenshot]()
<br /> <br />
<p style="text-align: justify;">
The focus of the work at <a href="https://www.nitgoa.ac.in/">National Institute of Technology Goa</a> is on Spoken Language Identification with a focus on development of Dialect Identification System. The task of spoken language identification (LID) involves automatically identifying the language in which a given speech utterance was spoken. An important aspect of a spoken language is its dialects. Dialects of a given language are differences in speaking style of the first language or native language (L1) because of geographical and ethnic differences.
</p>

<h2>Features</h2>

<p style="text-align: justify;">
India exhibits unity in diversity not only in culture and religion but also in languages spoken by people. Language plays a vital role in communication among people as well as in accessing information and building an inclusive society. India is home to 22 constitutionally recognized languages. However, there exist more than 1000 spoken languages in India. In this information era, it is important to ensure that a spoken language gets due representation in digital space.
</p>

<h2>Development of Dialect Identification System</h2>

<p style="text-align: justify;">
Dialect of a speech utterance acts as a virtual geo-tag for the utterance that helps in predicting the geographical location to which a speaker belongs. The dialectal variations of a spoken Indian language are a matter of concern for any automatic processing of speech utterances from that language. The project aims at addressing the issues of identifying the dialect in the conversational speech of the Indian languages. Dialect identification is difficult when compared to LID due to high interclass similarity among the dialects of a language. Very little work is done on dialect identification in Indian languages except Hindi. We also propose to explore deep learning methods for the dialect identification task.
</p>

<p align="right">(<a href="#top">⬆️</a>)</p>

### 🛠Built With

-   🌐 &nbsp; Frontend </br>
    ![HTML5](https://img.shields.io/badge/-HTML5-333333?style=flat&logo=HTML5)
    ![CSS](https://img.shields.io/badge/-CSS-333333?style=flat&logo=CSS3&logoColor=1572B6)
    ![JAVASCRIPT](https://img.shields.io/badge/-JS-333333?style=flat&logo=javascript)
-   🧾&nbsp; Backend </br>
    ![Python 3](https://img.shields.io/badge/-Python-333333?style=flat&logo=Python)
    ![PyTorch](https://img.shields.io/badge/-PyTorch-333333?style=flat&logo=pytorch)
    ![Flask](https://img.shields.io/badge/-Flask-333333?style=flat&logo=flask)
    ![NumPy](https://img.shields.io/badge/-NumPy-333333?style=flat&logo=numpy)
    ![Pandas](https://img.shields.io/badge/-Pandas-333333?style=flat&logo=pandas)


<p align="right">(<a href="#top">⬆️</a>)</p>

<!-- GETTING STARTED -->

## Getting Started

You can just visit the [NLTM_NIT_GOA](#) page for using this app.

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
5. To host the website, there are three steps: First `Install all the dependencies in Step 3`, secondly `Install all the dependencies as per requirements.txt` and finally `Run the server`.<br/>
  
6. Run the following command to start the backend
    ```sh
    # To go into the Backend Folder
    cd Backend/
    # Run the server
    python app.py
    ```

7. The site is hosted now on specified port.
    
<p align="right">(<a href="#top">⬆️</a>)</p>

<!-- CONTACT -->

## Contact

LinkedIn: [EvanderDS](https://www.linkedin.com/in/evanderds/)
<br>
Project Link: [NLTM-NITGOA](#)

<p align="right">(<a href="#top">⬆️</a>)</p>

## Note: Please follow the corresponding Readme.md file for more details.

# Limitation
These models are trained are trained on a number of Indian Languages for a variety of dialects. Therefore, these models may fail in the following conditions.

- Presence of high domain mismatch.
- Contains too much noise and unclear speech.
- Does not belong to the dialects of Puneri or Marawadi in Marathi Language.

# Acknowledgement

This work is performed with the support of the project named "Speech Technologies In Indian Languages". It is part of the NLTM (National Language Technology Mission) consortium project which is sponsored by Meity (Ministry of Electronics and Information Technology), India.


# License

This project is licensed under the NLTM License - see the [LICENSE](LICENSE) file for details.
<p align="right">(<a href="#top">⬆️</a>)</p>
