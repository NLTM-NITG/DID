<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NLTM-DID - NIT GOA</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      margin: 0;
      padding: 0;
    }
    header {
      background-color: #333;
      color: #fff;
      text-align: center;
      padding: 20px 0;
    }
    h1, h2, h3 {
      margin-top: 30px;
    }
    h1 {
      color: #333;
    }
    h2 {
      color: #666;
    }
    p {
      text-align: justify;
    }
    img {
      display: block;
      margin: 0 auto;
    }
    footer {
      background-color: #333;
      color: #fff;
      text-align: center;
      padding: 20px 0;
      position: fixed;
      bottom: 0;
      width: 100%;
    }
  </style>
</head>
<body>

<header>
  <h1>NLTM-DID - NIT GOA</h1>
  <h3>Dialect Identification for Indian Languages</h3>
</header>

<section>
  <h2>About the Project</h2>
  <img src="static/assets/img/bg-masthead.jpeg" alt="Screenshot">
  <p>
    The focus of the work at <a href="https://www.nitgoa.ac.in/">National Institute of Technology Goa</a> is on Spoken Language Identification with a focus on the development of Dialect Identification System. The task of spoken language identification (LID) involves automatically identifying the language in which a given speech utterance was spoken. An important aspect of a spoken language is its dialects. Dialects of a given language are differences in speaking style of the first language or native language (L1) because of geographical and ethnic differences.
  </p>

  <h2>Development of Dialect Identification System</h2>
  <p>
    Dialect of a speech utterance acts as a virtual geo-tag for the utterance that helps in predicting the geographical location to which a speaker belongs. The dialectal variations of a spoken Indian language are a matter of concern for any automatic processing of speech utterances from that language. The project aims at addressing the issues of identifying the dialect in the conversational speech of the Indian languages. Dialect identification is difficult when compared to LID due to high interclass similarity among the dialects of a language. We also propose to explore deep learning methods for the dialect identification task.
  </p>

  <h2>Model Information</h2>
  <h3>Model weights</h3>
  <p>Model weights are saved as a pth file and present at <a href="Model_Marathi.pth">Marathi Model</a>.</p>

  <h3>Model Architecture</h3>
  <p>
    We use <a href="https://huggingface.co/docs/transformers/en/model_doc/wav2vec2">wav2vec2</a> for feature extraction. And use u-vector model as defined in <a href="https://dblp.org/db/conf/specom/specom2023-2.html">Sean Monteiro, Ananya Angra, Muralikrishna H, Veena Thenkanidiyoor, and A. D. Dileep, Exploring the Impact of Different Approaches for Spoken Dialect Identification of Konkani Language, in Proceedings of 25th International Conference on Speech and Computer (SPECOM-2023), November-December 2023</a>.
  </p>

  <h3>Benchmark details</h3>
  <p>We use a variety of metrics as the benchmark for the validity of the model. Please see this document <a href="#">Benchmark</a> for more details.</p>
</section>

<footer>
  <p><a href="#top">Back to Top ⬆️</a></p>
</footer>

</body>
</html>
