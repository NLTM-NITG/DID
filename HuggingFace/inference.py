from model import DID_Model
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
aud_path = r"uploads\L.wav"
wave2vec_model_path = r"model_snapshots\wav2vec2_model.pth"
model_path = r"model_snapshots\Marathi_Model_Snapshot.pth"



if __name__ == "__main__":
    # Load the Wav2Vec 2.0 model from torchaudio pipelines

    # Load custom dialect identification model
    model = DID_Model()
    model.load_weights(model_path)
    # Predict dialect
    predicted_dialect = model.predict_dialect(aud_path, wave2vec_model_path) #
    print("Predicted Dialect:", predicted_dialect)
