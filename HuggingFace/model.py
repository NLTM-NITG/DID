import torch
import torch.nn as nn
import numpy as np
import torchaudio
import soundfile as sf
from   torch import Tensor

# Define your device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define constants based on the loaded checkpoint
e_dim = 512  # Update with the correct embedding dimension based on your model
n_classes = 2  # Number of language classes, based on your requirement
look_back1 = 30
look_back2 = 60
lan2id = {'MA': 0, 'PU': 1}

# Function to preprocess input data
def Get_data(X):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()

    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std

    Xdata1 = []
    Xdata2 = []
    for i in range(0, len(X)-look_back1, 1):
        a = X[i:(i+look_back1), :]
        Xdata1.append(a)
    Xdata1 = np.array(Xdata1)

    for i in range(0, len(X)-look_back2, 2):
        b = X[i+1:(i+look_back2):3, :]
        Xdata2.append(b)
    Xdata2 = np.array(Xdata2)

    return Xdata1, Xdata2



class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(1024, 512, bidirectional=True)
        self.lstm2 = nn.LSTM(1024, 256, bidirectional=True)

        self.fc_ha = nn.Linear(e_dim, 256)
        self.fc_1 = nn.Linear(256, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x1, _ = self.lstm1(x)
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0)

        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.softmax(alp)

        T = list(ht.shape)[1]
        batch_size = list(ht.shape)[0]
        D = list(ht.shape)[2]
        c = torch.bmm(al.view(batch_size, 1, T), ht.view(batch_size, T, D))
        c = torch.squeeze(c, 0)
        return c

class CCSL_Net(nn.Module):
    def __init__(self, model1, model2):
        super(CCSL_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2

        self.att1 = nn.Linear(e_dim, 256)
        self.att2 = nn.Linear(256, 1)

        self.softmax = nn.Softmax(dim=1)
        self.lang_classifier = nn.Linear(e_dim, n_classes, bias=False)

    def forward(self, x1, x2):
        e1 = self.model1(x1)
        e2 = self.model2(x2)

        ht_e = torch.cat((e1, e2), dim=0)
        ht_e = torch.unsqueeze(ht_e, 0)
        ha_e = torch.tanh(self.att1(ht_e))
        alp = torch.tanh(self.att2(ha_e))
        al = self.softmax(alp)
        Tb = list(ht_e.shape)[1]
        batch_size = list(ht_e.shape)[0]
        D = list(ht_e.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb), ht_e.view(batch_size, Tb, D))
        u_vec = torch.squeeze(u_vec, 0)

        lan_prim = self.lang_classifier(u_vec)

        return lan_prim

class DID_Model(nn.Module):
    def __init__(self):
        super(DID_Model, self).__init__()
        self.model1 = LSTMNet()
        self.model2 = LSTMNet()
        self.ccslnet = CCSL_Net(self.model1, self.model2)

    def forward(self, x1, x2):
        output = self.ccslnet(x1, x2)
        return output

    def load_weights(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load weights for model1
        self.model1.lstm1.load_state_dict({
            'weight_ih_l0': checkpoint['model1.lstm1.weight_ih_l0'],
            'weight_hh_l0': checkpoint['model1.lstm1.weight_hh_l0'],
            'bias_ih_l0': checkpoint['model1.lstm1.bias_ih_l0'],
            'bias_hh_l0': checkpoint['model1.lstm1.bias_hh_l0'],
            'weight_ih_l0_reverse': checkpoint['model1.lstm1.weight_ih_l0_reverse'],
            'weight_hh_l0_reverse': checkpoint['model1.lstm1.weight_hh_l0_reverse'],
            'bias_ih_l0_reverse': checkpoint['model1.lstm1.bias_ih_l0_reverse'],
            'bias_hh_l0_reverse': checkpoint['model1.lstm1.bias_hh_l0_reverse']
        })
        self.model1.lstm2.load_state_dict({
            'weight_ih_l0': checkpoint['model1.lstm2.weight_ih_l0'],
            'weight_hh_l0': checkpoint['model1.lstm2.weight_hh_l0'],
            'bias_ih_l0': checkpoint['model1.lstm2.bias_ih_l0'],
            'bias_hh_l0': checkpoint['model1.lstm2.bias_hh_l0'],
            'weight_ih_l0_reverse': checkpoint['model1.lstm2.weight_ih_l0_reverse'],
            'weight_hh_l0_reverse': checkpoint['model1.lstm2.weight_hh_l0_reverse'],
            'bias_ih_l0_reverse': checkpoint['model1.lstm2.bias_ih_l0_reverse'],
            'bias_hh_l0_reverse': checkpoint['model1.lstm2.bias_hh_l0_reverse']
        })
        self.model1.fc_ha.load_state_dict({
            'weight': checkpoint['model1.fc_ha.weight'],
            'bias': checkpoint['model1.fc_ha.bias']
        })
        self.model1.fc_1.load_state_dict({
            'weight': checkpoint['model1.fc_1.weight'],
            'bias': checkpoint['model1.fc_1.bias']
        })

        # Load weights for model2
        self.model2.lstm1.load_state_dict({
            'weight_ih_l0': checkpoint['model2.lstm1.weight_ih_l0'],
            'weight_hh_l0': checkpoint['model2.lstm1.weight_hh_l0'],
            'bias_ih_l0': checkpoint['model2.lstm1.bias_ih_l0'],
            'bias_hh_l0': checkpoint['model2.lstm1.bias_hh_l0'],
            'weight_ih_l0_reverse': checkpoint['model2.lstm1.weight_ih_l0_reverse'],
            'weight_hh_l0_reverse': checkpoint['model2.lstm1.weight_hh_l0_reverse'],
            'bias_ih_l0_reverse': checkpoint['model2.lstm1.bias_ih_l0_reverse'],
            'bias_hh_l0_reverse': checkpoint['model2.lstm1.bias_hh_l0_reverse']
        })
        self.model2.lstm2.load_state_dict({
            'weight_ih_l0': checkpoint['model2.lstm2.weight_ih_l0'],
            'weight_hh_l0': checkpoint['model2.lstm2.weight_hh_l0'],
            'bias_ih_l0': checkpoint['model2.lstm2.bias_ih_l0'],
            'bias_hh_l0': checkpoint['model2.lstm2.bias_hh_l0'],
            'weight_ih_l0_reverse': checkpoint['model2.lstm2.weight_ih_l0_reverse'],
            'weight_hh_l0_reverse': checkpoint['model2.lstm2.weight_hh_l0_reverse'],
            'bias_ih_l0_reverse': checkpoint['model2.lstm2.bias_ih_l0_reverse'],
            'bias_hh_l0_reverse': checkpoint['model2.lstm2.bias_hh_l0_reverse']
        })
        self.model2.fc_ha.load_state_dict({
            'weight': checkpoint['model2.fc_ha.weight'],
            'bias': checkpoint['model2.fc_ha.bias']
        })
        self.model2.fc_1.load_state_dict({
            'weight': checkpoint['model2.fc_1.weight'],
            'bias': checkpoint['model2.fc_1.bias']
        })

        # Load attention weights
        self.ccslnet.att1.load_state_dict({
            'weight': checkpoint['att1.weight'],
            'bias': checkpoint['att1.bias']
        })
        self.ccslnet.att2.load_state_dict({
            'weight': checkpoint['att2.weight'],
            'bias': checkpoint['att2.bias']
        })

        # Load language classifier weights
        self.ccslnet.lang_classifier.load_state_dict({
            'weight': checkpoint['lang_classifier.weight']
        })

        print("Weights loaded successfully!")
        print("Dialect Identification Model loaded!")
        
    def predict_dialect(self, audio_path, wave2vec_model_path):
        
        input_features = self.extract_wav2vec_features(audio_path, wave2vec_model_path)
        X1, X2 = Get_data(input_features)
        X1 = np.swapaxes(X1, 0, 1)
        X2 = np.swapaxes(X2, 0, 1)

        x1 = torch.from_numpy(X1).to(device)
        x2 = torch.from_numpy(X2).to(device)
            # Pass inputs through the model
        with torch.no_grad():
            output = self.forward(x1, x2)
            
        predicted_value = output.argmax().cpu().item()

        # Convert predicted value to dialect
        dialect = next(key for key, value in lan2id.items() if value == predicted_value)
        return dialect   

    def extract_wav2vec_features(self, audio_path, wave2vec_model_path):
          
        wave2vec2_bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
        wave2vec2_model = wave2vec2_bundle.get_model()

        # Load the state dictionary from the given path
        wave2vec2_model.load_state_dict(torch.load(wave2vec_model_path, map_location=device))
        wave2vec2_model = wave2vec2_model.to(device)
        wave2vec2_model.eval()
        print("Wav2Vec 2.0 model loaded!")
        
        print(f"\n\nLoading audio from {audio_path}.")
        X, sample_rate = sf.read(audio_path)
        waveform = Tensor(X)
        waveform = waveform.unsqueeze(0)
    
        if sample_rate != wave2vec2_bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, wave2vec2_bundle.sample_rate)
            waveform = waveform.squeeze(-1)
    
        with torch.inference_mode():
            features, _ = wave2vec2_model.extract_features(waveform)
    
        input_features = torch.squeeze(features[2])
        return input_features