import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import glob
import random
from torch.autograd import Variable, Function
from torch import optim, Tensor
import subprocess
import librosa
import soundfile as sf
import torchaudio
import argparse

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Wav2Vec 2.0 model from torchaudio pipelines
bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
model_wav2vec = bundle.get_model()

# Load the state dictionary from the given path
model_wav2vec.load_state_dict(torch.load(r"C:\Users\Eve\Downloads\Thoth\B_Viderva_copy\wav2vec2_model.pth", map_location=device))

# Move the model to the appropriate device (CPU or GPU)
model_wav2vec = model_wav2vec.to(device)

# Set the model to evaluation mode
model_wav2vec.eval()

print("Model Wav2Vec 2.0 is loaded")

###########################################################################################################################
def extract_wav2vec2(aud_path):
    #bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    #device = torch.device('cuda')
    #model_wav2vec = bundle.get_model()
    X, sample_rate = sf.read(aud_path)
    waveform = Tensor(X)
    waveform = waveform.unsqueeze(0)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        #print("resampling done")    
        waveform = waveform.squeeze(-1)
    with torch.inference_mode():
        features, _ = model_wav2vec.extract_features(waveform)
    f = torch.squeeze(features[2])
    return f
###########################################################################################################################



def Get_data(X):
    X = X.cpu().numpy()
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1) 
    X = (X - mu) / std 
   
    Xdata1 = []
    Xdata2 = []
    for i in range(0,len(X)-look_back1,1):    #High resolution low context        
        a=X[i:(i+look_back1),:]        
        Xdata1.append(a)
    Xdata1=np.array(Xdata1)

    for i in range(0,len(X)-look_back2,2):     #Low resolution long context       
        b=X[i+1:(i+look_back2):3,:]        
        Xdata2.append(b)
    Xdata2=np.array(Xdata2)
    
    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()
   
    
    return Xdata1,Xdata2

class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(1024, 512,bidirectional=True) #Give input size as 512
        self.lstm2 = nn.LSTM(2*512, 256,bidirectional=True)        
               
        self.fc_ha=nn.Linear(e_dim,256) 
        self.fc_1= nn.Linear(256,1)        
        self.sftmax = torch.nn.Softmax(dim=1)
        #self.lang_classifier = nn.Linear(e_dim, n_classes, bias = False)
        
    def forward(self, x):
        x1, _ = self.lstm1(x) 
        x2, _ = self.lstm2(x1)
        ht = x2[-1]
        ht = torch.unsqueeze(ht, 0) 
              
        ha = torch.tanh(self.fc_ha(ht))
        alp = self.fc_1(ha)
        al = self.sftmax(alp) 
        
       
        T = list(ht.shape)[1]  
        batch_size = list(ht.shape)[0]
        D = list(ht.shape)[2]
        c = torch.bmm(al.view(batch_size, 1, T),ht.view(batch_size,T,D))        
        c = torch.squeeze(c,0)  
        return (c)
#######################################################################################################
class CCSL_Net(nn.Module):
    def __init__(self, model1,model2):
        super(CCSL_Net, self).__init__()
        self.model1 = model1
        self.model2 = model2
        
        self.att1=nn.Linear(e_dim,256) 
        self.att2= nn.Linear(256,1)
        
        self.sftmx = torch.nn.Softmax(dim=1)
        self.lang_classifier = nn.Linear(e_dim, n_classes, bias = False)
        #self.dial_classifier = nn.Linear(e_dim, nc[1], bias = False)
 
        
    def forward(self, x1,x2):
        e1 = self.model1(x1) #e1
        e2 = self.model2(x2) # e2
        
	#This is self-attention to combine the embeddings e1 and e2 into u-vector
        ht_e = torch.cat((e1,e2), dim=0)  
        ht_e = torch.unsqueeze(ht_e, 0) 
        ha_e = torch.tanh(self.att1(ht_e))
        alp = torch.tanh(self.att2(ha_e))
        al= self.sftmx(alp)
        Tb = list(ht_e.shape)[1] 
        batch_size = list(ht_e.shape)[0]
        D = list(ht_e.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb),ht_e.view(batch_size,Tb,D))
        u_vec = torch.squeeze(u_vec,0) # This is u-vector
        #print("uvec shape",u_vec.shape)
        
        lan_prim = self.lang_classifier(u_vec)
        #lan_prim = self.sftmx(lan_prim) 
        
        #dial_prim = self.dial_classifier(u_vec)

        return lan_prim
        
def Prediction_model(X,model):
    X1,X2 = Get_data(X)
    X1 = np.swapaxes(X1,0,1)
    X2 = np.swapaxes(X2,0,1)
    
    x1 = Variable(X1, requires_grad=True).to(device)
    x2 = Variable(X2, requires_grad=True).to(device)
    o1 = model.forward(x1,x2)
    P = o1.argmax()
    P = P.cpu()
    return P


def save_results_to_csv(results, csv_file='Predicted_Dialect.csv'):
    df = pd.DataFrame(results)
    
    if os.path.exists(csv_file):
        # If the CSV file exists, append the new data
        df.to_csv(csv_file, mode='a', header=False, index=False)
        print(f"Results appended to {csv_file}")
    else:
        # If the CSV file does not exist, create a new one
        df.to_csv(csv_file, index=False)
        print(f"Results saved to {csv_file}")

def main_code(file_list, path):
    results = []

    model1 = LSTMNet()
    model2 = LSTMNet()
    model1.to(device)
    model2.to(device)
    model = CCSL_Net(model1, model2)
    model.to(device)
    model.load_state_dict(torch.load(path, map_location=device))

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    for filename in file_list:
        try:
            feature = extract_wav2vec2(filename)
            print(f"Features extracted for {filename}")

            predicted_value = Prediction_model(feature, model)
            dialect = next(key for key, value in lan2id.items() if value == predicted_value)
            
            if dialect == 'MA':
                Dialect_name = "Marwadi"
            elif dialect == 'PU':
                Dialect_name = "Puneri"
            else:
                Dialect_name = "Unknown"
            
            print(f"{filename} has the dialect of {Dialect_name}")
            print("########################################################")
            results.append({'Filename': filename, 'Dialect': Dialect_name})
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if results:
        # Save results to CSV
        save_results_to_csv(results)
    else:
        print("No results to save")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process an audio file or a directory of audio files.")
    parser.add_argument('--audio_path', type=str, help="Path to the audio file or directory containing audio files", required=True)
    args = parser.parse_args()

    lang = ['MA', 'PU']
    lan2id = {'MA': 0, 'PU': 1}
    look_back1 = 30
    look_back2 = 60
    e_dim = 256 * 2
    n_classes = 2  # Number of language classes
    nc = 2
    
    model_path = os.path.join(r"C:\\Users\\Eve\\Downloads\\Thoth\\B_Viderva_copy\\Marathi_Model.pth")

    # "D:\Param\Audio\test"
    
    if os.path.isdir(args.audio_path):
        # Process all .wav files in the directory
        file_list = [os.path.join(args.audio_path, f) for f in os.listdir(args.audio_path) if f.endswith('.wav')]
    else:
        # Process a single file
        file_list = [args.audio_path]
    print(f"Number of audio files to be processed: {len(file_list)}\n")
    main_code(file_list, model_path)