'''
from flask import Flask, request, jsonify, render_template, send_from_directory
import torchaudio
import soundfile as sf
import torch
from torch import Tensor
import os
import numpy as np
import random
from torch.autograd import Variable
from flask_socketio import SocketIO  # Comment out if you can't use it
import torch.nn as nn

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# socketio = SocketIO(app)  # Comment out if you can't use it
logs = []
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

lang = ['MA', 'PU']
lan2id = {'MA': 0, 'PU': 1}
look_back1 = 30
look_back2 = 60
e_dim = 256 * 2
n_classes = 2  # Number of language classes
nc = 2

def Get_data(X):
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

    Xdata1 = torch.from_numpy(Xdata1).float()
    Xdata2 = torch.from_numpy(Xdata2).float()

    return Xdata1, Xdata2

class LSTMNet(torch.nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.lstm1 = nn.LSTM(1024, 512, bidirectional=True)
        self.lstm2 = nn.LSTM(2*512, 256, bidirectional=True)

        self.fc_ha = nn.Linear(e_dim, 256)
        self.fc_1 = nn.Linear(256, 1)
        self.sftmax = torch.nn.Softmax(dim=1)

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

        self.sftmx = torch.nn.Softmax(dim=1)
        self.lang_classifier = nn.Linear(e_dim, n_classes, bias=False)

    def forward(self, x1, x2):
        e1 = self.model1(x1)
        e2 = self.model2(x2)

        ht_e = torch.cat((e1, e2), dim=0)
        ht_e = torch.unsqueeze(ht_e, 0)
        ha_e = torch.tanh(self.att1(ht_e))
        alp = torch.tanh(self.att2(ha_e))
        al = self.sftmx(alp)
        Tb = list(ht_e.shape)[1]
        batch_size = list(ht_e.shape)[0]
        D = list(ht_e.shape)[2]
        u_vec = torch.bmm(al.view(batch_size, 1, Tb), ht_e.view(batch_size, Tb, D))
        u_vec = torch.squeeze(u_vec, 0)

        lan_prim = self.lang_classifier(u_vec)

        return lan_prim

def load_models():
    global model_wav2vec, Model_Marathi, device, bundle

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_LARGE_960H
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_wav2vec = bundle.get_model()
    model_wav2vec.load_state_dict(torch.load('../wav2vec2_model.pth', map_location=device))
    model_wav2vec = model_wav2vec.to(device)
    model_wav2vec.eval()
    print("Modelwav2vec2 is loaded")

    model1 = LSTMNet()
    model2 = LSTMNet()
    Model_Marathi = CCSL_Net(model1, model2)
    Model_Marathi.load_state_dict(torch.load('../Marathi_Model.pth', map_location=device))
    Model_Marathi.eval()

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    print("ModelMarathi is loaded")

def extract_wav2vec2(aud_path):
    # socketio.emit('log', {'data': 'Reading audio file'})
    X, sample_rate = sf.read(aud_path)
    waveform = Tensor(X).to(device)
    waveform = waveform.unsqueeze(0)
    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        waveform = waveform.squeeze(-1)
    with torch.inference_mode():
        features, _ = model_wav2vec.extract_features(waveform.to('cpu'))
        # socketio.emit('log', {'data': 'Feature extraction done.'})
    f = torch.squeeze(features[2])
    return f

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/projects.html')
def projects():
    return send_from_directory('static', 'projects.html')

@app.route('/contact.html')
def contact():
    return send_from_directory('static', 'contact.html')

@app.route('/update_vercel.html')
def demo():
    return render_template('update_vercel.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/replay', methods=['GET'])
def replay_file():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    filename = None
    if files:
        filename = files[0]
        return send_from_directory(directory=app.config['UPLOAD_FOLDER'], path=filename)
    return 'No file to replay'


@app.route('/extract_features', methods=['POST'])
def extract_features():
    global logs
    logs = []  # Clear logs for new request
    log_message('Clear logs')
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    try:
        log_message('Reading audio file')
        features = extract_wav2vec2(file_path)
        log_message('Processing features for prediction.')
        X1, X2 = Get_data(features.to('cpu'))
        X1 = np.swapaxes(X1, 0, 1)
        X2 = np.swapaxes(X2, 0, 1)
        x1 = Variable(torch.Tensor(X1), requires_grad=True).to('cpu')
        x2 = Variable(torch.Tensor(X2), requires_grad=True).to('cpu')
        o1 = Model_Marathi.forward(x1, x2)
        predicted_value = o1.argmax().cpu().item()
        dialect_mapping = {
            'PU': 'Puneri',
            'MA': 'Marawadi'
        }
        dialect = next(key for key, value in lan2id.items() if value == predicted_value)
        dialect = dialect_mapping.get(dialect, 'Unknown')
        log_message(f'Detected dialect is {dialect}.')
        log_message('Thank you and have a nice day.')
        return jsonify({'dialect': dialect})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/logs', methods=['GET'])
def get_logs():
    return jsonify(logs)

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000)
'''
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello, World!"

if __name__ == "__main__":
    app.run()
