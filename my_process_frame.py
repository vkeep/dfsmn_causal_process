import sys
import torch
from modelscope.fileio import File
from modelscope.models.audio.ans.denoise_net import DfsmnAns
import soundfile as sf
import io
import librosa
import numpy as np
import time

HOP_LENGTH = 960
N_FFT = 1920
WINDOW_NAME_HAM = 'hamming'
STFT_WIN_LEN = 1920
WINLEN = 3840
STRIDE = 1920
device = "cpu"


SAMPLE_RATE = 48000
WINDOW_NAME_HAM = 'hamming'

def load_model(path):
    model = DfsmnAns(path)

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    print("HI start process audio")
    return model


window = torch.hamming_window(
    STFT_WIN_LEN, periodic=False, device=device)
def stft(x):
    return torch.stft(
        x,
        N_FFT,
        HOP_LENGTH,
        STFT_WIN_LEN,
        center=False,
        window=window,
        return_complex=False)


def istft(x, slen):
    return librosa.istft(
        x,
        hop_length=HOP_LENGTH,
        win_length=STFT_WIN_LEN,
        window=WINDOW_NAME_HAM,
        center=False,
        length=slen)
def load_file(in_path):
    with open(in_path, 'rb') as f:
        data_byte = f.read()

    return data_byte

def bytes2tensor(file_bytes):
    data1, fs = sf.read(io.BytesIO(file_bytes))
    data1 = data1.astype(np.float32)
    if len(data1.shape) > 1:
        data1 = data1[:, 0]
    if fs != SAMPLE_RATE:
        data1 = librosa.resample(data1, fs, SAMPLE_RATE)
    data = data1 * 32768
    data_tensor = torch.from_numpy(data).type(torch.FloatTensor)
    return data_tensor

def forward(audio_data, model):
    with torch.no_grad():
        audio_in = audio_data.unsqueeze(0)
        import torchaudio
        fbanks = torchaudio.compliance.kaldi.fbank(
            audio_in,
            dither=1,#1
            frame_length=40.0,
            frame_shift=20.0,
            num_mel_bins=120,
            sample_frequency=SAMPLE_RATE,
            window_type=WINDOW_NAME_HAM)
        #fbanks = fbanks[0,:]
        #fbanks = fbanks.unsqueeze(0)
        #fbanks = fbanks[0]

        #fbanks = fbanks.unsqueeze(0)
        input_buffer = torch.zeros(65536).float().cpu()

        frame_num = fbanks.shape[0]
        for i in range(frame_num):
            fbank = fbanks[i, :]
            fbank = fbank.unsqueeze(0)
            fbank = fbank.unsqueeze(0)

            mask, input_buffer = model(fbank, input_buffer)
            if i == 0:
                masks = mask
            else:
                masks = torch.cat((masks, mask), dim=1)


        spectrum = stft(audio_data)
        masks = masks.permute(2, 1, 0)
        masked_spec = (spectrum * masks).cpu()
        masked_spec = masked_spec.detach().numpy()
    '''
    np.savetxt("./mask_out0.txt", masks[:, 0, :], fmt ="%f", delimiter= " ")
    np.savetxt("./mask_out1.txt", masks[:, 1, :], fmt="%f", delimiter=" ")
    np.savetxt("./mask_out2.txt", masks[:, 2, :], fmt="%f", delimiter=" ")
    np.savetxt("./mask_out3.txt", masks[:, 3, :], fmt="%f", delimiter=" ")
    np.savetxt("ori_stft.txt", spectrum[:, 0, :],fmt = "%f", delimiter=" " )
    np.savetxt("ori_stft1.txt", spectrum[:, 1, :],fmt = "%f", delimiter=" " )
    np.savetxt("ori_stft2.txt", spectrum[:, 2, :],fmt = "%f", delimiter=" " )
    np.savetxt("out_stft.txt", masked_spec[:, 0, :],fmt = "%f", delimiter=" " )
    np.savetxt("out_stft1.txt", masked_spec[:, 1, :], fmt="%f", delimiter=" ")
    '''
    masked_spec_complex = masked_spec[:, :, 0] + 1j * masked_spec[:, :, 1]
    masked_sig = istft(masked_spec_complex, len(audio_data))
    outputs = masked_sig.astype(np.int16).tobytes()
    return outputs

def process(model, path):
    pcms, fs = sf.read(path)
    total_time = pcms.shape[0]/fs
    audio_data = load_file(path)
    audio_tensor = bytes2tensor(audio_data)
    t1 = time.time()
    out_put = forward(audio_tensor, model)
    t2 = time.time()
    tproc = t2-t1
    rtf = tproc/total_time
    print("rtf is ",rtf)


    sf.write('denoise_file.wav',np.frombuffer(out_put, dtype=np.int16),SAMPLE_RATE)

    return out_put

if __name__ == '__main__':
    model = load_model('./speech_dfsmn_ans_psm_48k_causal/pytorch_model.bin')
    process(model, "noisefile_48k.wav")