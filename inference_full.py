import torch
from torch import nn
from torch.nn import functional as F
from pathlib import Path
from typing import List, Tuple, Union, Optional
import os
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import sys


def load_libritts_item(
    fileid: str,
    path: str,
    ext_audio: str,
    ext_original_txt: str,
    ext_normalized_txt: str,
    target_sample_rate: int = None,
    offset_mode: str = 'start',
    duration: Optional[float] = None,  # seconds
    # gain: float = -3.0,
) -> Tuple[Tensor, int, str, str, int, int, str]:
    speaker_id, chapter_id, segment_id, utterance_id = fileid.split("_")
    utterance_id = fileid
    file_audio = utterance_id + ext_audio
    file_audio = os.path.join(path, speaker_id, chapter_id, file_audio)

    # soundfile 방식
    info = sf.info(file_audio)
    sample_rate = info.samplerate
    total_frames = info.frames
    
    # Calculate frame offset and number of frames to load
    if offset_mode == 'start':
        frame_offset = 0
    elif offset_mode == 'random':
        if total_frames - int(duration * sample_rate) <= 0:
            frame_offset = 0
        else:
            frame_offset = np.random.randint(0, total_frames - int(duration * sample_rate))
    num_frames = -1 if duration is None else int(duration * sample_rate)
    
    # Load audio with offset and duration
    with sf.SoundFile(file_audio, 'r') as f:
        f.seek(frame_offset)
        frames_to_read = num_frames if num_frames != -1 else -1
        waveform = f.read(frames=frames_to_read)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)

    # gain = np.random.uniform(-1, -6) if offset_mode == 'random' else -3
    # waveform, _ = torchaudio.sox_effects.apply_effects_tensor(waveform, sample_rate, [["norm", f"{gain:.2f}"]])

    # Pad if duration is specified and waveform is shorter
    if duration is not None:
        target_length = int(duration * sample_rate)
        current_length = waveform.size(1)
        if current_length < target_length:
            padding_length = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (padding_length,0), mode='constant', value=0)
    
    # Resample if needed
    if target_sample_rate and target_sample_rate != sample_rate:
        resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    return (
        waveform,
        sample_rate,
    )
    
class LibriTTSDataset(Dataset):
    """LibriTTS dataset with customizable audio duration and offset modes.
    
    Args:
        root (str or Path): Path to the directory where the dataset is found
        subsets (str or List[str]): Subset(s) to use
        sample_rate (int, optional): Target sample rate
        duration (float, optional): Fixed duration in seconds for all audio clips
        offset_mode (str): Either 'random' or 'start'. If 'random', picks random offset for each sample
    """
    
    _ext_original_txt = ".original.txt"
    _ext_normalized_txt = ".normalized.txt"
    _ext_audio = ".wav"
    
    def __init__(
        self,
        root: Union[str, Path],
        subsets: Union[str, List[str]],
        sample_rate: int = None,
        duration: Optional[float] = None,
        offset_mode: str = 'start',
    ) -> None:
        if offset_mode not in ['random', 'start']:
            raise ValueError("offset_mode must be either 'random' or 'start'")
            
        root = os.fspath(root)
        # Convert single subset to list
        if isinstance(subsets, str):
            subsets = [subsets]
            
        # Validate subsets
        valid_subsets = {
            "dev-clean", "dev-other", "test-clean", "test-other",
            "train-clean-100", "train-clean-360", "train-other-500"
        }
        for subset in subsets:
            if subset not in valid_subsets:
                raise ValueError(f"Invalid subset '{subset}'. Must be one of {valid_subsets}")
        
        self.sample_rate = sample_rate
        self.duration = duration
        self.offset_mode = offset_mode
        
        # Collect all file paths
        self._walker = []
        for subset in subsets:
            path = os.path.join(root, "LibriTTS/LibriTTS", subset)
            if not os.path.isdir(path):
                raise RuntimeError(f"Dataset not found at {path}")
            
            self._walker.extend([
                (path, str(p.stem)) 
                for p in Path(path).glob(f"*/*/*{self._ext_audio}")
            ])

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int, int, str]:
        """Load the n-th sample from the dataset."""
        path, fileid = self._walker[n]
        
        output = load_libritts_item(
            fileid,
            path,
            self._ext_audio,
            self._ext_original_txt,
            self._ext_normalized_txt,
            self.sample_rate,
            self.offset_mode,
            duration=self.duration,
            # gain=np.random.uniform(-1, -6) if self.offset_mode == 'random' else -3
        )
        return output

    def __len__(self) -> int:
        return len(self._walker)


import numpy as np
import librosa
import matplotlib.pyplot as plt
from IPython.display import Audio, display

def plot_spectrograms_and_audio(x_sample, recon_x, sr=16000, 
                               save_path=None, n_examples=10):
    """
    스펙트로그램과 오디오를 파일로 저장합니다.
    """
    if hasattr(x_sample, 'cpu'):
        x_sample = x_sample.cpu().float().numpy()
    if hasattr(recon_x, 'cpu'):
        recon_x = recon_x.cpu().float().numpy()
        
    n_examples = min(n_examples, len(x_sample), len(recon_x))
    
    # 저장 디렉토리 생성
    spec_dir = os.path.join(save_path, 'spec')
    audio_dir = os.path.join(save_path, 'audio')
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    n_fft = 1024
    hop_length = 256
    
    for i in range(n_examples):
        # 스펙트로그램 저장
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        orig_wave = x_sample[i][0]
        D_orig = librosa.amplitude_to_db(
            np.abs(librosa.stft(orig_wave, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        ax1.imshow(D_orig, origin='lower', aspect='auto', cmap='magma')
        ax1.set_title(f"Original {i+1}")
        ax1.axis('off')

        recon_wave = recon_x[i][0]
        D_recon = librosa.amplitude_to_db(
            np.abs(librosa.stft(recon_wave, n_fft=n_fft, hop_length=hop_length)),
            ref=np.max
        )
        ax2.imshow(D_recon, origin='lower', aspect='auto', cmap='magma')
        ax2.set_title(f"Reconstructed {i+1}")
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(spec_dir, f'spec_{i+1}.png'))
        plt.close()
        
        # 오디오 저장
        sf.write(os.path.join(audio_dir, f'orig_{i+1}.wav'), orig_wave, sr)
        sf.write(os.path.join(audio_dir, f'recon_{i+1}.wav'), recon_wave, sr)



import random
import os
import numpy as np
import math
import gc
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from collections import Counter

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class AverageMeter:
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all the statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the meter with a new value.
        
        Args:
            val (float): The new value to update.
            n (int): The number of occurrences of this value (default is 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterDict:
    """여러 개의 AverageMeter를 딕셔너리 형태로 관리하는 클래스"""
    def __init__(self):
        self.meters = {}

    def reset(self, name=None):
        """특정 미터 또는 모든 미터를 초기화합니다.
        
        Args:
            name (str, optional): 초기화할 미터의 이름. None이면 모든 미터 초기화
        """
        if name is None:
            # 모든 미터 초기화
            for meter in self.meters.values():
                meter.reset()
        else:
            # 특정 미터만 초기화
            if name in self.meters:
                self.meters[name].reset()

    def update(self, name, val, n=1):
        """특정 이름의 미터를 업데이트합니다. 없으면 새로 생성합니다.
        
        Args:
            name (str): 업데이트할 미터의 이름
            val (float): 새로운 값
            n (int): 해당 값의 발생 횟수 (기본값: 1)
        """
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(val, n)

    def get_average(self, name):
        """특정 미터의 평균값을 반환합니다.
        
        Args:
            name (str): 조회할 미터의 이름
            
        Returns:
            float: 해당 미터의 평균값
        """
        if name in self.meters:
            return self.meters[name].avg
        raise KeyError(f"Meter '{name}' is not found.")

    def get_all_averages(self):
        """모든 미터의 평균값을 딕셔너리 형태로 반환합니다.
        
        Returns:
            dict: 미터 이름을 키로, 평균값을 값으로 하는 딕셔너리
        """
        return {name: meter.avg for name, meter in self.meters.items()}
        
def get_cosine_decay_with_warmup(total_steps=1000, warmup_steps=100, max_lr=1e-3, min_lr=1e-7):
    
    def get_lr(step):

        if step < warmup_steps:
            # Linear warmup
            return max_lr * step / warmup_steps
        else:
            # Cosine decay
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
            return min_lr + (max_lr - min_lr) * cosine_decay
        
    return get_lr

class LRScheduler:
    def __init__(self, optimizer, lr_fn):
        self.current_step = 0
        self.optimizer = optimizer
        self.lr_fn = lr_fn
    
    def step(self):
        lr = self.lr_fn(self.current_step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_step += 1
        return lr



from pesq import pesq_batch
from pystoi import stoi

class PESQ:
    def __init__(self, in_sr=16000, sr=16000, on_error=1, mode='wb'):
        self.in_sr = in_sr
        self.sr = sr
        self.on_error = on_error
        self.mode = mode
        if in_sr != sr:
            self.resampler = torchaudio.transforms.Resample(in_sr, sr)
        else:
            self.resampler = None
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, x, y):
        if self.resampler:
            x = self.resampler(x.float().cpu())
            y = self.resampler(y.float().cpu())
        x = x[:,0].float().cpu().numpy()
        y = y[:,0].float().cpu().numpy()
        min_len = min(x.shape[1], y.shape[1])
        x = x[:,:min_len]
        y = y[:,:min_len]
        n = x.shape[0]
        val = np.mean(pesq_batch(fs=self.sr, ref=x, deg=y, on_error=self.on_error, mode=self.mode))
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.avg

class STOI:
    def __init__(self, in_sr=16000, sr=16000):
        self.in_sr = in_sr
        self.sr = sr
        if in_sr != sr:
            self.resampler = torchaudio.transforms.Resample(in_sr, sr)
        else:
            self.resampler = None
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
    
    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        
    def update(self, x, y):
        if self.resampler:
            x = self.resampler(x.float().cpu())
            y = self.resampler(y.float().cpu())
        x = x[:,0].float().cpu().numpy()
        y = y[:,0].float().cpu().numpy()
        min_len = min(x.shape[1], y.shape[1])
        x = x[:,:min_len]
        y = y[:,:min_len]
        n = x.shape[0]
        val = 0
        for ref, deg in zip(x,y):
            val += stoi(x=ref, y=deg, fs_sig=self.sr, extended=False) / n
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def compute(self):
        return self.avg

import os
import pytorch_lightning as pl
import hydra
import librosa
import soundfile as sf
import torch
import numpy as np
from os.path import join, exists, dirname, basename
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader
from data_module import DataModule
from lightning_module import CodecLightningModule
from tqdm import tqdm
from glob import glob
from time import time
from omegaconf import OmegaConf
class BigCodecModel(nn.Module):
    def __init__(self, ckpt_path, config_path):
        super(BigCodecModel, self).__init__()
        
        # 설정 직접 로드
        cfg = OmegaConf.load(config_path)
        
        # Lightning 모듈 생성
        self.lm = CodecLightningModule(cfg=cfg)
        
        # checkpoint 로드
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
        
        # 가중치 로드
        self.lm.load_state_dict(state_dict, strict=True)
        # self.lm.eval()
        
        # # GPU로 이동 (필요한 경우)
        # if torch.cuda.is_available():
        #     self.lm = self.lm.cuda()
        
    def forward(self, x):
        vq_emb = self.lm.model['CodecEnc'](x)
        vq_post_emb, vq_code, _ = self.lm.model['generator'](vq_emb, vq=True)
        recon = self.lm.model['generator'](vq_post_emb, vq=False)
        return {'x_rec': recon, 'indices': vq_code, 'loss': {}}


import librosa
import matplotlib.pyplot as plt
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio, ScaleInvariantSignalDistortionRatio, SignalDistortionRatio, SignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality

def train(model, 
          train_loader=None, 
          val_loader=None,
          eval_loader=None,
          sample_rate=24000,
          val_freq=1,
          visualize_freq=1,
          epochs=5, 
          device='cuda', 
          seed=42,
          weight_decay=0.01,
          lr=2e-4,
          bf16=True,
          torch_compile=False,
          clip_grad=1.0,
          save_path=None):
    seed_everything(seed=seed)
    torch.cuda.empty_cache()
    gc.collect()

    model = model.to(device)
    if torch_compile:
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)
    if hasattr(model, 'quantizer'):
        codebook_size = model.quantizer.codebook_size
    else:
        codebook_size = 1 #No codebook
    print('codebook_size', codebook_size)

    optimizer = torch.optim.AdamW([
        {'params': [param for param in model.parameters() if param.ndim>=2], 'weight_decay': weight_decay},
        {'params': [param for param in model.parameters() if param.ndim<2], 'weight_decay': 0.0}
    ], lr=lr)
    
    accum_metrics = AverageMeterDict()
    si_snr = ScaleInvariantSignalNoiseRatio().to(device)
    si_sdr = ScaleInvariantSignalDistortionRatio().to(device)
    stoi = STOI(in_sr=sample_rate, sr=sample_rate)
    pesq = PESQ(in_sr=sample_rate, sr=16000)
    total_steps = len(train_loader) * epochs if train_loader is not None else 1
    lr_fn = get_cosine_decay_with_warmup(total_steps=total_steps, warmup_steps=0, max_lr=lr, min_lr=1e-7)
    scheduler = LRScheduler(optimizer, lr_fn)

    for epoch in range(1, epochs+1):
        if train_loader is not None:
            model.train()
            accum_metrics.reset()
            # si_snr.reset()
            # si_sdr.reset()
            # stoi.reset()
            # pesq.reset()
            train_codebook_counter = Counter()  # 각 에폭마다 초기화

            pbar = tqdm(train_loader, desc=f'TRAIN epoch {epoch}', total=len(train_loader))
            for data in pbar:
                # print(data)
                x = data[0].to(device)
                with torch.amp.autocast(device_type=device, dtype=torch.bfloat16, enabled=bf16):
                    output = model(x)
                    loss = sum(output['loss'].values())

                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                lr = scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                
                # Codebook usage 모니터링
                if 'indices' in output:
                    indices = output['indices'].flatten().cpu().numpy()
                    train_codebook_counter.update(indices)

                for name, val in output['loss'].items():
                    accum_metrics.update(name, val.detach().cpu().item(), len(x))

                # si_snr.update(output['x_rec'].float(), x.float())
                # stoi.update(output['x_rec'].float(), x.float())
                # pesq.update(output['x_rec'], x)

                metrics_dict = accum_metrics.get_all_averages()
                metrics_dict.update({
                    'lr': f'{lr:.6f}',
                    'grad_norm': f'{norm:.4f}',
                    'codebook_usage': f'{len(train_codebook_counter)/codebook_size:.3f}',
                    # 'si_snr': f'{si_snr.compute().cpu().item():.4f}',
                    # 'stoi': f'{stoi.compute():.4f}',
                    # 'pesq': f'{pesq.compute():.4f}'
                })
                pbar.set_postfix(metrics_dict)
        
        if val_loader is not None and (epoch % val_freq == 0 or epoch == epochs):
            # Validation
            model.eval()
            accum_metrics.reset()
            si_snr.reset()
            si_sdr.reset()
            stoi.reset()
            pesq.reset()
            test_codebook_counter = Counter()

            for data in val_loader:
                x = data[0].to(device)
                x = F.pad(x, (0, (200 - (x.shape[2] % 200))))
                with torch.no_grad():
                    output = model(x)
                        
                    if 'indices' in output:
                        indices = output['indices'].flatten().cpu().numpy()
                        test_codebook_counter.update(indices)
                        
                    for name, val in output['loss'].items():
                        accum_metrics.update(name, val.detach().cpu().item(), len(x))

                    si_snr.update(output['x_rec'].float(), x.float())
                    stoi.update(output['x_rec'].float(), x.float())
                    pesq.update(output['x_rec'].float(), x.float())

            metrics_dict = accum_metrics.get_all_averages()
            metrics_dict['codebook_usage'] = len(test_codebook_counter)/codebook_size
            metrics_dict['si_snr'] = si_snr.compute().cpu().item()
            metrics_dict['si_sdr'] = si_sdr.compute().cpu().item()
            metrics_dict['stoi'] = stoi.compute()
            metrics_dict['pesq'] = pesq.compute()
            val_result = ' '.join([f'val_{name} {val:.4f}' for name, val in metrics_dict.items()])
            print(f'Epoch{epoch}: {val_result}')

        if val_loader is not None and (epoch % visualize_freq == 0 or epoch == epochs):
            model.eval()
            torch.manual_seed(seed)
            for data in val_loader:
                x_sample = data[0][:10].to(device)
                x_sample = F.pad(x_sample, (0, (200 - (x_sample.shape[2] % 200))))
            
            with torch.no_grad():
                output = model(x_sample)
                recon_x = output['x_rec']
            
            # 스펙트로그램과 오디오 플레이어 표시
            plot_spectrograms_and_audio(x_sample, recon_x, sr=sample_rate, save_path=save_path)

    if eval_loader is not None:
        model.eval()
        accum_metrics.reset()
        si_snr.reset()
        si_sdr.reset()
        stoi.reset()
        pesq.reset()
        test_codebook_counter = Counter()

        for data in eval_loader:
            x = data[0].to(device)
            with torch.no_grad():
                output = model(x)
                    
                if 'indices' in output:
                    indices = output['indices'].flatten().cpu().numpy()
                    test_codebook_counter.update(indices)
                    
                for name, val in output['loss'].items():
                    accum_metrics.update(name, val.detach().cpu().item(), len(x))

                si_snr.update(output['x_rec'].float(), x.float())
                si_sdr.update(output['x_rec'].float(), x.float())
                stoi.update(output['x_rec'].float(), x.float())
                pesq.update(output['x_rec'].float(), x.float())

        metrics_dict = accum_metrics.get_all_averages()
        metrics_dict['codebook_usage'] = len(test_codebook_counter)/codebook_size
        metrics_dict['si_snr'] = si_snr.compute().cpu().item()
        metrics_dict['si_sdr'] = si_sdr.compute().cpu().item()
        metrics_dict['stoi'] = stoi.compute()
        metrics_dict['pesq'] = pesq.compute()
        val_result = ' '.join([f'eval_{name} {val:.4f}' for name, val in metrics_dict.items()])
        print(f'Epoch{epoch}: {val_result}')

    # 학습 완료 후 코드북 사용 히스토그램 시각화
    plt.figure(figsize=(15, 5))
    indices = sorted(test_codebook_counter.keys())
    counts = [test_codebook_counter[i] for i in indices]
    # print(indices, counts)
    plt.bar(indices, counts)
    plt.title('Codebook Usage Distribution (Test Set)')
    plt.xlabel('Codebook Index')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()

    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--output_folder', type=str, default='outputs')
    parser.add_argument('--duration', type=float, default=1)
    parser.add_argument('--sample_rate', type=int, default=16000)
    args = parser.parse_args()
    
    # 로그 파일 설정
    out_path = os.path.join(args.save_path, args.output_folder)
    os.makedirs(out_path, exist_ok=True)
    log_path = os.path.join(out_path, 'log.txt')
    
    # stdout을 파일과 콘솔 모두에 출력하도록 설정
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
            
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_path)
    
    # duration 처리
    duration = None if args.duration == -1 else args.duration
    
    config_path = f'{args.save_path}/hydra/config.yaml'
    ckpt_path = f'{args.save_path}/pl_log/last.ckpt'
    model = BigCodecModel(ckpt_path, config_path)

    test_dataset = LibriTTSDataset(
        root="../../../datasets",
        subsets=["test-clean", "test-other"],
        sample_rate=args.sample_rate,
        duration=duration,
        offset_mode="start",
    )

    print(f"테스트 데이터셋 크기: {len(test_dataset)}")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                           drop_last=False, num_workers=4, pin_memory=True)
    
    _ = train(model, None, test_loader, None, 
             sample_rate=args.sample_rate, 
             val_freq=1, 
             visualize_freq=1, 
             epochs=1,
             save_path=out_path)