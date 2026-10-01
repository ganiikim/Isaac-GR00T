# RB5 + DG5F GR00T Training Guide
RB5 robot arm + DG5F hand dataset을 사용하여 NVIDIA Isaac GR00T N1.7 모델을 fine-tuning하고, 최종 모델을 Hugging Face에 업로드하는 
과정. ---
## 1. RunPod 환경
현재 학습에 사용한 환경: - GPU: NVIDIA A100-SXM4-80GB - Network Volume: `/workspace` - Isaac-GR00T repository: `/workspace/Isaac-GR00T` - Dataset: 
`/workspace/datasets/rb5_grape_sort` - Checkpoint directory: `/workspace/checkpoints` - Python: 3.12 - CUDA: 12.8 - PyTorch: 2.9.0+cu128 RunPod의 Network Volume은 
`/workspace`에 mount하여 사용한다. ---
## 2. SSH 접속 후 작업 폴더 이동
```bash cd /workspace/Isaac-GR00T ``` GPU 확인: ```bash nvidia-smi ``` PyTorch에서 GPU가 정상적으로 인식되는지 확인: ```bash uv run python -c 
"import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('CUDA available:', 
torch.cuda.is_available())" ``` 정상 예시: ```text
