# RB5 + DG5F GR00T Training Guide

RB5 robot arm + DG5F hand dataset을 사용하여 NVIDIA Isaac GR00T N1.7 모델을 fine-tuning하고,
최종 모델을 Hugging Face에 업로드하는 과정.

---

## 1. RunPod 환경

학습에 사용한 환경:

- GPU: NVIDIA A100-SXM4-80GB
- Network Volume: `/workspace`
- Isaac-GR00T repository: `/workspace/Isaac-GR00T`
- Dataset: `/workspace/datasets/rb5_grape_sort`
- Checkpoint directory: `/workspace/checkpoints`
- Python: 3.12
- CUDA: 12.8
- PyTorch: 2.9.0+cu128

RunPod Network Volume은 `/workspace`에 mount하여 사용한다.

---

## 2. SSH 접속 후 환경 확인

Repository로 이동:

```bash
cd /workspace/Isaac-GR00T
```

GPU 확인:

```bash
nvidia-smi
```

PyTorch에서 GPU가 정상적으로 인식되는지 확인:

```bash
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('CUDA available:', torch.cuda.is_available())"
```

정상 실행 예시:

```text
torch: 2.9.0+cu128
CUDA: 12.8
GPU: NVIDIA A100-SXM4-80GB
CUDA available: True
```

---

## 3. Hugging Face 로그인 확인

```bash
uv run hf auth whoami
```

정상적으로 로그인되어 있다면:

```text
user: ganikim
```

로그인이 필요한 경우:

```bash
uv run hf auth login
```

Hugging Face Write Token을 입력한다.

> Hugging Face Token은 GitHub repository나 문서에 저장하지 않는다.

---

## 4. Dataset 확인

사용 Dataset:

```text
/workspace/datasets/rb5_grape_sort
```

Dataset 용량 확인:

```bash
du -sh /workspace/datasets/rb5_grape_sort
```

Metadata 확인:

```bash
cat /workspace/datasets/rb5_grape_sort/meta/info.json
```

Modality 확인:

```bash
cat /workspace/datasets/rb5_grape_sort/meta/modality.json
```

현재 Dataset 구성:

- Robot: RB5 + DG5F
- Total episodes: 13
- FPS: 30
- State dimension: 26
  - RB5 arm: 6
  - DG5F right hand: 20
- Action dimension: 26
  - RB5 arm: 6
  - DG5F right hand: 20
- Camera:
  - `observation.images.front`
  - `observation.images.second`

Task:

```text
Sort one bunch of grapes by color: place green grapes on the right and red grapes on the left.
```

---

## 5. Custom GR00T Config

사용하는 custom config:

```text
examples/RB5_DG5F/rb5_dg5f_config.py
```

확인:

```bash
ls -lh examples/RB5_DG5F/rb5_dg5f_config.py
```

현재 주요 구성:

### Video

- `front`
- `second`

### State

- `rb5_arm`
- `right_hand`

### Action

- `rb5_arm`
- `right_hand`

### Action Representation

RB5:

```text
RELATIVE
```

DG5F:

```text
ABSOLUTE
```

Embodiment:

```text
NEW_EMBODIMENT
```

---

## 6. 학습 전 저장공간 확인

전체 사용량 확인:

```bash
du -sh /workspace/* /workspace/.cache/* 2>/dev/null | sort -h
```

주요 경로:

```text
/workspace/Isaac-GR00T
/workspace/datasets
/workspace/checkpoints
/workspace/.cache/huggingface
```

이전 테스트 checkpoint가 필요 없는 경우:

```bash
rm -rf /workspace/checkpoints/*
```

주의:

`/workspace/.cache/huggingface`에는 다운로드한 GR00T base model이 들어 있을 수 있으므로
특별한 이유가 없다면 삭제하지 않는다.

---

## 7. GR00T Fine-tuning

### Training Configuration

```text
Base Model          : nvidia/GR00T-N1.7-3B
GPU                 : 1
Max Steps           : 10000
Global Batch Size   : 32
Dataloader Workers  : 4
W&B                 : OFF
Save Steps          : 20000
```

10,000 step 학습을 수행한다.

`SAVE_STEPS=20000`으로 설정하여 학습 도중 주기적인 checkpoint 저장은 발생하지 않도록 한다.

### Training Command

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
NUM_GPUS=1 \
MAX_STEPS=10000 \
SAVE_STEPS=20000 \
USE_WANDB=0 \
GLOBAL_BATCH_SIZE=32 \
DATALOADER_NUM_WORKERS=4 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /workspace/datasets/rb5_grape_sort \
  --modality-config-path examples/RB5_DG5F/rb5_dg5f_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /workspace/checkpoints/rb5_grape_sort_10k \
  --experiment-name rb5_grape_sort_10k
```

---

## 8. Training 진행 확인

정상적으로 시작되면 training step과 loss가 출력된다.

예:

```text
0/10000
100/10000
1000/10000
...
10000/10000
```

학습 중에는 RunPod Pod를 Stop하지 않는다.

학습 시간은 GPU, dataset caching, dataloader 상태 등에 따라 달라질 수 있다.

---

## 9. 최종 모델 확인

현재 설정의 최종 모델 경로:

```text
/workspace/checkpoints/rb5_grape_sort_10k/rb5_grape_sort_10k
```

용량 확인:

```bash
du -sh /workspace/checkpoints/rb5_grape_sort_10k/rb5_grape_sort_10k
```

파일 확인:

```bash
ls -lh /workspace/checkpoints/rb5_grape_sort_10k/rb5_grape_sort_10k
```

주요 최종 모델 파일:

```text
config.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
processor/
experiment_cfg/
training_args.bin
wandb_config.json
```

현재 학습 결과에서 최종 model weights는 약 12 GB이다.

---

## 10. checkpoint-10000 주의

학습 종료 후 다음 폴더가 생성될 수 있다.

```text
checkpoint-10000/
```

실제 학습에서는 이 폴더가 약 24 GB를 차지했다.

내부에는 학습 재개를 위한 다음과 같은 파일들이 포함될 수 있다.

```text
optimizer.pt
scheduler.pt
model weights
trainer state
```

따라서 최종 output directory 전체는 약 36 GB가 될 수 있다.

최종 모델을 inference 목적으로 보관하는 경우
Hugging Face 업로드 시 `checkpoint-10000`을 제외한다.

---

## 11. Hugging Face Model Repository 생성

로그인 확인:

```bash
cd /workspace/Isaac-GR00T
uv run hf auth whoami
```

Model repository 생성:

```bash
uv run hf repo create rb5_grape_sort_groot_10k --repo-type model
```

현재 사용한 repository:

```text
ganikim/rb5_grape_sort_groot_10k
```

이미 repository가 존재한다면 다시 생성할 필요가 없다.

---

## 12. 최종 모델 Hugging Face 업로드

전체 output directory를 그대로 업로드하면
`checkpoint-10000`까지 포함되어 약 36 GB 이상이 업로드될 수 있다.

따라서 `checkpoint-10000`을 제외하고 업로드한다.

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run hf upload ganikim/rb5_grape_sort_groot_10k \
  /workspace/checkpoints/rb5_grape_sort_10k/rb5_grape_sort_10k \
  . \
  --repo-type model \
  --exclude "checkpoint-10000/**"
```

업로드 초기에는 다음과 같은 메시지가 나타날 수 있다.

```text
Start hashing ...
Finished hashing ...
Processing Files ...
```

대용량 safetensors 파일을 업로드하기 때문에 시간이 걸릴 수 있다.

업로드 중에는:

- RunPod Stop 금지
- Terminal 종료 금지
- `Ctrl+C` 금지

---

## 13. Hugging Face 업로드 확인

Model Repository:

```text
https://huggingface.co/ganikim/rb5_grape_sort_groot_10k
```

최소한 다음 파일이 존재하는지 확인한다.

```text
config.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
processor/
experiment_cfg/
```

업로드가 정상적으로 완료된 것을 확인한 후 RunPod를 Stop한다.

---

## 14. RunPod 종료

학습과 Hugging Face 업로드가 모두 완료된 후 GPU를 더 이상 사용하지 않는다면
RunPod Pod를 Stop한다.

Network Volume을 계속 유지하면 `/workspace` 데이터는 Pod와 별도로 유지된다.

필요한 데이터가 Hugging Face와 GitHub 등에 모두 백업되었고
Network Volume도 더 이상 필요하지 않은 경우에만 Network Volume을 별도로 삭제한다.

---

## 15. 다음 Dataset 학습 시 변경할 항목

동일한 RB5 + DG5F 구성으로 새로운 Dataset을 학습할 경우 주로 다음 항목을 변경한다.

```text
--dataset-path
--output-dir
--experiment-name
Hugging Face model repository name
```

예:

```text
Dataset:
  /workspace/datasets/NEW_DATASET

Output:
  /workspace/checkpoints/NEW_EXPERIMENT

Experiment:
  NEW_EXPERIMENT

Hugging Face:
  ganikim/NEW_MODEL_NAME
```

Robot state/action 구조 또는 camera 구성이 달라지는 경우에는 다음 파일도 수정한다.

```text
examples/RB5_DG5F/rb5_dg5f_config.py
meta/modality.json
```

---

# Quick Reference

## Train

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
NUM_GPUS=1 \
MAX_STEPS=10000 \
SAVE_STEPS=20000 \
USE_WANDB=0 \
GLOBAL_BATCH_SIZE=32 \
DATALOADER_NUM_WORKERS=4 \
uv run bash examples/finetune.sh \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /workspace/datasets/rb5_grape_sort \
  --modality-config-path examples/RB5_DG5F/rb5_dg5f_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /workspace/checkpoints/rb5_grape_sort_10k \
  --experiment-name rb5_grape_sort_10k
```

## Upload Final Model

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run hf upload ganikim/rb5_grape_sort_groot_10k \
  /workspace/checkpoints/rb5_grape_sort_10k/rb5_grape_sort_10k \
  . \
  --repo-type model \
  --exclude "checkpoint-10000/**"
```

## Check Storage

```bash
du -sh /workspace/* /workspace/.cache/* 2>/dev/null | sort -h
```

## Check Hugging Face Login

```bash
uv run hf auth whoami
```

