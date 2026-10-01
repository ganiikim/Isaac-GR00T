# RB5 + DG5F GR00T Training Guide

RB5 robot arm + DG5F hand dataset을 사용하여 NVIDIA Isaac GR00T N1.7 모델을 fine-tuning하고,
최종 모델을 Hugging Face에 업로드하는 과정.

전체 파이프라인:

```text
[Local PC]

Teleoperation data collection
        ↓
convert_rb5_grape.py
        ↓
LeRobot v2.1 conversion
        ↓
stats.json / relative_stats.json generation
        ↓
Hugging Face Dataset upload

[RunPod]

Hugging Face Dataset download
        ↓
GR00T Fine-tuning
        ↓
Final Model
        ↓
Hugging Face Model upload
```

---

## 0. Dataset / Model 이름 설정

RunPod SSH 접속 후 이번 학습에 사용할 Dataset 이름을 먼저 지정한다.

예:

```bash
DATASET_NAME=rb5_grape_sort_v2
MODEL_NAME=${DATASET_NAME}_groot_10k
```

설정값 확인:

```bash
echo "DATASET_NAME=$DATASET_NAME"
echo "MODEL_NAME=$MODEL_NAME"
```

위 예시에서는 자동으로 다음과 같이 사용된다.

```text
Hugging Face Dataset:
ganikim/rb5_grape_sort_v2

RunPod Dataset:
 /workspace/datasets/rb5_grape_sort_v2

Training Output:
 /workspace/checkpoints/rb5_grape_sort_v2_10k

Experiment:
rb5_grape_sort_v2_10k

Hugging Face Model:
ganikim/rb5_grape_sort_v2_groot_10k
```

다음 Dataset이 `rb5_grape_sort_v3`인 경우에는 다음 부분만 변경하면 된다.

```bash
DATASET_NAME=rb5_grape_sort_v3
MODEL_NAME=${DATASET_NAME}_groot_10k
```

이후 명령은 수정하지 않고 그대로 사용한다.

> SSH 연결을 종료하거나 새로운 shell을 실행하면 환경변수가 초기화될 수 있다.
> 새로운 SSH 세션에서는 위의 `DATASET_NAME`, `MODEL_NAME`을 다시 설정한다.

---

## 1. RunPod 환경

학습에 사용한 환경:

- GPU: NVIDIA A100-SXM4-80GB
- Network Volume: `/workspace`
- Isaac-GR00T repository: `/workspace/Isaac-GR00T`
- Dataset root: `/workspace/datasets`
- Checkpoint root: `/workspace/checkpoints`
- Python: 3.12
- CUDA: 12.8
- PyTorch: 2.9.0+cu128

RunPod Network Volume은 `/workspace`에 mount하여 사용한다.

Network Volume을 유지하면 Pod를 Stop 또는 Terminate한 뒤에도 `/workspace`의 데이터는 유지된다.

---

## 2. SSH 접속 후 환경 확인

Repository로 이동:

```bash
cd /workspace/Isaac-GR00T
```

현재 Dataset 설정 확인:

```bash
echo "DATASET_NAME=$DATASET_NAME"
echo "MODEL_NAME=$MODEL_NAME"
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
cd /workspace/Isaac-GR00T

uv run hf auth whoami
```

정상적으로 로그인되어 있다면 사용자 계정이 출력된다.

예:

```text
ganikim
```

로그인이 필요한 경우:

```bash
uv run hf auth login
```

Hugging Face Write Token을 입력한다.

> Hugging Face Token은 GitHub repository나 문서에 저장하지 않는다.

---

## 4. Hugging Face Dataset 다운로드

Local PC에서 `convert_rb5_grape.py` 실행이 완료되었다면 Dataset은 다음 Hugging Face repository에 업로드되어 있어야 한다.

```text
ganikim/$DATASET_NAME
```

Dataset을 저장할 directory 생성:

```bash
mkdir -p /workspace/datasets/$DATASET_NAME
```

Hugging Face에서 Dataset 다운로드:

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run hf download ganikim/$DATASET_NAME \
  --repo-type dataset \
  --local-dir /workspace/datasets/$DATASET_NAME
```

다운로드 완료 후 확인:

```bash
ls -lh /workspace/datasets/$DATASET_NAME
```

Dataset 용량 확인:

```bash
du -sh /workspace/datasets/$DATASET_NAME
```

---

## 5. Dataset Metadata 확인

Dataset metadata 확인:

```bash
cat /workspace/datasets/$DATASET_NAME/meta/info.json
```

Modality 확인:

```bash
cat /workspace/datasets/$DATASET_NAME/meta/modality.json
```

GR00T normalization statistics 확인:

```bash
ls -lh /workspace/datasets/$DATASET_NAME/meta/stats.json
ls -lh /workspace/datasets/$DATASET_NAME/meta/relative_stats.json
```

현재 RB5 + DG5F Dataset의 기본 구성:

- Robot: RB5 + DG5F
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

Episode 수와 Total frame 수는 Dataset마다 달라질 수 있으므로 다음 파일에서 확인한다.

```bash
cat /workspace/datasets/$DATASET_NAME/meta/info.json
```

---

## 6. Custom GR00T Config

사용하는 custom config:

```text
examples/RB5_DG5F/rb5_dg5f_config.py
```

확인:

```bash
cd /workspace/Isaac-GR00T

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

`relative_stats.json`은 현재 modality config 및 action representation과 일치해야 한다.

Robot state/action 구조, action horizon 또는 modality configuration을 변경한 경우에는 Dataset statistics를 다시 생성해야 한다.

---

## 7. 학습 전 저장공간 확인

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

> 위 명령은 `/workspace/checkpoints` 아래의 모든 checkpoint를 삭제하므로 필요한 학습 결과가 없는지 먼저 확인한다.

`/workspace/.cache/huggingface`에는 다운로드한 GR00T base model이 들어 있을 수 있으므로 특별한 이유가 없다면 삭제하지 않는다.

---

## 8. GR00T Fine-tuning

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
  --dataset-path /workspace/datasets/$DATASET_NAME \
  --modality-config-path examples/RB5_DG5F/rb5_dg5f_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /workspace/checkpoints/${DATASET_NAME}_10k \
  --experiment-name ${DATASET_NAME}_10k
```

예를 들어:

```bash
DATASET_NAME=rb5_grape_sort_v2
```

이면 실제 사용되는 경로는:

```text
Dataset:
/workspace/datasets/rb5_grape_sort_v2

Output:
/workspace/checkpoints/rb5_grape_sort_v2_10k

Experiment:
rb5_grape_sort_v2_10k
```

---

## 9. Training 진행 확인

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

학습 시간은 GPU, Dataset caching, dataloader 상태 등에 따라 달라질 수 있다.

---

## 10. 최종 모델 확인

현재 Dataset에 대한 output directory:

```bash
echo /workspace/checkpoints/${DATASET_NAME}_10k
```

최종 모델 directory:

```bash
echo /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k
```

용량 확인:

```bash
du -sh /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k
```

파일 확인:

```bash
ls -lh /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k
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

이전 10,000 step 학습에서는 최종 model weights가 약 12 GB였다.

---

## 11. checkpoint-10000 주의

10,000 step 학습 종료 후 다음 directory가 생성될 수 있다.

```bash
echo /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k/checkpoint-10000
```

이전 학습에서는 이 directory가 약 24 GB를 차지했다.

내부에는 학습 재개를 위한 다음과 같은 파일들이 포함될 수 있다.

```text
optimizer.pt
scheduler.pt
model weights
trainer state
```

따라서 최종 output directory 전체는 약 36 GB가 될 수 있다.

Inference용 최종 모델만 Hugging Face에 보관할 경우 `checkpoint-10000`은 업로드하지 않는다.

---

## 12. Hugging Face Model Repository 생성

현재 Model 이름 확인:

```bash
echo $MODEL_NAME
```

예:

```text
rb5_grape_sort_v2_groot_10k
```

로그인 확인:

```bash
cd /workspace/Isaac-GR00T

uv run hf auth whoami
```

Model repository 생성:

```bash
uv run hf repo create $MODEL_NAME \
  --repo-type model \
  --exist-ok
```

예를 들어:

```bash
DATASET_NAME=rb5_grape_sort_v2
MODEL_NAME=${DATASET_NAME}_groot_10k
```

이면 생성되는 repository는:

```text
ganikim/rb5_grape_sort_v2_groot_10k
```

이미 repository가 존재한다면 `--exist-ok`에 의해 그대로 사용한다.

---

## 13. 최종 모델 Hugging Face 업로드

전체 output directory를 그대로 업로드하면 `checkpoint-10000`까지 포함될 수 있으므로 최종 model directory를 업로드하면서 checkpoint directory를 제외한다.

현재 설정 확인:

```bash
echo "DATASET_NAME=$DATASET_NAME"
echo "MODEL_NAME=$MODEL_NAME"
```

업로드:

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run hf upload ganikim/$MODEL_NAME \
  /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k \
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

## 14. Hugging Face 업로드 확인

현재 Model repository 이름:

```bash
echo "ganikim/$MODEL_NAME"
```

Hugging Face에서 해당 Model repository를 열고 최소한 다음 파일이 존재하는지 확인한다.

```text
config.json
model-00001-of-00003.safetensors
model-00002-of-00003.safetensors
model-00003-of-00003.safetensors
model.safetensors.index.json
processor/
experiment_cfg/
```

`checkpoint-10000/`이 업로드되지 않았는지도 확인한다.

업로드가 정상적으로 완료된 것을 확인한 후 RunPod를 Stop한다.

---

## 15. 학습 완료 후 Local Checkpoint 삭제

Hugging Face에 최종 모델이 정상적으로 업로드된 것을 확인한 뒤 Network Volume 공간을 확보하려면 현재 학습 결과를 삭제할 수 있다.

삭제 전 경로 확인:

```bash
echo /workspace/checkpoints/${DATASET_NAME}_10k
```

용량 확인:

```bash
du -sh /workspace/checkpoints/${DATASET_NAME}_10k
```

정말 필요 없는 것이 확인된 후 삭제:

```bash
rm -rf /workspace/checkpoints/${DATASET_NAME}_10k
```

> Hugging Face 업로드가 정상적으로 완료된 것을 확인하기 전에는 삭제하지 않는다.

---

## 16. RunPod 종료

학습과 Hugging Face 업로드가 모두 완료된 후 GPU를 더 이상 사용하지 않는다면 RunPod Pod를 Stop한다.

Network Volume을 계속 유지하면 `/workspace` 데이터는 Pod와 별도로 유지된다.

따라서 다음 작업에서 새로운 Pod를 생성한 후 동일한 Network Volume을 `/workspace`에 mount하여 다시 사용할 수 있다.

필요한 데이터가 Hugging Face와 GitHub 등에 모두 백업되었고 Network Volume도 더 이상 필요하지 않은 경우에만 Network Volume을 별도로 삭제한다.

---

## 17. 다음 Dataset 학습

동일한 RB5 + DG5F 구성으로 새로운 Dataset을 학습할 경우 새로운 SSH session에서 Dataset 이름만 변경한다.

예:

```bash
DATASET_NAME=rb5_grape_sort_v3
MODEL_NAME=${DATASET_NAME}_groot_10k
```

이후 Dataset 다운로드부터 동일한 명령을 사용한다.

자동으로 다음과 같이 설정된다.

```text
Hugging Face Dataset:
ganikim/rb5_grape_sort_v3

RunPod Dataset:
/workspace/datasets/rb5_grape_sort_v3

Training Output:
/workspace/checkpoints/rb5_grape_sort_v3_10k

Experiment:
rb5_grape_sort_v3_10k

Hugging Face Model:
ganikim/rb5_grape_sort_v3_groot_10k
```

Robot state/action 구조 또는 camera 구성이 달라지는 경우에는 다음 파일도 확인 및 수정한다.

```text
examples/RB5_DG5F/rb5_dg5f_config.py
meta/modality.json
```

---

# Quick Reference

새 Dataset을 학습할 때 아래 순서대로 실행한다.

## 1. Dataset / Model 이름 설정

이번 Dataset 이름만 변경한다.

```bash
DATASET_NAME=rb5_grape_sort_v2
MODEL_NAME=${DATASET_NAME}_groot_10k
```

확인:

```bash
echo "DATASET_NAME=$DATASET_NAME"
echo "MODEL_NAME=$MODEL_NAME"
```

---

## 2. Hugging Face 로그인 확인

```bash
cd /workspace/Isaac-GR00T

uv run hf auth whoami
```

필요한 경우:

```bash
uv run hf auth login
```

---

## 3. Dataset 다운로드

```bash
cd /workspace/Isaac-GR00T

mkdir -p /workspace/datasets/$DATASET_NAME

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run hf download ganikim/$DATASET_NAME \
  --repo-type dataset \
  --local-dir /workspace/datasets/$DATASET_NAME
```

확인:

```bash
ls -lh /workspace/datasets/$DATASET_NAME
ls -lh /workspace/datasets/$DATASET_NAME/meta/stats.json
ls -lh /workspace/datasets/$DATASET_NAME/meta/relative_stats.json
```

---

## 4. GPU 확인

```bash
nvidia-smi
```

```bash
uv run python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('CUDA available:', torch.cuda.is_available())"
```

---

## 5. Train

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
  --dataset-path /workspace/datasets/$DATASET_NAME \
  --modality-config-path examples/RB5_DG5F/rb5_dg5f_config.py \
  --embodiment-tag NEW_EMBODIMENT \
  --output-dir /workspace/checkpoints/${DATASET_NAME}_10k \
  --experiment-name ${DATASET_NAME}_10k
```

---

## 6. 최종 모델 확인

```bash
du -sh /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k
```

```bash
ls -lh /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k
```

---

## 7. Hugging Face Model Repository 생성

```bash
uv run hf repo create $MODEL_NAME \
  --repo-type model \
  --exist-ok
```

---

## 8. Upload Final Model

```bash
cd /workspace/Isaac-GR00T

HF_HUB_ENABLE_HF_TRANSFER=0 \
uv run hf upload ganikim/$MODEL_NAME \
  /workspace/checkpoints/${DATASET_NAME}_10k/${DATASET_NAME}_10k \
  . \
  --repo-type model \
  --exclude "checkpoint-10000/**"
```

---

## 9. Check Storage

```bash
du -sh /workspace/* /workspace/.cache/* 2>/dev/null | sort -h
```

---

## 10. Check Hugging Face Login

```bash
uv run hf auth whoami
```

---

# Local PC Dataset Conversion Quick Reference

새로운 Teleoperation Dataset을 수집한 후 Local PC에서 실행한다.

예를 들어 원본 데이터가 다음 위치에 있는 경우:

```text
~/rb5_teleop/dataset/1/
```

Dataset 변환:

```bash
cd ~/Isaac-GR00T

python convert_rb5_grape.py \
  --dataset-id 1 \
  --dataset-name rb5_grape_sort_v2
```

정상적으로 완료되면 다음 과정이 자동으로 수행된다.

```text
Episode auto discovery
        ↓
LeRobot v2.1 conversion
        ↓
GR00T stats.json generation
        ↓
GR00T relative_stats.json generation
        ↓
Hugging Face Dataset upload
```

Hugging Face Dataset:

```text
ganikim/rb5_grape_sort_v2
```

이후 RunPod에 접속하여:

```bash
DATASET_NAME=rb5_grape_sort_v2
MODEL_NAME=${DATASET_NAME}_groot_10k
```

를 설정하고 위의 `Quick Reference` 순서대로 학습한다.