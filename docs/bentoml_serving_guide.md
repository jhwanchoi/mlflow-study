# BentoML Model Serving Guide

**작성일**: 2025-01-24
**버전**: 1.0
**대상**: Phase 1 - BentoML 모델 서빙

---

## 개요

### 목표

MLflow로 학습된 PyTorch 비전 모델을 BentoML REST API로 서빙하여 프로덕션 배포 준비

### 핵심 기능

- **MLflow 통합**: MLflow에서 모델 자동 로드
- **REST API**: 이미지 분류 API 엔드포인트
- **배치 처리**: 여러 이미지 동시 예측
- **Docker Compose**: 로컬 환경에서 MLflow + BentoML 통합 실행

---

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                      Client (API 호출)                       │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP POST/GET
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              BentoML Service (port 3000)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  API Endpoints:                                      │   │
│  │  - POST /predict_image (단일 이미지 예측)            │   │
│  │  - POST /predict_batch (배치 이미지 예측)            │   │
│  │  - GET  /get_model_info (모델 정보)                │   │
│  │  - GET  /health (헬스 체크)                          │   │
│  └────────────────────┬─────────────────────────────────┘   │
│                       │                                      │
│  ┌────────────────────┴─────────────────────────────────┐   │
│  │  VisionModelRunnable                                 │   │
│  │  - MLflow 모델 로드                                  │   │
│  │  - PyTorch 추론                                      │   │
│  │  - 이미지 전처리 (resize, normalize)                │   │
│  └──────────────────────────────────────────────────────┘   │
└───────────────────────┬─────────────────────────────────────┘
                        │ MLflow Client
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              MLflow Tracking Server (port 5001)              │
│  - Run artifacts (runs:/<run_id>/model)                     │
│  - Model Registry (models:/<name>/<version>)                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌───────────────────┐              ┌──────────────────────────┐
│   PostgreSQL      │              │   MinIO (S3)             │
│  (Metadata)       │              │   (Model artifacts)      │
└───────────────────┘              └──────────────────────────┘
```

---

## 빠른 시작

### 전제 조건

1. **MLflow 인프라 실행 중**:
   ```bash
   make start  # MLflow, PostgreSQL, MinIO 시작
   ```

2. **학습된 모델 존재**:
   ```bash
   # 모델 학습 (MLflow에 등록)
   make train

   # MLflow UI에서 Run ID 확인
   make mlflow-ui
   # → http://localhost:5001
   ```

### Step 1: 의존성 설치

```bash
# Poetry 의존성 설치 (bentoml 포함)
make install

# 또는 직접 설치
poetry add bentoml ray[tune]
```

### Step 2: BentoML 서버 시작

```bash
# 방법 1: Run ID로 모델 로드
make serve MODEL_RUN_ID=<your_run_id>

# 방법 2: Model Registry에서 로드
make serve MODEL_NAME=vision-model MODEL_VERSION=1

# 예시
make serve MODEL_RUN_ID=abc123def456
```

서버가 시작되면 다음 메시지가 표시됩니다:

```
BentoML server started!

API Endpoints:
  - Predict Image: POST http://localhost:3000/predict_image
  - Predict Batch: POST http://localhost:3000/predict_batch
  - Model Info: GET http://localhost:3000/get_model_info
  - Health: GET http://localhost:3000/health
```

### Step 3: API 테스트

#### Health Check
```bash
curl -X GET http://localhost:3000/health
```

응답:
```json
{
  "status": "healthy",
  "service": "mlflow_vision_classifier"
}
```

#### 모델 정보 조회
```bash
curl -X GET http://localhost:3000/get_model_info | python -m json.tool
```

응답:
```json
{
  "model_metadata": {
    "params": {...},
    "metrics": {...}
  },
  "device": "cpu",
  "classes": ["airplane", "automobile", ...],
  "num_classes": 10
}
```

#### 단일 이미지 예측
```bash
# 이미지 파일을 POST로 전송
curl -X POST http://localhost:3000/predict_image \
  -F "image=@test_image.jpg" \
  | python -m json.tool
```

응답:
```json
{
  "predicted_class": "cat",
  "predicted_index": 3,
  "confidence": 0.8543,
  "probabilities": {
    "airplane": 0.0123,
    "automobile": 0.0456,
    "bird": 0.0234,
    "cat": 0.8543,
    ...
  }
}
```

#### 배치 이미지 예측 (Python)

```python
import requests
import numpy as np
from PIL import Image

# 이미지 로드
images = []
for img_path in ["img1.jpg", "img2.jpg", "img3.jpg"]:
    img = Image.open(img_path).resize((32, 32))
    img_array = np.array(img)
    images.append(img_array)

# Numpy 배열로 변환 (batch, height, width, channels)
batch = np.stack(images)

# API 호출
response = requests.post(
    "http://localhost:3000/predict_batch",
    json={"input": batch.tolist()}
)

results = response.json()
for i, result in enumerate(results):
    print(f"Image {i}: {result['predicted_class']} ({result['confidence']:.2%})")
```

---

## 사용 가능한 명령어

### Makefile 명령어

| 명령어 | 설명 |
|--------|------|
| `make serve` | BentoML 서버 시작 (MODEL_RUN_ID 또는 MODEL_NAME 필요) |
| `make serve-stop` | BentoML 서버 중지 |
| `make serve-logs` | BentoML 서버 로그 확인 |
| `make serve-test` | API 엔드포인트 테스트 (health, model info) |
| `make serve-test-predict` | 이미지 예측 테스트 (test_image.jpg 필요) |
| `make serve-build` | BentoML 서비스 빌드 (bento build) |
| `make serve-list` | 빌드된 BentoML 서비스 목록 |
| `make bentoml-ui` | BentoML API 문서 열기 (http://localhost:3000) |

### 예제

```bash
# 1. MLflow 인프라 시작
make start

# 2. 모델 학습
make train

# 3. MLflow UI에서 Run ID 확인
make mlflow-ui

# 4. BentoML 서버 시작
make serve MODEL_RUN_ID=<your_run_id>

# 5. API 테스트
make serve-test

# 6. 로그 확인
make serve-logs

# 7. 서버 중지
make serve-stop
```

---

## 환경 변수 설정

### .env 파일

BentoML 서버는 다음 환경 변수를 사용합니다:

```bash
# MLflow 설정
MLFLOW_TRACKING_URI=http://localhost:5001

# 모델 선택 (둘 중 하나 필수)
MODEL_RUN_ID=<mlflow_run_id>        # Run ID로 로드
# 또는
MODEL_NAME=<model_name>             # Model Registry에서 로드
MODEL_VERSION=latest                # 버전 (기본값: latest)

# AWS/MinIO 설정 (MLflow S3 아티팩트용)
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
MLFLOW_S3_ENDPOINT_URL=http://minio:9000
```

### Docker Compose 환경 변수

`docker-compose.yml`에서 환경 변수를 오버라이드할 수 있습니다:

```bash
# .env 파일 또는 쉘 환경 변수로 설정
export MODEL_RUN_ID=abc123def456
export MODEL_NAME=vision-model
export MODEL_VERSION=2

# Docker Compose 시작
docker-compose up -d bentoml
```

---

## API 엔드포인트 상세

### 1. POST /predict_image

**단일 이미지 예측**

- **입력**: `multipart/form-data` (image 필드에 이미지 파일)
- **출력**: JSON (predicted_class, confidence, probabilities)
- **지원 형식**: JPEG, PNG, BMP

**cURL 예제**:
```bash
curl -X POST http://localhost:3000/predict_image \
  -F "image=@cat.jpg"
```

**Python 예제**:
```python
import requests

with open("cat.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:3000/predict_image",
        files={"image": f}
    )

result = response.json()
print(f"Predicted: {result['predicted_class']} ({result['confidence']:.2%})")
```

### 2. POST /predict_batch

**배치 이미지 예측**

- **입력**: JSON (numpy 배열을 리스트로 변환)
- **출력**: JSON 배열 (각 이미지별 예측 결과)
- **배치 크기**: 최대 32장 (설정 가능)

**Python 예제**:
```python
import requests
import numpy as np
from PIL import Image

# 이미지 로드 및 전처리
images = []
for path in ["img1.jpg", "img2.jpg"]:
    img = Image.open(path).resize((32, 32))
    images.append(np.array(img))

batch = np.stack(images)  # (2, 32, 32, 3)

# API 호출
response = requests.post(
    "http://localhost:3000/predict_batch",
    json={"input": batch.tolist()}
)

results = response.json()
for i, res in enumerate(results):
    print(f"{i}: {res['predicted_class']} ({res['confidence']:.2%})")
```

### 3. GET /get_model_info

**모델 메타데이터 조회**

- **입력**: 없음
- **출력**: JSON (모델 파라미터, 메트릭, 클래스 정보)

**cURL 예제**:
```bash
curl -X GET http://localhost:3000/get_model_info | python -m json.tool
```

**응답 예시**:
```json
{
  "model_metadata": {
    "params": {
      "learning_rate": "0.001",
      "batch_size": "64",
      "model_name": "mobilenet_v3_small"
    },
    "metrics": {
      "test_accuracy": 0.8543,
      "test_f1": 0.8421
    }
  },
  "device": "cpu",
  "classes": ["airplane", "automobile", ...],
  "num_classes": 10
}
```

### 4. GET /health

**헬스 체크**

- **입력**: 없음
- **출력**: JSON (상태 확인)

```bash
curl -X GET http://localhost:3000/health
```

---

## 고급 사용법

### 1. BentoML 서비스 빌드 및 컨테이너화

```bash
# 1. BentoML 서비스 빌드
cd src/serving
bentoml build

# 2. 빌드된 서비스 확인
bentoml list

# 출력 예시:
# Tag                                  Size       Creation Time
# mlflow_vision_classifier:latest      234.5 MB   2025-01-24 10:30:15

# 3. Docker 이미지로 컨테이너화
bentoml containerize mlflow_vision_classifier:latest

# 4. Docker 이미지 실행
docker run -p 3000:3000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5001 \
  -e MODEL_RUN_ID=abc123 \
  mlflow_vision_classifier:latest
```

### 2. Model Registry 사용

```python
# MLflow Model Registry에 모델 등록
import mlflow

# 1. 모델 등록
run_id = "abc123def456"
model_uri = f"runs:/{run_id}/model"
model_name = "vision-model"

mlflow.register_model(model_uri, model_name)

# 2. 모델 버전 확인
from mlflow.tracking import MlflowClient
client = MlflowClient()

versions = client.search_model_versions(f"name='{model_name}'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")

# 3. Production으로 승격
client.transition_model_version_stage(
    name=model_name,
    version="1",
    stage="Production"
)
```

```bash
# BentoML에서 Model Registry 사용
make serve MODEL_NAME=vision-model MODEL_VERSION=1
```

### 3. 성능 튜닝

#### bentofile.yaml 수정

```yaml
# src/serving/bentofile.yaml
service: "service:svc"

python:
  requirements_txt: "./requirements.txt"

docker:
  python_version: "3.11"

  # GPU 지원 (선택)
  cuda_version: "11.8"

  # 리소스 제한
  env:
    BENTOML_NUM_WORKERS: "4"  # Worker 프로세스 수
    BENTOML_RUNNER_TIMEOUT: "300"  # 타임아웃 (초)
```

#### Runner 설정 (service.py)

```python
# src/serving/service.py
vision_model_runner = bentoml.Runner(
    VisionModelRunnable,
    name="vision_model_runner",
    max_batch_size=64,        # 배치 크기 증가
    max_latency_ms=500,       # 레이턴시 감소
    runnable_init_params={
        "device": "cuda"      # GPU 사용
    }
)
```

### 4. 로드 테스트

```bash
# Apache Bench 사용
ab -n 1000 -c 10 -p test_image.jpg -T multipart/form-data \
  http://localhost:3000/predict_image

# 또는 Locust 사용
pip install locust
locust -f tests/load_test.py
```

**tests/load_test.py** 예제:
```python
from locust import HttpUser, task, between

class VisionAPIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def predict_image(self):
        with open("test_image.jpg", "rb") as f:
            self.client.post("/predict_image", files={"image": f})

    @task
    def health_check(self):
        self.client.get("/health")
```

---

## 문제 해결

### 1. 모델 로드 실패

**증상**:
```
RuntimeError: Failed to load model from MLflow
```

**해결**:
```bash
# 1. MLflow 서버 확인
curl http://localhost:5001/health

# 2. Run ID 확인
make mlflow-ui

# 3. 환경 변수 확인
docker-compose exec bentoml env | grep MODEL

# 4. MLflow 로그 확인
make logs
```

### 2. BentoML 서버 시작 실패

**증상**:
```
Error: Either MODEL_RUN_ID or MODEL_NAME must be set
```

**해결**:
```bash
# 환경 변수 설정 확인
echo $MODEL_RUN_ID
echo $MODEL_NAME

# .env 파일에 추가
echo "MODEL_RUN_ID=abc123" >> .env

# 또는 명령어로 직접 전달
make serve MODEL_RUN_ID=abc123
```

### 3. 이미지 예측 오류

**증상**:
```
ValueError: Image must be RGB format
```

**해결**:
```python
# 이미지를 RGB로 변환
from PIL import Image

img = Image.open("test.png")
if img.mode != "RGB":
    img = img.convert("RGB")
img.save("test_rgb.jpg")
```

### 4. 메모리 부족

**증상**:
```
OOM (Out of Memory) error
```

**해결**:
```yaml
# docker-compose.yml에 메모리 제한 추가
services:
  bentoml:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

---

## 다음 단계

1. **Phase 2: Ray Tune 통합**
   - 하이퍼파라미터 최적화
   - 최적 모델 자동 배포

2. **Phase 3: 프로덕션 배포**
   - Kubernetes로 확장
   - Auto-scaling 설정
   - 모니터링 및 로깅

3. **고급 기능**
   - A/B 테스트
   - 모델 버저닝
   - Canary 배포

---

**참고 자료**:
- [BentoML Documentation](https://docs.bentoml.org/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [BentoML + MLflow Integration](https://docs.bentoml.org/en/latest/integrations/mlflow.html)

**작성자**: MLOps Team
**최종 업데이트**: 2025-01-24
