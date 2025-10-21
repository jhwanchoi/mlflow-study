# VSCode 개발 환경 설정 가이드

**작성일**: 2025-10-21
**버전**: 1.0
**대상**: Phase 5.3 - 클라이언트 마이그레이션

---

## 목차

1. [개요](#개요)
2. [환경 변수 설정](#환경-변수-설정)
3. [VSCode 설정](#vscode-설정)
4. [Python 환경 구성](#python-환경-구성)
5. [MLflow 연결 테스트](#mlflow-연결-테스트)
6. [개발 워크플로우](#개발-워크플로우)
7. [문제 해결](#문제-해결)

---

## 개요

### 목표

로컬 VSCode 환경에서 원격 EKS MLflow 서버로 전환하여:
- **기존 코드 수정 없음**: 환경 변수만 변경
- **원격 실험 추적**: 모든 실험이 중앙 서버에 기록
- **S3 아티팩트**: 모델과 아티팩트가 S3에 자동 저장
- **팀 협업**: 다른 팀원의 실험 결과 공유

### 변경 사항

**이전 (로컬)**:
```bash
MLFLOW_TRACKING_URI=http://localhost:5001
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000  # MinIO
AWS_ACCESS_KEY_ID=minio
AWS_SECRET_ACCESS_KEY=minio123
```

**이후 (원격)**:
```bash
MLFLOW_TRACKING_URI=https://mlflow.mdpg.ai
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
# S3 접근은 자동 (로컬 개발 시 AWS credentials 필요)
AWS_REGION=us-west-2
```

---

## 환경 변수 설정

### 1. .env 파일 생성

프로젝트 루트에 `.env` 파일 생성:

```bash
# 기존 .env 백업
cp .env .env.local.backup

# 새 .env 파일 생성
cat > .env <<'EOF'
# ==========================================
# MLflow Configuration - Remote (EKS)
# ==========================================

# MLflow Tracking Server
MLFLOW_TRACKING_URI=https://mlflow.mdpg.ai

# MLflow Authentication
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password

# AWS Configuration
AWS_REGION=us-west-2
AWS_DEFAULT_REGION=us-west-2

# S3 Access (로컬 개발 시 필요)
# Option 1: AWS CLI credentials 사용 (권장)
# aws configure로 설정한 credentials 자동 사용

# Option 2: 직접 입력 (보안상 권장하지 않음)
# AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY
# AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY

# ==========================================
# Training Configuration (기존과 동일)
# ==========================================

EXPERIMENT_NAME=vision-model-training
MODEL_NAME=mobilenet_v3_small
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=20
NUM_WORKERS=4

# Device Configuration (로컬 M2 Mac)
DEVICE=mps

# Data Configuration
DATA_DIR=./data
DATASET=CIFAR10
NUM_CLASSES=10

# Logging
LOG_LEVEL=INFO
EOF
```

### 2. 사용자별 .env 파일

팀원마다 다른 계정을 사용하므로 `.env.local` 파일 생성:

```bash
# .env.local (gitignore에 추가)
cat > .env.local <<EOF
MLFLOW_TRACKING_USERNAME=ml_engineer_1
MLFLOW_TRACKING_PASSWORD=SecurePassword3!
EOF
```

`.env` 파일에서 `.env.local` import:

```bash
# .env 파일 수정
echo '# Load user-specific credentials' >> .env
echo 'set -a; [ -f .env.local ] && . .env.local; set +a' >> .env
```

또는 Python code에서:

```python
# src/config/settings.py
from dotenv import load_dotenv
import os

# 먼저 .env 로드
load_dotenv(".env")

# 그 다음 .env.local 로드 (덮어쓰기)
load_dotenv(".env.local", override=True)
```

### 3. AWS Credentials 설정 (로컬 개발용)

로컬에서 S3 아티팩트를 업로드하려면 AWS credentials 필요:

```bash
# AWS CLI 설정
aws configure

# AWS Access Key ID: YOUR_ACCESS_KEY
# AWS Secret Access Key: YOUR_SECRET_KEY
# Default region name: us-west-2
# Default output format: json

# 또는 ~/.aws/credentials 파일 직접 편집
cat >> ~/.aws/credentials <<EOF

[mlops]
aws_access_key_id = YOUR_ACCESS_KEY
aws_secret_access_key = YOUR_SECRET_KEY
EOF

# Profile 사용 시
export AWS_PROFILE=mlops
```

---

## VSCode 설정

### 1. settings.json 설정

`.vscode/settings.json` 생성:

```json
{
  // Python 설정
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
  "python.envFile": "${workspaceFolder}/.env",

  // 환경 변수 자동 로드
  "python.terminal.activateEnvironment": true,

  // Linting & Formatting
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,

  // Testing
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "tests",
    "-v",
    "--tb=short"
  ],

  // Files to exclude
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".pytest_cache": true,
    ".mypy_cache": true,
    "mlruns": true  // 로컬 mlruns 폴더 숨기기
  },

  // Terminal 환경 변수
  "terminal.integrated.env.osx": {
    "MLFLOW_TRACKING_URI": "https://mlflow.mdpg.ai"
  },
  "terminal.integrated.env.linux": {
    "MLFLOW_TRACKING_URI": "https://mlflow.mdpg.ai"
  }
}
```

### 2. launch.json 설정 (디버깅)

`.vscode/launch.json` 생성:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Python: Train Model (Remote MLflow)",
      "type": "python",
      "request": "launch",
      "module": "src.training.train",
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env",
      "env": {
        "PYTHONUNBUFFERED": "1",
        "EPOCHS": "5"  // 디버깅 시 짧게
      },
      "justMyCode": false
    },
    {
      "name": "Python: Pytest (Current File)",
      "type": "python",
      "request": "launch",
      "module": "pytest",
      "args": [
        "${file}",
        "-v",
        "--tb=short"
      ],
      "console": "integratedTerminal",
      "envFile": "${workspaceFolder}/.env.test"
    }
  ]
}
```

### 3. tasks.json 설정 (빌드 태스크)

`.vscode/tasks.json` 생성:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Train Model (Remote MLflow)",
      "type": "shell",
      "command": "poetry run python -m src.training.train",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "problemMatcher": []
    },
    {
      "label": "Run Tests",
      "type": "shell",
      "command": "poetry run pytest tests/ -v",
      "group": "test",
      "presentation": {
        "reveal": "always",
        "panel": "shared"
      }
    },
    {
      "label": "Open MLflow UI",
      "type": "shell",
      "command": "open https://mlflow.mdpg.ai",
      "problemMatcher": []
    },
    {
      "label": "Format Code",
      "type": "shell",
      "command": "poetry run black src/ tests/ && poetry run isort src/ tests/",
      "problemMatcher": []
    }
  ]
}
```

### 4. 추천 확장 프로그램

`.vscode/extensions.json` 생성:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-toolsai.jupyter",
    "charliermarsh.ruff",
    "ms-azuretools.vscode-docker",
    "redhat.vscode-yaml",
    "hashicorp.terraform",
    "github.copilot"  // 선택
  ]
}
```

---

## Python 환경 구성

### 1. Poetry 가상환경 생성

```bash
# Poetry 설치 확인
poetry --version

# 가상환경 생성 및 의존성 설치
poetry install

# 가상환경 활성화
poetry shell

# 또는 VSCode에서 자동 선택
# Cmd+Shift+P → "Python: Select Interpreter" → .venv/bin/python
```

### 2. 환경 변수 테스트

```bash
# 터미널에서 환경 변수 확인
source .env
echo $MLFLOW_TRACKING_URI
# https://mlflow.mdpg.ai

echo $MLFLOW_TRACKING_USERNAME
# your_username
```

### 3. Python에서 환경 변수 확인

```python
# 간단한 테스트 스크립트
python3 <<EOF
import os
from dotenv import load_dotenv

load_dotenv()

print("MLflow Tracking URI:", os.getenv("MLFLOW_TRACKING_URI"))
print("MLflow Username:", os.getenv("MLFLOW_TRACKING_USERNAME"))
print("AWS Region:", os.getenv("AWS_REGION"))
EOF
```

---

## MLflow 연결 테스트

### 1. 기본 연결 테스트

```python
# test_mlflow_connection.py
import os
import mlflow
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

print("🔍 Testing MLflow Remote Connection...")
print(f"MLflow URI: {os.getenv('MLFLOW_TRACKING_URI')}")
print(f"Username: {os.getenv('MLFLOW_TRACKING_USERNAME')}")

try:
    # Experiment 생성
    mlflow.set_experiment("connection-test")

    # Run 시작
    with mlflow.start_run(run_name="vscode-connection-test"):
        # 파라미터 로깅
        mlflow.log_param("environment", "vscode-local")
        mlflow.log_param("user", os.getenv("MLFLOW_TRACKING_USERNAME"))

        # 메트릭 로깅
        mlflow.log_metric("test_metric", 0.95)

        # 아티팩트 로깅 (S3 업로드 테스트)
        with open("test_artifact.txt", "w") as f:
            f.write("This is a test artifact from VSCode")
        mlflow.log_artifact("test_artifact.txt")

        run_id = mlflow.active_run().info.run_id
        print(f"\n✅ Success! Run ID: {run_id}")
        print(f"🌐 View in MLflow UI: {os.getenv('MLFLOW_TRACKING_URI')}/#/experiments/0/runs/{run_id}")

except Exception as e:
    print(f"\n❌ Connection failed: {e}")
    import traceback
    traceback.print_exc()
```

실행:

```bash
poetry run python test_mlflow_connection.py
```

### 2. 전체 학습 파이프라인 테스트

```bash
# 짧은 학습 실행 (1 epoch)
export EPOCHS=1
poetry run python -m src.training.train

# MLflow UI에서 결과 확인
open https://mlflow.mdpg.ai
```

### 3. S3 아티팩트 확인

```bash
# 학습 후 S3에 업로드된 아티팩트 확인
aws s3 ls s3://mdpg-mlops-mlflow-artifacts/ --recursive | grep $(date +%Y-%m-%d)

# 특정 Run ID의 아티팩트 확인
RUN_ID="abc123"
aws s3 ls s3://mdpg-mlops-mlflow-artifacts/0/${RUN_ID}/artifacts/
```

---

## 개발 워크플로우

### 일반적인 개발 사이클

#### 1. 코드 수정

```bash
# 새 브랜치 생성
git checkout -b feature/improve-accuracy

# 코드 수정 (예: src/training/train.py)
code src/training/train.py
```

#### 2. 로컬 테스트

```bash
# 유닛 테스트 실행
poetry run pytest tests/test_training.py -v

# 빠른 학습 테스트 (1 epoch)
EPOCHS=1 poetry run python -m src.training.train
```

#### 3. 실험 실행

```bash
# 실제 학습 실행 (원격 MLflow에 기록)
poetry run python -m src.training.train

# 또는 VSCode에서:
# Cmd+Shift+B → "Train Model (Remote MLflow)"
```

#### 4. MLflow UI에서 결과 확인

```bash
# MLflow UI 열기
open https://mlflow.mdpg.ai

# 또는 VSCode Task 사용:
# Cmd+Shift+P → "Tasks: Run Task" → "Open MLflow UI"
```

#### 5. 코드 커밋

```bash
# 코드 포맷팅
poetry run black src/ tests/
poetry run isort src/ tests/

# 린팅 확인
poetry run flake8 src/ tests/

# 테스트 전체 실행
poetry run pytest tests/ -v

# 커밋
git add .
git commit -m "feat: improve model accuracy to 92%"
git push origin feature/improve-accuracy
```

### 팀 협업 워크플로우

#### 시나리오 1: 다른 팀원의 실험 재현

```python
# 1. MLflow UI에서 Run ID 확인
# https://mlflow.mdpg.ai/#/experiments/1/runs/abc123

# 2. 해당 Run의 파라미터 확인
import mlflow

run = mlflow.get_run("abc123")
params = run.data.params

print("Parameters used:")
for key, value in params.items():
    print(f"  {key}: {value}")

# 3. 동일한 파라미터로 재실행
# .env 파일에 파라미터 설정 또는 CLI로 전달
```

#### 시나리오 2: 최고 성능 모델 로드

```python
import mlflow.pytorch

# 1. Production 모델 로드
model_uri = "models:/vision-classifier/Production"
model = mlflow.pytorch.load_model(model_uri)

# 2. 로컬에서 평가
from src.training.evaluate import evaluate_model
results = evaluate_model(model, test_loader)

print(f"Production model accuracy: {results['test_accuracy']:.2f}%")
```

#### 시나리오 3: 여러 실험 비교

```python
import mlflow
import pandas as pd

# 1. Experiment의 모든 Runs 가져오기
experiment_id = "1"  # vision-model-training
runs = mlflow.search_runs(experiment_ids=[experiment_id])

# 2. Top 5 정확도 모델 확인
top_runs = runs.nlargest(5, "metrics.test_accuracy")
print(top_runs[["run_id", "metrics.test_accuracy", "params.learning_rate", "params.batch_size"]])

# 3. CSV로 저장
top_runs.to_csv("top_models.csv", index=False)
```

---

## 문제 해결

### 문제 1: 인증 실패 (401 Unauthorized)

**증상**:
```
mlflow.exceptions.RestException: UNAUTHORIZED: User is not authenticated
```

**해결**:
```bash
# 1. 환경 변수 확인
echo $MLFLOW_TRACKING_USERNAME
echo $MLFLOW_TRACKING_PASSWORD

# 2. .env 파일 확인
cat .env | grep MLFLOW_TRACKING

# 3. 수동으로 설정
export MLFLOW_TRACKING_USERNAME=your_username
export MLFLOW_TRACKING_PASSWORD=your_password

# 4. Python에서 확인
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('MLFLOW_TRACKING_USERNAME'))"
```

### 문제 2: S3 업로드 실패 (AccessDenied)

**증상**:
```
botocore.exceptions.ClientError: An error occurred (AccessDenied) when calling the PutObject operation
```

**해결**:
```bash
# 1. AWS credentials 확인
aws sts get-caller-identity

# 2. S3 버킷 접근 테스트
aws s3 ls s3://mdpg-mlops-mlflow-artifacts/

# 3. IAM 권한 확인 (필요 권한: s3:PutObject, s3:GetObject, s3:ListBucket)

# 4. AWS_REGION 설정 확인
echo $AWS_REGION
# us-west-2
```

### 문제 3: 네트워크 연결 실패

**증상**:
```
requests.exceptions.ConnectionError: Failed to establish a new connection
```

**해결**:
```bash
# 1. MLflow 서버 상태 확인
curl -I https://mlflow.mdpg.ai/health

# 2. DNS 확인
nslookup mlflow.mdpg.ai

# 3. VPN 연결 확인 (회사 VPN 사용 시)

# 4. 방화벽/Proxy 확인
```

### 문제 4: 모델 아티팩트가 S3에 업로드되지 않음

**증상**: MLflow UI에 실험은 보이지만 아티팩트가 없음

**해결**:
```python
# 1. 명시적으로 아티팩트 로깅
import mlflow
import torch

model = ...  # Your model

with mlflow.start_run():
    # 파라미터 로깅
    mlflow.log_params({"lr": 0.001, "epochs": 20})

    # 모델 저장 (반드시!)
    mlflow.pytorch.log_model(model, "model")

    # 또는 수동 저장
    torch.save(model.state_dict(), "model.pth")
    mlflow.log_artifact("model.pth")

# 2. S3 경로 확인
run = mlflow.get_run(mlflow.active_run().info.run_id)
print(f"Artifact URI: {run.info.artifact_uri}")
# s3://mdpg-mlops-mlflow-artifacts/1/abc123/artifacts
```

### 문제 5: VSCode Python interpreter를 찾지 못함

**증상**: "No Python interpreter is selected"

**해결**:
```bash
# 1. Poetry 가상환경 경로 확인
poetry env info --path
# /Users/username/Library/Caches/pypoetry/virtualenvs/mlflow-study-abc123-py3.9

# 2. VSCode에서 Interpreter 선택
# Cmd+Shift+P → "Python: Select Interpreter"
# .venv/bin/python 또는 위 경로 선택

# 3. settings.json 확인
cat .vscode/settings.json | grep defaultInterpreterPath
```

---

## 참고 자료

- [MLflow Authentication](https://mlflow.org/docs/latest/auth/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [AWS SDK for Python (Boto3)](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [VSCode Python Extension](https://code.visualstudio.com/docs/python/python-tutorial)

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
