# MLflow Vision Training - 빠른 시작 가이드

이 가이드는 5분 안에 시스템을 시작하고 첫 학습을 실행하는 방법을 설명합니다.

## 전제 조건

- MacBook M2 Air (또는 Apple Silicon Mac)
- Docker Desktop 실행 중
- Poetry 설치됨

## 1단계: 의존성 설치 (1분)

```bash
# Poetry로 의존성 설치
make install

# 환경 변수 설정
make setup
```

## 2단계: MLflow 인프라 시작 (2분)

```bash
# Docker Compose로 인프라 시작
make start

# 상태 확인 (모두 healthy 확인)
make status
```

**접속 정보**:
- MLflow UI: http://localhost:5001
- MinIO Console: http://localhost:9001 (minio / minio123)

## 3단계: 첫 모델 학습 실행 (2분)

기본 설정으로 테스트 학습 실행:

```bash
# EPOCHS를 5로 줄여서 빠른 테스트
echo "EPOCHS=5" >> .env

# 학습 시작
make train
```

학습이 시작되면 다음이 자동으로 진행됩니다:
1. CIFAR-10 데이터셋 다운로드 (최초 1회)
2. MobileNetV3-Small 모델 초기화
3. MLflow 실험 생성 및 추적 시작
4. 5 에포크 학습
5. 메트릭 및 모델 자동 로깅

## 4단계: 결과 확인

### MLflow UI 열기
```bash
make mlflow-ui
```

또는 브라우저에서 직접: http://localhost:5001

### 확인할 내용
- **Experiments** 탭: `vision-model-training` 실험 확인
- **Runs** 목록: 방금 실행한 run 클릭
- **Metrics** 탭:
  - train_accuracy, val_accuracy 그래프
  - train_loss, val_loss 그래프
- **Parameters** 탭: 모든 하이퍼파라미터 기록
- **Artifacts** 탭:
  - 저장된 PyTorch 모델
  - 체크포인트 파일

## 5단계: 테스트 실행 (선택)

코드가 정상 동작하는지 확인:

```bash
# 빠른 테스트만 실행 (~8초)
make test-fast

# 전체 테스트 실행 (커버리지 포함, ~1분)
make test
```

**테스트 구성**:
- 총 52개 테스트 (모두 자동화)
- 코드 커버리지 56%
- MLflow 통합 테스트 포함 (격리된 환경에서 실행)

**중요**: 테스트는 프로덕션 MLflow 서버를 오염시키지 않습니다. 모든 테스트 데이터는 임시 디렉토리에 저장되고 자동으로 삭제됩니다.

더 자세한 내용은 [TESTING.md](TESTING.md)를 참고하세요.

## 다음 단계

### 본격적인 학습 실행

`.env` 파일 수정:
```bash
EPOCHS=20
BATCH_SIZE=64
LEARNING_RATE=0.001
```

```bash
make train
```

### 모델 평가

```bash
# MLflow UI에서 Run ID 복사
make evaluate RUN_ID=<your_run_id>

# evaluation_results/ 디렉토리에 시각화 저장됨
```

### Docker에서 학습

```bash
make train-docker
```

### 다른 모델 시도

`.env` 파일에서 변경:
```bash
# 옵션: mobilenet_v3_small, mobilenet_v3_large, resnet18
MODEL_NAME=resnet18
```

## 일반적인 명령어

```bash
make help           # 모든 명령어 보기
make status         # 인프라 상태 확인
make logs           # 인프라 로그 보기
make stop           # 인프라 중지
make restart        # 인프라 재시작
make clean          # 임시 파일 정리
make test           # 전체 테스트 실행
make test-fast      # 빠른 테스트만 실행
```

## 문제 해결

### 포트 충돌
MLflow 포트(5001)가 사용 중인 경우:
```bash
# docker-compose.yml에서 포트 변경
ports:
  - "5002:5000"  # 5001 → 5002

# .env 파일도 동일하게 변경
MLFLOW_TRACKING_URI=http://localhost:5002
```

### M2 GPU 사용 안 됨
```bash
# .env 파일 확인
DEVICE=mps  # M2 GPU용

# PyTorch MPS 지원 확인
python -c "import torch; print(torch.backends.mps.is_available())"
```

### 메모리 부족
```bash
# .env 파일에서 배치 크기 감소
BATCH_SIZE=32
```

## 프로덕션 체크리스트

로컬 테스트 완료 후, 다음 단계로 확장:

- [ ] 하이퍼파라미터 튜닝 실험 (MLflow에 기록)
- [ ] 최적 모델 선정 (MLflow UI에서 비교)
- [ ] 자동화 테스트 실행 및 검증 (`make test`)
- [ ] Dockerfile 빌드 및 테스트
- [ ] Terraform으로 인프라 관리 연습
- [ ] Kubernetes YAML 작성 계획
- [ ] Airflow DAG 설계 초안
- [ ] AI 엔지니어와 협업 프로세스 수립

## 학습 커브

**1일차**: 로컬 환경에서 기본 학습 실행 ✅ (지금!)
**2-3일차**: 하이퍼파라미터 실험 및 MLflow 숙달
**4-5일차**: Docker 컨테이너화 및 테스트
**1-2주차**: Terraform + Kubernetes 학습
**2-4주차**: Airflow 통합 및 프로덕션 파이프라인 구축

## 추가 리소스

- [상세 README](README.md)
- [테스트 가이드](TESTING.md) - 전체 테스트 실행 방법 및 MLflow 격리 메커니즘
- [MLflow 문서](https://mlflow.org/docs/latest/index.html)
- [Terraform Docker Provider](https://registry.terraform.io/providers/kreuzwerker/docker/latest/docs)

---

**질문이나 문제가 있으신가요?**
README.md의 "문제 해결" 섹션을 참조하거나 프로젝트 이슈를 확인하세요.
