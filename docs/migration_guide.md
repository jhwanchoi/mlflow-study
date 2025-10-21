# 로컬 → EKS 마이그레이션 가이드

**작성일**: 2025-10-21
**버전**: 1.0

---

## 개요

로컬 Docker Compose 환경에서 AWS EKS 환경으로 안전하게 마이그레이션하는 단계별 가이드입니다.

---

## 마이그레이션 체크리스트

### Phase 1: 준비 (1일)

- [ ] EKS 인프라 배포 완료 확인
- [ ] MLflow 서버 정상 작동 확인
- [ ] 로컬 실험 데이터 백업
- [ ] 팀원 계정 생성 완료

### Phase 2: 데이터 마이그레이션 (1일)

- [ ] PostgreSQL 데이터 마이그레이션
- [ ] MinIO → S3 아티팩트 마이그레이션
- [ ] 마이그레이션 검증

### Phase 3: 클라이언트 전환 (1일)

- [ ] 개발 환경 설정 변경
- [ ] 연결 테스트
- [ ] 기존 학습 코드 재실행

---

## 데이터 마이그레이션

### 1. PostgreSQL 마이그레이션

```bash
# 1. 로컬 PostgreSQL 데이터 덤프
docker-compose exec postgres pg_dump -U mlflow mlflow > mlflow_local.sql

# 2. RDS로 복원
cd terraform/aws-eks
RDS_ENDPOINT=$(terraform output -raw rds_endpoint)

psql -h $RDS_ENDPOINT -U mlflow -d mlflow < mlflow_local.sql

# 3. 데이터 확인
psql -h $RDS_ENDPOINT -U mlflow -d mlflow -c "SELECT COUNT(*) FROM experiments;"
```

### 2. MinIO → S3 마이그레이션

```bash
# 1. S3 버킷 이름 가져오기
S3_BUCKET=$(cd terraform/aws-eks && terraform output -raw s3_bucket_name)

# 2. MinIO 데이터 동기화
docker-compose exec minio mc alias set local http://localhost:9000 minio minio123

docker-compose exec minio mc mirror local/mlflow s3://$S3_BUCKET/

# 3. 검증
aws s3 ls s3://$S3_BUCKET/ --recursive | wc -l
```

---

## 클라이언트 전환

### 1. 환경 변수 변경

```bash
# .env 백업
cp .env .env.local.backup

# 새 .env 생성 (원격)
cat > .env <<EOF
MLFLOW_TRACKING_URI=https://mlflow.mdpg.ai
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
AWS_REGION=us-west-2

# Training Configuration (동일)
EXPERIMENT_NAME=vision-model-training
MODEL_NAME=mobilenet_v3_small
BATCH_SIZE=64
LEARNING_RATE=0.001
EPOCHS=20
EOF
```

### 2. 연결 테스트

```bash
# MLflow 연결 테스트
python3 <<EOF
import mlflow
import os

mlflow.set_experiment("migration-test")
with mlflow.start_run():
    mlflow.log_param("migration", "local-to-eks")
    mlflow.log_metric("success", 1.0)

print("✅ 마이그레이션 성공!")
EOF
```

### 3. 기존 코드 재실행

```bash
# 기존 학습 코드 그대로 실행
poetry run python -m src.training.train

# MLflow UI에서 확인
open https://mlflow.mdpg.ai
```

---

## 롤백 계획

문제 발생 시 로컬 환경으로 즉시 복귀:

```bash
# 1. 로컬 환경 복원
cp .env.local.backup .env

# 2. 로컬 MLflow 재시작
make start

# 3. 연결 확인
curl http://localhost:5001/health
```

---

## 문제 해결

### 문제: 기존 실험이 보이지 않음

**원인**: 데이터 마이그레이션 미완료

**해결**:
```bash
# PostgreSQL 마이그레이션 재실행
psql -h $RDS_ENDPOINT -U mlflow -d mlflow < mlflow_local.sql
```

### 문제: 아티팩트 다운로드 실패

**원인**: S3 마이그레이션 미완료 또는 권한 부족

**해결**:
```bash
# S3 동기화 재실행
docker-compose exec minio mc mirror local/mlflow s3://$S3_BUCKET/

# AWS credentials 확인
aws s3 ls s3://$S3_BUCKET/
```

---

## 마이그레이션 완료 확인

```bash
# 체크리스트
✅ 로컬 실험 데이터가 원격 MLflow에서 보임
✅ 새 실험이 S3에 저장됨
✅ 팀원들이 모두 접속 가능
✅ 기존 학습 코드가 수정 없이 동작
```

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
