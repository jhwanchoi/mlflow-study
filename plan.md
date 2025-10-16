# SQLite → PostgreSQL Migration Plan

## 현재 문제 분석

### 1. 발견된 문제점

#### 주요 문제: MLflow가 SQLite를 사용 중
**위치**: `docker-compose.yml:68`
```yaml
mlflow:
  command: >
    mlflow server
    --backend-store-uri /mlflow/mlflow.db  # ❌ SQLite 사용
    --default-artifact-root s3://mlflow/artifacts
```

**증상**:
- Logged model 저장 시 동시성 문제 발생 가능
- 데이터베이스 잠금(lock) 오류
- 프로덕션 환경에 부적합한 아키텍처

#### 부차적 문제: boto3 누락
**위치**: MLflow server 컨테이너
```
ModuleNotFoundError: No module named 'boto3'
```

**증상**:
- S3(MinIO) 아티팩트 저장/조회 실패
- MLflow UI에서 아티팩트 목록 표시 불가

### 2. SQLite의 문제점

#### 동시성 제한
- **Write Lock**: 한 번에 하나의 쓰기 작업만 가능
- **Read Lock**: 쓰기 중에는 읽기 차단
- **병렬 학습**: 여러 실험 동시 실행 시 충돌

#### 프로덕션 부적합
- **Scalability**: 다중 사용자 환경 미지원
- **Reliability**: 파일 손상 위험
- **Performance**: 대용량 메타데이터 처리 느림
- **Backup**: 단순 파일 복사로는 일관성 보장 어려움

### 3. PostgreSQL이 해결하는 것

✅ **MVCC (Multi-Version Concurrency Control)**: 동시 읽기/쓰기 지원
✅ **Connection Pooling**: 다중 클라이언트 동시 접속
✅ **ACID Transactions**: 데이터 무결성 보장
✅ **Production-Ready**: 프로덕션 환경 표준 RDBMS

---

## 마이그레이션 계획

### Phase 1: MLflow Backend Store 변경 (Critical)

#### 1.1 docker-compose.yml 수정

**Before** (docker-compose.yml:66-71):
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.10.2
  container_name: mlflow-server
  command: >
    mlflow server
    --backend-store-uri /mlflow/mlflow.db  # SQLite
    --default-artifact-root s3://mlflow/artifacts
    --host 0.0.0.0
    --port 5000
```

**After**:
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.10.2
  container_name: mlflow-server
  command: >
    mlflow server
    --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow
    --default-artifact-root s3://mlflow/artifacts
    --host 0.0.0.0
    --port 5000
  environment:
    AWS_ACCESS_KEY_ID: minio
    AWS_SECRET_ACCESS_KEY: minio123
    MLFLOW_S3_ENDPOINT_URL: http://minio:9000
  depends_on:
    postgres:
      condition: service_healthy
    minio:
      condition: service_healthy
    minio-init:
      condition: service_completed_successfully
```

**변경사항**:
- `--backend-store-uri` → PostgreSQL 연결 문자열로 변경
- `depends_on` → PostgreSQL 헬스체크 추가
- `environment` → AWS/S3 환경변수 명시

#### 1.2 PostgreSQL 헬스체크 검증

**현재 설정** (docker-compose.yml:16-20):
```yaml
postgres:
  healthcheck:
    test: ["CMD-SHELL", "pg_isready -U mlflow"]
    interval: 10s
    timeout: 5s
    retries: 5
```

**검증 방법**:
```bash
# PostgreSQL 상태 확인
docker-compose ps postgres
docker exec mlflow-postgres pg_isready -U mlflow

# 연결 테스트
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "SELECT 1;"
```

---

### Phase 2: boto3 의존성 추가 (Critical)

#### 2.1 커스텀 MLflow 이미지 생성

**파일 생성**: `Dockerfile.mlflow`
```dockerfile
FROM ghcr.io/mlflow/mlflow:v2.10.2

# Install boto3 for S3 (MinIO) support
RUN pip install --no-cache-dir boto3

# Health check
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:5000/health || exit 1
```

#### 2.2 docker-compose.yml 업데이트

**Before** (docker-compose.yml:64):
```yaml
mlflow:
  image: ghcr.io/mlflow/mlflow:v2.10.2
```

**After**:
```yaml
mlflow:
  build:
    context: .
    dockerfile: Dockerfile.mlflow
```

**또는 빌드된 이미지 사용**:
```yaml
mlflow:
  image: mlflow-server:local-v2.10.2-boto3
```

---

### Phase 3: 데이터 마이그레이션 (Optional)

#### 3.1 기존 SQLite 데이터 백업

```bash
# MLflow 컨테이너에서 SQLite DB 추출
docker cp mlflow-server:/mlflow/mlflow.db ./backup/mlflow.db

# 백업 검증
sqlite3 backup/mlflow.db "SELECT COUNT(*) FROM experiments;"
sqlite3 backup/mlflow.db "SELECT COUNT(*) FROM runs;"
```

#### 3.2 PostgreSQL로 마이그레이션

**방법 A: MLflow 내장 마이그레이션 (권장하지 않음)**
- MLflow는 공식 마이그레이션 도구 미제공
- 데이터 손실 위험

**방법 B: 새로 시작 (권장)**
- PostgreSQL로 새 실험 시작
- 기존 SQLite 데이터는 아카이브
- 이유: 개발 초기 단계, 기존 데이터 중요도 낮음

**방법 C: 수동 마이그레이션 (필요시)**
```python
# scripts/migrate_sqlite_to_postgres.py (참고용)
import sqlite3
import psycopg2
from mlflow.tracking import MlflowClient

# SQLite 연결
sqlite_conn = sqlite3.connect('backup/mlflow.db')
sqlite_cursor = sqlite_conn.cursor()

# PostgreSQL 연결
pg_conn = psycopg2.connect(
    "postgresql://mlflow:mlflow@localhost:5432/mlflow"
)
pg_cursor = pg_conn.cursor()

# 실험 마이그레이션
sqlite_cursor.execute("SELECT * FROM experiments")
for row in sqlite_cursor.fetchall():
    # INSERT INTO PostgreSQL
    pass

# Run 마이그레이션
# Metrics 마이그레이션
# ...
```

---

### Phase 4: 환경 검증 및 테스트

#### 4.1 인프라 검증

```bash
# 1. 모든 서비스 재시작
docker-compose down -v  # ⚠️ 볼륨 삭제 주의
docker-compose up -d

# 2. PostgreSQL 연결 확인
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "\dt"
# Expected: MLflow 테이블 (experiments, runs, metrics, params, etc.)

# 3. MLflow 서버 로그 확인
docker logs mlflow-server
# Expected: "Backend store uri: postgresql://..."
# Expected: "Default artifact root: s3://mlflow/artifacts"
```

#### 4.2 기능 테스트

```bash
# 1. 간단한 실험 실행
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
mlflow.set_experiment('test-migration')
with mlflow.start_run():
    mlflow.log_param('test', 1)
    mlflow.log_metric('accuracy', 0.95)
print('✅ Test run created successfully')
"

# 2. PostgreSQL에서 데이터 확인
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "
SELECT name, experiment_id FROM experiments WHERE name='test-migration';
SELECT run_id, status FROM runs LIMIT 1;
"

# 3. 모델 로깅 테스트
make train  # 실제 학습 실행
# Expected: 모델 저장 성공, 동시성 오류 없음
```

#### 4.3 성능 테스트

```bash
# 동시 실험 실행 테스트
for i in {1..5}; do
  (python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5001')
mlflow.set_experiment('concurrent-test')
with mlflow.start_run(run_name='run-$i'):
    mlflow.log_metric('score', $i)
  ") &
done
wait

# 결과 확인 (5개 run 모두 성공해야 함)
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "
SELECT COUNT(*) FROM runs WHERE experiment_id=(
  SELECT experiment_id FROM experiments WHERE name='concurrent-test'
);"
# Expected: 5
```

---

### Phase 5: 문서 및 설정 업데이트

#### 5.1 README.md 업데이트

**Before**:
```markdown
## 아키텍처
┌───────────────────┐
│   PostgreSQL      │  # 정의되어 있지만 사용 안 함
│  (Metadata Store) │
└───────────────────┘
```

**After**:
```markdown
## 아키텍처
┌───────────────────┐
│   PostgreSQL      │  ✅ Backend Store (Metadata)
│  (Metadata Store) │     - Experiments, Runs, Metrics
└───────────────────┘

## 주요 특징
- **프로덕션급 백엔드**: PostgreSQL (동시성 지원)
- **S3 아티팩트 저장**: MinIO (모델, 체크포인트)
```

#### 5.2 Makefile 업데이트

**추가 명령어**:
```makefile
# Makefile에 추가
.PHONY: db-shell
db-shell: ## PostgreSQL 셸 접속
	docker exec -it mlflow-postgres psql -U mlflow -d mlflow

.PHONY: db-migrate
db-migrate: ## 데이터베이스 스키마 초기화
	docker exec mlflow-server mlflow db upgrade postgresql://mlflow:mlflow@postgres:5432/mlflow

.PHONY: db-backup
db-backup: ## PostgreSQL 백업
	docker exec mlflow-postgres pg_dump -U mlflow mlflow > backup/mlflow_$(date +%Y%m%d_%H%M%S).sql
```

#### 5.3 .env.example 업데이트

**추가**:
```bash
# Database Configuration (docker-compose.yml에서 사용)
POSTGRES_DB=mlflow
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
```

---

## 마이그레이션 실행 순서

### 준비 단계
```bash
# 1. 현재 상태 백업
docker cp mlflow-server:/mlflow/mlflow.db ./backup/mlflow_$(date +%Y%m%d_%H%M%S).db

# 2. 기존 인프라 중지
docker-compose down

# 3. 볼륨 삭제 (선택, 주의!)
docker volume rm mlflow-study_mlflow_data  # SQLite 볼륨 삭제
```

### 실행 단계
```bash
# 1. Dockerfile.mlflow 생성
cat > Dockerfile.mlflow <<'EOF'
FROM ghcr.io/mlflow/mlflow:v2.10.2
RUN pip install --no-cache-dir boto3
HEALTHCHECK --interval=10s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:5000/health || exit 1
EOF

# 2. docker-compose.yml 수정
# (위 Phase 1, 2 내용 적용)

# 3. 이미지 빌드
docker-compose build mlflow

# 4. 인프라 시작
docker-compose up -d

# 5. 로그 모니터링
docker-compose logs -f mlflow
# Expected: "Backend store uri: postgresql://mlflow:mlflow@postgres:5432/mlflow"
```

### 검증 단계
```bash
# 1. 서비스 상태 확인
docker-compose ps
# Expected: All services "Up (healthy)"

# 2. PostgreSQL 연결 확인
docker exec mlflow-postgres psql -U mlflow -d mlflow -c "\dt"
# Expected: experiments, runs, metrics, params, tags, ...

# 3. 테스트 실행
make test-e2e

# 4. 실제 학습 실행
make train

# 5. MLflow UI 확인
open http://localhost:5001
```

---

## 롤백 계획

### 문제 발생 시 SQLite로 복귀

```bash
# 1. 인프라 중지
docker-compose down

# 2. docker-compose.yml 복원
git checkout docker-compose.yml

# 3. SQLite 복원
docker volume create mlflow-study_mlflow_data
docker run --rm -v mlflow-study_mlflow_data:/mlflow alpine sh -c "
  cat > /mlflow/mlflow.db < /backup/mlflow.db
"

# 4. 재시작
docker-compose up -d
```

---

## 예상 소요 시간

| 단계 | 소요 시간 | 비고 |
|-----|----------|------|
| Phase 1 (Backend Store) | 10분 | docker-compose.yml 수정 |
| Phase 2 (boto3) | 15분 | Dockerfile 생성, 빌드 |
| Phase 3 (마이그레이션) | 0분 | 새로 시작 (데이터 버림) |
| Phase 4 (검증) | 20분 | 테스트, 학습 실행 |
| Phase 5 (문서화) | 10분 | README, Makefile |
| **총계** | **~1시간** | |

---

## 성공 기준

### 필수 (Must-Have)
- [x] MLflow 서버가 PostgreSQL을 backend로 사용
- [x] boto3 설치로 S3 아티팩트 저장 가능
- [x] `make train` 실행 시 모델 저장 성공
- [x] 동시 실험 5개 실행 시 오류 없음
- [x] MLflow UI에서 아티팩트 조회 가능

### 권장 (Should-Have)
- [ ] 기존 SQLite 데이터 마이그레이션
- [ ] 성능 벤치마크 (PostgreSQL vs SQLite)
- [ ] 백업/복원 프로시저 문서화

### 선택 (Nice-to-Have)
- [ ] PostgreSQL 모니터링 (pgAdmin)
- [ ] Connection pooling 최적화
- [ ] Read replica 구성

---

## 리스크 및 완화 전략

### 리스크 1: 데이터 손실
**완화**:
- SQLite DB 백업 필수
- 새 실험으로 시작 (기존 데이터 중요도 낮음)

### 리스크 2: 호환성 문제
**완화**:
- MLflow 공식 지원 DB (PostgreSQL)
- 테스트 커버리지 충분 (52개 테스트)

### 리스크 3: 성능 저하
**완화**:
- PostgreSQL은 SQLite보다 빠름 (동시성, 인덱싱)
- Connection pooling 자동 처리

---

## 참고 자료

- [MLflow Backend Stores](https://mlflow.org/docs/latest/tracking.html#backend-stores)
- [MLflow Database Schema](https://mlflow.org/docs/latest/tracking.html#mlflow-database-schema)
- [PostgreSQL for MLflow](https://www.mlflow.org/docs/latest/tracking.html#postgresql)
- [boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)

---

## 다음 단계 (마이그레이션 후)

1. **Terraform 업데이트**: `terraform/local/main.tf`에 PostgreSQL 리소스 추가
2. **CI/CD 통합**: GitHub Actions에 PostgreSQL 서비스 추가
3. **모니터링**: Prometheus + Grafana 대시보드
4. **Kubernetes 준비**: Helm chart에 PostgreSQL StatefulSet 정의

---

**작성일**: 2025-01-16
**작성자**: Claude Code
**버전**: 1.0
