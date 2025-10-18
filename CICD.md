# CI/CD 파이프라인 가이드

이 문서는 MLflow Vision Training System의 CI/CD 파이프라인 설정 및 사용 방법을 설명합니다.

## 개요

GitHub Actions를 사용한 완전 자동화 CI/CD 파이프라인:
- ✅ 자동화된 테스트 (52개, 56% 커버리지)
- ✅ 코드 품질 검사 (Black, isort, flake8, mypy)
- ✅ 보안 스캔 (Trivy, Bandit)
- ✅ Docker 이미지 자동 빌드 및 푸시
- ✅ 릴리스 자동화

---

## 워크플로우 구조

### 1. Tests ([.github/workflows/test.yml](.github/workflows/test.yml))

**트리거**: PR 및 main/develop 브랜치 푸시

**Job 1: test** - Docker 기반 테스트
```yaml
- Build Docker image (development stage)
- Run pytest in container
  - Fast tests only (-m "not slow")
  - Coverage report (HTML + XML)
- Upload coverage to Codecov
- Upload HTML report as artifact
```

**Job 2: lint** - 코드 품질 검사
```yaml
- Setup Python 3.11 + Poetry
- Run Black (format check)
- Run isort (import order)
- Run flake8 (linting)
- Run mypy (type checking)
```

**Job 3: security** - 보안 스캔
```yaml
- Run Trivy (filesystem scan)
- Upload SARIF to GitHub Security
```

### 2. Docker Build ([.github/workflows/docker.yml](.github/workflows/docker.yml))

**트리거**: main 브랜치 푸시, 태그 푸시, 수동 실행

**Job 1: build-and-push** - 학습 이미지
```yaml
Matrix:
  - target: production
  - target: development

For each target:
  - Build Docker image
  - Push to ghcr.io
  - Run Trivy security scan
  - Upload scan results
```

**Job 2: build-mlflow** - MLflow 서버 이미지
```yaml
- Build Dockerfile.mlflow
- Push to ghcr.io/[user]/mlflow-study-mlflow
```

### 3. Release ([.github/workflows/release.yml](.github/workflows/release.yml))

**트리거**: 태그 푸시 (v*.*.*)

**Job 1: create-release**
```yaml
- Generate changelog from commits
- Create GitHub Release
- Include Docker image links
```

**Job 2: run-e2e-tests**
```yaml
- Build test image
- Run ALL tests (including slow)
- Upload test results as artifact
```

---

## 사용 방법

### 로컬 개발

#### 1. Pre-commit Hook 설정 (권장)

```bash
# 설치
make pre-commit-install

# 수동 실행
make pre-commit-run
```

**Pre-commit이 자동으로 검사하는 항목**:
- ✅ Black 포맷팅
- ✅ isort import 정렬
- ✅ flake8 린팅
- ✅ mypy 타입 체킹
- ✅ Bandit 보안 검사
- ✅ Trailing whitespace, EOF fixer
- ✅ YAML 문법 검사
- ✅ Dockerfile 린팅 (hadolint)

#### 2. CI/CD와 동일한 테스트 로컬 실행

```bash
# Docker 기반 테스트 (CI/CD와 동일)
make test-docker

# 빠른 테스트만
make test-docker-fast

# 코드 품질 검사
make format  # 자동 포맷팅
make lint    # 수동 검사
```

### Pull Request 워크플로우

1. **브랜치 생성**
```bash
git checkout -b feature/new-feature
```

2. **코드 작성 및 커밋**
```bash
# Pre-commit이 자동으로 검사
git add .
git commit -m "feat: add new feature"
```

3. **PR 생성**
```bash
git push origin feature/new-feature
# GitHub에서 PR 생성
```

4. **자동 검사 대기**
- ✅ Tests (약 5분)
- ✅ Lint (약 2분)
- ✅ Security (약 1분)

5. **문제 발생 시**
```bash
# 로컬에서 수정
make format  # 포맷 이슈
make lint    # 린팅 이슈
make test-docker  # 테스트 실패

# 재커밋
git add .
git commit -m "fix: resolve CI issues"
git push
```

### Release 워크플로우

#### 1. Semantic Versioning

```
v{MAJOR}.{MINOR}.{PATCH}

MAJOR: 호환성 없는 API 변경
MINOR: 기능 추가 (하위 호환)
PATCH: 버그 수정
```

#### 2. 릴리스 생성

```bash
# 1. 버전 업데이트
# pyproject.toml의 version 수정

# 2. 변경사항 커밋
git add pyproject.toml
git commit -m "chore: bump version to v1.2.3"

# 3. 태그 생성 및 푸시
git tag v1.2.3
git push origin main --tags
```

#### 3. 자동 실행 항목

- ✅ GitHub Release 자동 생성
- ✅ Changelog 자동 생성
- ✅ Docker 이미지 빌드 (3개)
  - `ghcr.io/[user]/mlflow-study:v1.2.3`
  - `ghcr.io/[user]/mlflow-study:development-v1.2.3`
  - `ghcr.io/[user]/mlflow-study-mlflow:v1.2.3`
- ✅ 전체 E2E 테스트 실행
- ✅ 테스트 결과 아티팩트 업로드

---

## 설정

### GitHub Secrets

다음 secrets를 GitHub 리포지토리에 추가하세요:

**필수**:
- `CODECOV_TOKEN`: Codecov 통합 (선택사항이지만 권장)

**자동 제공** (GitHub Actions):
- `GITHUB_TOKEN`: Docker 이미지 푸시용 (자동 생성)

### Codecov 설정

1. [codecov.io](https://codecov.io) 방문
2. GitHub 계정으로 로그인
3. 리포지토리 추가
4. Token 복사
5. GitHub Settings → Secrets → Add `CODECOV_TOKEN`

### Docker Registry 권한

GitHub Container Registry (ghcr.io) 사용:
- ✅ 자동으로 `GITHUB_TOKEN` 사용
- ✅ 별도 설정 불필요
- ✅ 리포지토리 Packages 탭에서 확인

---

## 트러블슈팅

### 1. Pre-commit 실패

**문제**: pre-commit hook이 커밋을 차단
```bash
black....................................................................Failed
```

**해결**:
```bash
# 자동 수정
make format

# 재커밋
git add .
git commit -m "fix: format code"
```

### 2. Docker 빌드 실패

**문제**: GitHub Actions에서 Docker 빌드 실패

**해결**:
```bash
# 로컬에서 빌드 테스트
docker build -t test:local --target development .

# 문제 확인 후 수정
git add Dockerfile
git commit -m "fix: Docker build issue"
git push
```

### 3. 테스트 타임아웃

**문제**: GitHub Actions에서 테스트 시간 초과

**해결**:
```bash
# 느린 테스트 마킹
@pytest.mark.slow
def test_long_running():
    pass

# 기본적으로 slow 테스트 제외됨
make test-docker-fast
```

### 4. Coverage 낮음

**문제**: PR이 커버리지 요구사항 미달

**해결**:
```bash
# 커버리지 확인
make test-coverage
open htmlcov/index.html

# 테스트 추가
# tests/test_*.py 작성

# 재테스트
make test-docker
```

---

## 모범 사례

### 커밋 메시지 규칙

Conventional Commits 사용:

```
<type>(<scope>): <subject>

Type:
  - feat: 새 기능
  - fix: 버그 수정
  - docs: 문서만 수정
  - style: 코드 포맷팅
  - refactor: 리팩토링
  - test: 테스트 추가/수정
  - chore: 빌드/설정 변경

Example:
  feat(model): add ResNet50 support
  fix(training): resolve CUDA OOM error
  docs(readme): update installation guide
```

### 브랜치 전략

```
main: 프로덕션 릴리스
  ↑
develop: 개발 통합
  ↑
feature/*: 기능 개발
hotfix/*: 긴급 수정
```

### PR 체크리스트

- [ ] Pre-commit hooks 통과
- [ ] 테스트 추가/업데이트
- [ ] 문서 업데이트 (필요시)
- [ ] 변경사항 설명 작성
- [ ] CI/CD 전체 통과

---

## 성능 최적화

### GitHub Actions 캐싱

**Docker 빌드 캐시**:
```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

**Poetry 캐시**:
```yaml
- uses: actions/setup-python@v5
  with:
    cache: 'pip'
```

### 병렬 실행

**Matrix 전략**:
```yaml
strategy:
  matrix:
    target: [production, development]
```

**Job 병렬화**:
- test, lint, security는 독립적으로 동시 실행
- 총 실행 시간 단축

---

## 비용 관리

### GitHub Actions 사용량

- ✅ Public 리포지토리: 무료 무제한
- ⚠️ Private 리포지토리: 월 2,000분 무료

### 최적화 팁

1. **캐싱 활용**: Docker layer cache, Poetry cache
2. **조건부 실행**: `paths` 필터 사용
3. **빠른 실패**: `fail-fast: true`

```yaml
on:
  push:
    paths:
      - 'src/**'
      - 'tests/**'
      - 'pyproject.toml'
```

---

## 다음 단계

### Phase 5: 고급 CI/CD

- [ ] Multi-stage deployment (dev/staging/prod)
- [ ] Canary deployment
- [ ] Automated rollback
- [ ] Performance benchmarking in CI
- [ ] Slack/Discord 알림 통합

### Phase 6: Kubernetes CI/CD

- [ ] ArgoCD/FluxCD GitOps
- [ ] Helm chart 테스트
- [ ] K8s manifest validation

---

**Last Updated**: 2025-10-18
**Version**: 1.0
