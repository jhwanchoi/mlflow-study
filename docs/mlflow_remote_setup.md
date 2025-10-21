# MLflow Remote Server 설정 가이드

**작성일**: 2025-10-21
**버전**: 1.0
**대상**: Phase 5.2 - MLflow 서버 배포

---

## 목차

1. [개요](#개요)
2. [Helm Chart 작성](#helm-chart-작성)
3. [MLflow Authentication 설정](#mlflow-authentication-설정)
4. [배포 실행](#배포-실행)
5. [검증 및 테스트](#검증-및-테스트)
6. [사용자 관리](#사용자-관리)
7. [문제 해결](#문제-해결)

---

## 개요

### 목표

EKS 클러스터에 중앙화된 MLflow Tracking Server를 배포하여:
- **멀티 유저 인증**: 사용자별 READ/EDIT/MANAGE 권한
- **고가용성**: HPA (Horizontal Pod Autoscaler) 2-5 pods
- **HTTPS 지원**: ALB Ingress + ACM 인증서
- **S3 통합**: IRSA를 통한 안전한 아티팩트 저장

### 아키텍처

```
Client (VSCode/Jupyter)
    ↓ HTTPS (mlflow.mdpg.ai)
ALB Ingress
    ↓
MLflow Service (LoadBalancer)
    ↓
MLflow Pods (2-5 replicas)
    ↓                    ↓
RDS PostgreSQL      S3 Bucket (via IRSA)
```

---

## Helm Chart 작성

### 디렉토리 구조

```
charts/
└── mlflow/
    ├── Chart.yaml
    ├── values.yaml
    ├── values-production.yaml
    ├── templates/
    │   ├── deployment.yaml
    │   ├── service.yaml
    │   ├── ingress.yaml
    │   ├── hpa.yaml
    │   ├── serviceaccount.yaml
    │   └── secret.yaml
    └── README.md
```

### Chart.yaml

```yaml
# charts/mlflow/Chart.yaml
apiVersion: v2
name: mlflow
description: MLflow Tracking Server with Authentication
type: application
version: 1.0.0
appVersion: "2.10.2"

maintainers:
  - name: MLOps Team
    email: mlops@example.com

keywords:
  - mlflow
  - ml-tracking
  - model-registry

home: https://mlflow.org
sources:
  - https://github.com/mlflow/mlflow
```

### values.yaml (기본 설정)

```yaml
# charts/mlflow/values.yaml

# 복제본 수
replicaCount: 2

# Docker 이미지
image:
  repository: ghcr.io/mlflow/mlflow
  tag: "v2.10.2"
  pullPolicy: IfNotPresent

# Service Account (IRSA)
serviceAccount:
  create: true
  name: mlflow
  annotations:
    eks.amazonaws.com/role-arn: ""  # Terraform output에서 가져옴

# MLflow 설정
mlflow:
  # Backend Store (PostgreSQL)
  backendStoreUri: ""  # postgresql://user:pass@host:5432/mlflow

  # Artifact Store (S3)
  artifactRoot: ""  # s3://bucket-name/

  # Authentication
  authentication:
    enabled: true
    # admin 계정은 초기 설정 후 수동으로 생성

  # Server 설정
  host: "0.0.0.0"
  port: 5000
  workers: 4  # Gunicorn workers

# 환경 변수
env:
  - name: AWS_DEFAULT_REGION
    value: "us-west-2"
  # S3 접근은 IRSA로 처리 (AWS credentials 불필요)

# 리소스 제한
resources:
  requests:
    cpu: 500m
    memory: 1Gi
  limits:
    cpu: 2000m
    memory: 4Gi

# Health Checks
livenessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health
    port: 5000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3

# Service
service:
  type: ClusterIP
  port: 80
  targetPort: 5000

# Ingress (ALB)
ingress:
  enabled: true
  className: alb
  annotations:
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/ssl-redirect: '443'
    alb.ingress.kubernetes.io/certificate-arn: ""  # ACM 인증서 ARN
  hosts:
    - host: mlflow.mdpg.ai
      paths:
        - path: /
          pathType: Prefix
  tls:
    - hosts:
        - mlflow.mdpg.ai

# Horizontal Pod Autoscaler
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 5
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

# Node Selector
nodeSelector: {}

# Tolerations
tolerations: []

# Affinity (Pod Anti-Affinity for HA)
affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchExpressions:
              - key: app.kubernetes.io/name
                operator: In
                values:
                  - mlflow
          topologyKey: kubernetes.io/hostname
```

### values-production.yaml (프로덕션 오버라이드)

```yaml
# charts/mlflow/values-production.yaml

replicaCount: 3  # 프로덕션: 최소 3개

image:
  tag: "v2.10.2"

serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: "arn:aws:iam::123456789012:role/mdpg-mlops-mlflow-s3-access"

mlflow:
  backendStoreUri: "postgresql://mlflow:SECURE_PASSWORD@mdpg-mlops-mlflow-db.xxxxx.us-west-2.rds.amazonaws.com:5432/mlflow"
  artifactRoot: "s3://mdpg-mlops-mlflow-artifacts/"
  authentication:
    enabled: true
  workers: 8  # 더 많은 동시 요청 처리

resources:
  requests:
    cpu: 1000m
    memory: 2Gi
  limits:
    cpu: 4000m
    memory: 8Gi

ingress:
  annotations:
    alb.ingress.kubernetes.io/certificate-arn: "arn:aws:acm:us-west-2:123456789012:certificate/xxxxx"
  hosts:
    - host: mlflow.mdpg.ai
      paths:
        - path: /
          pathType: Prefix

autoscaling:
  minReplicas: 3
  maxReplicas: 10
```

### templates/deployment.yaml

```yaml
# charts/mlflow/templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "mlflow.fullname" . }}
  labels:
    {{- include "mlflow.labels" . | nindent 4 }}
spec:
  {{- if not .Values.autoscaling.enabled }}
  replicas: {{ .Values.replicaCount }}
  {{- end }}
  selector:
    matchLabels:
      {{- include "mlflow.selectorLabels" . | nindent 6 }}
  template:
    metadata:
      labels:
        {{- include "mlflow.selectorLabels" . | nindent 8 }}
    spec:
      serviceAccountName: {{ include "mlflow.serviceAccountName" . }}
      containers:
        - name: mlflow
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          imagePullPolicy: {{ .Values.image.pullPolicy }}
          command:
            - mlflow
            - server
            - --backend-store-uri
            - {{ .Values.mlflow.backendStoreUri }}
            - --default-artifact-root
            - {{ .Values.mlflow.artifactRoot }}
            - --host
            - {{ .Values.mlflow.host }}
            - --port
            - "{{ .Values.mlflow.port }}"
            - --workers
            - "{{ .Values.mlflow.workers }}"
            {{- if .Values.mlflow.authentication.enabled }}
            - --app-name
            - basic-auth
            {{- end }}
          ports:
            - name: http
              containerPort: {{ .Values.mlflow.port }}
              protocol: TCP
          env:
            {{- toYaml .Values.env | nindent 12 }}
          livenessProbe:
            {{- toYaml .Values.livenessProbe | nindent 12 }}
          readinessProbe:
            {{- toYaml .Values.readinessProbe | nindent 12 }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
      {{- with .Values.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

### templates/service.yaml

```yaml
# charts/mlflow/templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "mlflow.fullname" . }}
  labels:
    {{- include "mlflow.labels" . | nindent 4 }}
spec:
  type: {{ .Values.service.type }}
  ports:
    - port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.targetPort }}
      protocol: TCP
      name: http
  selector:
    {{- include "mlflow.selectorLabels" . | nindent 4 }}
```

### templates/ingress.yaml

```yaml
# charts/mlflow/templates/ingress.yaml
{{- if .Values.ingress.enabled -}}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ include "mlflow.fullname" . }}
  labels:
    {{- include "mlflow.labels" . | nindent 4 }}
  {{- with .Values.ingress.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  ingressClassName: {{ .Values.ingress.className }}
  {{- if .Values.ingress.tls }}
  tls:
    {{- range .Values.ingress.tls }}
    - hosts:
        {{- range .hosts }}
        - {{ . | quote }}
        {{- end }}
    {{- end }}
  {{- end }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host | quote }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ include "mlflow.fullname" $ }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```

### templates/hpa.yaml

```yaml
# charts/mlflow/templates/hpa.yaml
{{- if .Values.autoscaling.enabled }}
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: {{ include "mlflow.fullname" . }}
  labels:
    {{- include "mlflow.labels" . | nindent 4 }}
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: {{ include "mlflow.fullname" . }}
  minReplicas: {{ .Values.autoscaling.minReplicas }}
  maxReplicas: {{ .Values.autoscaling.maxReplicas }}
  metrics:
    {{- if .Values.autoscaling.targetCPUUtilizationPercentage }}
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetCPUUtilizationPercentage }}
    {{- end }}
    {{- if .Values.autoscaling.targetMemoryUtilizationPercentage }}
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: {{ .Values.autoscaling.targetMemoryUtilizationPercentage }}
    {{- end }}
{{- end }}
```

### templates/serviceaccount.yaml

```yaml
# charts/mlflow/templates/serviceaccount.yaml
{{- if .Values.serviceAccount.create -}}
apiVersion: v1
kind: ServiceAccount
metadata:
  name: {{ include "mlflow.serviceAccountName" . }}
  labels:
    {{- include "mlflow.labels" . | nindent 4 }}
  {{- with .Values.serviceAccount.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}
```

### templates/_helpers.tpl

```yaml
# charts/mlflow/templates/_helpers.tpl
{{/*
Expand the name of the chart.
*/}}
{{- define "mlflow.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "mlflow.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "mlflow.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "mlflow.labels" -}}
helm.sh/chart: {{ include "mlflow.chart" . }}
{{ include "mlflow.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "mlflow.selectorLabels" -}}
app.kubernetes.io/name: {{ include "mlflow.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "mlflow.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "mlflow.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}
```

---

## MLflow Authentication 설정

### 1. Admin 계정 생성

```bash
# Namespace 생성
kubectl create namespace ml-platform

# MLflow 배포 (인증 비활성화 상태로 초기 배포)
helm install mlflow ./charts/mlflow \
  --namespace ml-platform \
  --set mlflow.authentication.enabled=false \
  --wait

# Pod 확인
kubectl get pods -n ml-platform

# Admin 계정 생성
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-admin-user \
    --username admin \
    --password SecureAdminPassword123!

# 출력:
# Admin user 'admin' created successfully.
```

### 2. 일반 사용자 계정 생성

```bash
# MLOps Engineer 1
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-user \
    --username mlops_engineer_1 \
    --password SecurePassword1!

# MLOps Engineer 2
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-user \
    --username mlops_engineer_2 \
    --password SecurePassword2!

# ML Engineer
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-user \
    --username ml_engineer_1 \
    --password SecurePassword3!
```

### 3. 권한 설정

MLflow Authentication의 권한 레벨:
- **READ**: 실험 조회, 모델 조회
- **EDIT**: 실험 생성/수정, 모델 등록
- **MANAGE**: 실험 삭제, 사용자 관리

```bash
# Admin으로 로그인하여 권한 설정 (MLflow UI 또는 API 사용)

# 예: Python API로 권한 설정
kubectl exec -it deployment/mlflow -n ml-platform -- python3 <<EOF
import mlflow
from mlflow.server import get_app_client

# MLflow 클라이언트 생성
client = mlflow.tracking.MlflowClient()

# Experiment 권한 설정 예시
# (실제로는 MLflow UI에서 설정하는 것이 더 편리함)
EOF
```

### 4. 인증 활성화

```bash
# Helm upgrade로 인증 활성화
helm upgrade mlflow ./charts/mlflow \
  --namespace ml-platform \
  --set mlflow.authentication.enabled=true \
  --reuse-values \
  --wait

# Pod 재시작 확인
kubectl rollout status deployment/mlflow -n ml-platform
```

---

## 배포 실행

### 1. Terraform Output 가져오기

```bash
cd terraform/aws-eks

# IRSA Role ARN
export MLFLOW_IRSA_ROLE_ARN=$(terraform output -raw mlflow_irsa_role_arn)

# RDS 연결 문자열
export RDS_CONNECTION_STRING=$(terraform output -raw rds_connection_string)

# S3 Bucket 이름
export S3_BUCKET_NAME=$(terraform output -raw s3_bucket_name)

# ACM 인증서 ARN (수동으로 설정 필요)
export ACM_CERTIFICATE_ARN="arn:aws:acm:us-west-2:123456789012:certificate/xxxxx"
```

### 2. values-production.yaml 업데이트

```bash
cat > charts/mlflow/values-production.yaml <<EOF
serviceAccount:
  annotations:
    eks.amazonaws.com/role-arn: "${MLFLOW_IRSA_ROLE_ARN}"

mlflow:
  backendStoreUri: "${RDS_CONNECTION_STRING}"
  artifactRoot: "s3://${S3_BUCKET_NAME}/"
  authentication:
    enabled: true

ingress:
  annotations:
    alb.ingress.kubernetes.io/certificate-arn: "${ACM_CERTIFICATE_ARN}"
  hosts:
    - host: mlflow.mdpg.ai
      paths:
        - path: /
          pathType: Prefix
EOF
```

### 3. Helm 배포

```bash
# Namespace 생성
kubectl create namespace ml-platform

# Dry-run 확인
helm install mlflow ./charts/mlflow \
  --namespace ml-platform \
  --values charts/mlflow/values-production.yaml \
  --dry-run --debug

# 실제 배포
helm install mlflow ./charts/mlflow \
  --namespace ml-platform \
  --values charts/mlflow/values-production.yaml \
  --wait

# 배포 상태 확인
kubectl get all -n ml-platform
```

### 4. DNS 설정

```bash
# ALB DNS 이름 확인
kubectl get ingress mlflow -n ml-platform -o jsonpath='{.status.loadBalancer.ingress[0].hostname}'
# k8s-mlplatfo-mlflow-xxxxx-123456789.us-west-2.elb.amazonaws.com

# Route 53 또는 외부 DNS에 CNAME 레코드 추가
# mlflow.mdpg.ai -> k8s-mlplatfo-mlflow-xxxxx-123456789.us-west-2.elb.amazonaws.com
```

---

## 검증 및 테스트

### 1. Pod 상태 확인

```bash
# Pod 확인
kubectl get pods -n ml-platform

# 로그 확인
kubectl logs -f deployment/mlflow -n ml-platform

# 정상 출력:
# [INFO] Starting gunicorn 20.1.0
# [INFO] Listening at: http://0.0.0.0:5000
# [INFO] MLflow server started successfully
```

### 2. Service 확인

```bash
# Service 확인
kubectl get svc mlflow -n ml-platform

# Port-forward로 로컬 테스트
kubectl port-forward svc/mlflow 5000:80 -n ml-platform

# 브라우저에서 http://localhost:5000 접속
```

### 3. Ingress 확인

```bash
# Ingress 상태 확인
kubectl get ingress mlflow -n ml-platform

# ALB 생성 확인 (1-2분 소요)
kubectl describe ingress mlflow -n ml-platform

# HTTPS 접속 테스트
curl -I https://mlflow.mdpg.ai
# HTTP/2 200
```

### 4. MLflow 연결 테스트

```bash
# 로컬에서 MLflow 연결 테스트
export MLFLOW_TRACKING_URI=https://mlflow.mdpg.ai
export MLFLOW_TRACKING_USERNAME=admin
export MLFLOW_TRACKING_PASSWORD=SecureAdminPassword123!

python3 <<EOF
import mlflow

mlflow.set_experiment("connection-test")

with mlflow.start_run():
    mlflow.log_param("test", "success")
    mlflow.log_metric("accuracy", 0.95)

print("✅ MLflow connection successful!")
EOF
```

### 5. S3 아티팩트 확인

```bash
# S3에 아티팩트가 업로드되었는지 확인
aws s3 ls s3://${S3_BUCKET_NAME}/ --recursive

# 출력 예:
# 2025-10-21 10:30:15      12345 0/abc123/artifacts/model.pkl
```

---

## 사용자 관리

### 사용자 추가

```bash
# 새 사용자 생성
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-user \
    --username new_user \
    --password SecurePassword!
```

### 사용자 삭제

```bash
# 사용자 삭제
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server delete-user \
    --username old_user
```

### 비밀번호 변경

```bash
# 사용자 비밀번호 변경
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server update-user \
    --username user_name \
    --password NewSecurePassword!
```

### 사용자 목록 조회

```python
# Python으로 사용자 목록 조회
import mlflow
from mlflow.server.auth import client

auth_client = client.AuthServiceClient("https://mlflow.mdpg.ai")
auth_client.login("admin", "SecureAdminPassword123!")

users = auth_client.list_users()
for user in users:
    print(f"Username: {user.username}, Admin: {user.is_admin}")
```

---

## 문제 해결

### 문제 1: Pod가 CrashLoopBackOff

**증상**: `kubectl get pods -n ml-platform`에서 CrashLoopBackOff

**해결**:
```bash
# 로그 확인
kubectl logs deployment/mlflow -n ml-platform --previous

# 일반적인 원인:
# 1. RDS 연결 실패 -> backendStoreUri 확인
# 2. S3 접근 권한 없음 -> IRSA Role 확인
# 3. 환경 변수 오류 -> values.yaml 확인

# Security Group 확인
# RDS Security Group이 EKS Node Security Group으로부터 5432 포트 허용하는지 확인
```

### 문제 2: Ingress에서 502 Bad Gateway

**증상**: `curl https://mlflow.mdpg.ai` → 502

**해결**:
```bash
# Service가 Pod를 정확히 선택하는지 확인
kubectl get endpoints mlflow -n ml-platform

# Pod IP와 일치하는지 확인
kubectl get pods -n ml-platform -o wide

# Target Group Health 확인
# AWS Console → EC2 → Target Groups → mdpg-mlops-xxx
```

### 문제 3: 인증 실패

**증상**: `Unauthorized` 에러

**해결**:
```bash
# 인증이 활성화되었는지 확인
kubectl exec -it deployment/mlflow -n ml-platform -- \
  env | grep -i auth

# Admin 계정 재생성
kubectl exec -it deployment/mlflow -n ml-platform -- \
  mlflow server create-admin-user \
    --username admin \
    --password NewSecurePassword123!
```

### 문제 4: S3 업로드 실패

**증상**: `AccessDenied` when uploading artifacts

**해결**:
```bash
# IRSA Role이 ServiceAccount에 제대로 연결되었는지 확인
kubectl get sa mlflow -n ml-platform -o yaml | grep eks.amazonaws.com/role-arn

# Pod에서 AWS credentials 확인
kubectl exec -it deployment/mlflow -n ml-platform -- \
  aws sts get-caller-identity

# IAM Policy 확인
aws iam get-policy-version \
  --policy-arn $(aws iam list-attached-role-policies --role-name mdpg-mlops-mlflow-s3-access --query 'AttachedPolicies[0].PolicyArn' --output text) \
  --version-id v1
```

---

## 다음 단계

MLflow 서버 배포가 완료되면:

1. **클라이언트 설정**: [docs/vscode_setup.md](vscode_setup.md)
2. **마이그레이션**: [docs/migration_guide.md](migration_guide.md)
3. **자동화 스크립트**: [docs/deployment_scripts.md](deployment_scripts.md)

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
