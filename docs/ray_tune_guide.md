# Ray Tune 하이퍼파라미터 최적화 가이드

**작성일**: 2025-10-21
**버전**: 1.0
**대상**: Phase 6 - Ray Tune 하이퍼파라미터 최적화

---

## 개요

### 목표

Ray Tune을 사용하여 100+ trials 병렬 실행으로 CIFAR-10 정확도 90%+ 달성

### 핵심 기능

- **분산 실행**: GPU auto-scaling (0 → 5 → 0)
- **MLflow 통합**: 모든 trial 자동 기록
- **ASHA Scheduler**: 조기 종료로 비용 절감
- **Bayesian Optimization**: 효율적인 탐색

---

## Ray Cluster 배포

### Helm Chart

```yaml
# charts/ray-cluster/values.yaml
head:
  cpu: 2
  memory: 4Gi
  replicas: 1

worker:
  minReplicas: 0
  maxReplicas: 5
  cpu: 4
  memory: 16Gi
  gpu: 1
  nodeType: p3.2xlarge
```

### 배포

```bash
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/default/operator.yaml

helm install ray-cluster ./charts/ray-cluster \
  --namespace ml-platform \
  --wait
```

---

## Ray Tune 코드

### src/tuning/ray_tune.py

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.integration.mlflow import MLflowLoggerCallback
import mlflow

def train_cifar10(config):
    model = VisionModel(
        model_name=config["model_name"],
        num_classes=10
    )

    for epoch in range(config["epochs"]):
        train_loss = train_epoch(model, config["lr"], config["batch_size"])
        val_acc = validate(model)

        tune.report(val_accuracy=val_acc, train_loss=train_loss)

# 탐색 공간
config = {
    "model_name": tune.choice(["mobilenet_v3_small", "resnet18"]),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([32, 64, 128]),
    "epochs": 20,
}

# Ray Tune 실행
analysis = tune.run(
    train_cifar10,
    config=config,
    num_samples=100,
    resources_per_trial={"gpu": 1},
    scheduler=ASHAScheduler(metric="val_accuracy", mode="max"),
    callbacks=[MLflowLoggerCallback(
        tracking_uri="https://mlflow.mdpg.ai",
        experiment_name="ray-tune-cifar10"
    )]
)

# 최고 모델
best_trial = analysis.best_trial
print(f"Best accuracy: {best_trial.last_result['val_accuracy']:.2f}%")
```

---

## 실행

```bash
# Ray Tune 실행
poetry run python -m src.tuning.ray_tune --num-samples 100

# Ray Dashboard에서 모니터링
kubectl port-forward svc/ray-head 8265:8265 -n ml-platform
open http://localhost:8265
```

---

**작성자**: MLOps Team
**최종 업데이트**: 2025-10-21
