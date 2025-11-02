# Enterprise AI Fault Diagnosis Platform — GPU-Powered Deployment (Triton + FastAPI + EKS)
> Production-grade enterprise system for automated fault detection, image similarity search, and engineering knowledge retrieval — deployed on NVIDIA GPU infrastructure & Kubernetes (EKS).


#### This repository delivers a real-world industrial AI platform capable of:

- Vision-based fault classification (CNN backbone — future ViT upgrade)

- FAISS-based vector similarity search for historical defect images

- Engineering knowledge lookup via internal LLM/RAG connectors (future plug-in)

- GPU-accelerated inference using NVIDIA Triton

- FastAPI microservice with gRPC → Triton

- Helm chart for AWS EKS GPU deployment

- CI/CD automation (GitHub Actions)

#### This repo represents the deployment-ready component of the larger Enterprise Fault-Diagnosis AI program.

## Repository Structure

```python
enterprise-image-fault-diagnosis-deploy/
│
├─ src/
│   ├─ fastapi_service/
│   │   ├─ app/                  # FastAPI + Triton gRPC client
│   │   └─ Dockerfile.fastapi    # Microservice build
│   │
│   └─ triton_model_repo/        # Triton model structure + config.pbtxt
│
├─ helm/
│   └─ enterprise-deploy/        # Helm chart for EKS GPU launch
│
├─ k8s/                          # Raw Kubernetes YAML files (non-Helm)
│
├─ .github/workflows/ci.yml      # CI automation example
│
├─ README.md
└─ LICENSE

```

---

## Key Platform Features

#### <ins> Fault Image Recognition </ins>

- Deep CNN inference served via Triton

- Hardware acceleration on NVIDIA GPUs

- Ready for Vision Transformers upgrade
- 

#### <ins>  Similar Image Search (FAISS)</ins>

- Return top-N visually similar faults

- Enterprise-scale vector search pipeline

- GPU-accelerated indexing supported
- 

#### <ins>  Internal Knowledge Retrieval (RAG-ready)</ins>

- Plug-in interface for PDF fault report retrieval

- Integrate with enterprise SharePoint / Confluence / PLM

- LLM-assisted recommended fixes (future)
- 

#### <ins>  MLOps & Reliability</ins>

- Containerized inference microservice

- CI/CD automation pipeline

- Kubernetes GPU autoscaling support

---

## Deployment — AWS EKS + NVIDIA GPUs

<ins>Prerequisites</ins>

- AWS EKS cluster

- GPU worker nodes (A10G / A100 / V100)

- NVIDIA device plugin installed

- Helm installed

#### Deploy with Helm

```python
kubectl create namespace fault-ai
helm install fault-ai ./helm/enterprise-deploy -n fault-ai
```

#### Microservice — FastAPI (gRPC to Triton)

##### Local Run

```python
cd src/fastapi_service
docker build -f Dockerfile.fastapi -t fault-ai-api .
docker run --gpus all -p 8080:8080 fault-ai-api
```

---

#### CI/CD Pipeline

##### GitHub Actions file:

```python
.github/workflows/ci.yml
```

<ins>Includes</ins>:

- Build & scan containers
-  Push to registry
-  Lint Kubernetes manifests
-  Helm dry-run & validation

## Architecture Diagram

### Vision Model + Vector Search + GPU Inference + EKS MLOps

```python
Customer Fault Images → Triton (GPU) → FastAPI → Similarity Engine (FAISS)
                                    ↓
                        Engineering RAG + PDF RCA
```


