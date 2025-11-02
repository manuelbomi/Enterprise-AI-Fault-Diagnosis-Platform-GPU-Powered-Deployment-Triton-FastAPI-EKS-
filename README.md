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

> [!NOTE]
> NVIDIA Triton refers to NVIDIA Triton Inference Server, a software platform used to deploy and serve AI/ML models efficiently on:

> NVIDIA GPUs

> CPUs

> Cloud environments (AWS, GCP, Azure)

> Kubernetes (e.g., EKS, GKE, AKS)

> Edge devices
>


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

---


## Enterprise Use Cases

| Department | Benefit |
|------------|---------|
| Field Engineering | Faster fault triage |
| Quality Engineering | RCA acceleration |
| Manufacturing | Systemic defect detection |
| Customer Support | Visual case lookup & automation |
| R&D | Feedback loop for new designs |

---

## Maintainer Notes

#### This repo is part of a multi-repo enterprise AI program including:

- Model training pipeline (TensorFlow/PyTorch GPU)

- FAISS vector indexing pipeline

- RAG for engineering PDFs + fault reports

- Triton GPU serving

- EKS Helm deployment

---

Thank you for reading

---

### **AUTHOR'S BACKGROUND**

### Author's Name:  Emmanuel Oyekanlu
```
Skillset:   I have experience spanning several years in data science, software and AI solution design and deployments, data engineering,
high performance computing (GPU, CUDA), machine learning, MLOps, NLP, Agentic-AI and LLM applications, developing scalable enterprise data pipelines,
enterprise solution architecture, architecting enterprise systems data and AI applications, as well as deploying scalable solutions (apps) on-prem and in the cloud.

I can be reached through: manuelbomi@yahoo.com

Website:  http://emmanueloyekanlu.com/
Publications:  https://scholar.google.com/citations?user=S-jTMfkAAAAJ&hl=en
LinkedIn:  https://www.linkedin.com/in/emmanuel-oyekanlu-6ba98616
Github:  https://github.com/manuelbomi

```
[![Icons](https://skillicons.dev/icons?i=aws,azure,gcp,scala,mongodb,redis,cassandra,kafka,anaconda,matlab,nodejs,django,py,c,anaconda,git,github,mysql,docker,kubernetes&theme=dark)](https://skillicons.dev)









