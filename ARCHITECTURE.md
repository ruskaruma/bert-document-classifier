# Project Architecture

## Overview

This project implements a BERT-based document classifier that categorizes scientific paper references into three types: Primary, Secondary, or Missing. The system consists of a training pipeline, a BERT model, and a FastAPI-based inference service.

## System Architecture

```mermaid
graph TB
    subgraph Training
        A[XML/PDF Documents] --> B[Text Extraction]
        B --> C[Feature Engineering]
        C --> D[BERT Tokenizer]
        D --> E[BERT Model Training]
        E --> F[Saved Model Files]
    end
    
    subgraph Inference
        G[PDF Upload] --> H[FastAPI Endpoint]
        H --> I[Text Extraction]
        I --> J[BERT Tokenizer]
        J --> K[Trained Model]
        F -.-> K
        K --> L[Classification Result]
        L --> H
    end
```

## Data Flow

```mermaid
flowchart LR
    A[Input PDF] --> B[PyMuPDF Extraction]
    B --> C[Text String]
    C --> D[BERT Tokenizer]
    D --> E[Token IDs + Attention Mask]
    E --> F[BERT Model]
    F --> G[Logits]
    G --> H[Argmax]
    H --> I[Label Encoder]
    I --> J[Primary/Secondary/Missing]
```

## Model Architecture

```mermaid
graph TB
    subgraph Input
        A[Text Input]
    end
    
    subgraph Tokenization
        B[BERT Tokenizer]
        B1[input_ids]
        B2[attention_mask]
    end
    
    subgraph BERTModel
        C[BERT Base Uncased]
        C1[12 Transformer Layers]
        C2[768 Hidden Size]
        C3[Pooler Output]
    end
    
    subgraph Classifier
        D[Dropout 0.3]
        E[Linear Layer 768 -> 3]
        F[Softmax]
    end
    
    subgraph Output
        G[Class Probabilities]
        H[Predicted Label]
    end
    
    A --> B
    B --> B1
    B --> B2
    B1 --> C
    B2 --> C
    C --> C1
    C1 --> C2
    C2 --> C3
    C3 --> D
    D --> E
    E --> F
    F --> G
    G --> H
```

## API Architecture

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant TempFile
    participant Inference
    participant Model
    participant LabelEncoder
    
    Client->>FastAPI: POST /predict/ with PDF
    FastAPI->>FastAPI: Validate file type
    FastAPI->>TempFile: Save uploaded file
    FastAPI->>Inference: predict_pdf(temp_path)
    Inference->>Inference: extract_text_from_pdf()
    Inference->>Model: tokenize + forward pass
    Model->>Model: BERT inference
    Model-->>Inference: logits
    Inference->>LabelEncoder: inverse_transform(prediction)
    LabelEncoder-->>Inference: label string
    Inference-->>FastAPI: classification result
    FastAPI->>TempFile: cleanup temp file
    FastAPI-->>Client: JSON response
```

## Directory Structure

```mermaid
graph TB
    A[bert-document-classifier/]
    A --> B[model/]
    A --> C[api/]
    A --> D[saved_model/]
    A --> E[data/]
    A --> F[experiment_01_document_classifier.ipynb]
    
    B --> B1[bert_classifier.py]
    B --> B2[inference.py]
    B --> B3[__init__.py]
    
    C --> C1[app.py]
    C --> C2[__init__.py]
    
    D --> D1[model.pt]
    D --> D2[label_encoder.joblib]
    D --> D3[README.md]
    
    E --> E1[train/]
    E --> E2[test/]
    E --> E3[train_labels.csv]
```

## Component Interactions

```mermaid
graph LR
    subgraph API Layer
        A[FastAPI App]
        A1[/predict/]
        A2[/health]
        A3[/]
    end
    
    subgraph Model Layer
        B[inference.py]
        B1[extract_text_from_pdf]
        B2[predict_pdf]
    end
    
    subgraph Core Model
        C[BertClassifier]
        C1[BERT Base]
        C2[Dropout]
        C3[Linear Layer]
    end
    
    subgraph External Files
        D[saved_model/]
        D1[model.pt]
        D2[label_encoder.joblib]
    end
    
    A --> A1
    A --> A2
    A --> A3
    A1 --> B2
    B2 --> B1
    B2 --> C
    C --> C1
    C1 --> C2
    C2 --> C3
    C --> D1
    B2 --> D2
```

## Training Pipeline

```mermaid
flowchart TD
    A[Load Training Data] --> B[Extract Text from XML/PDF]
    B --> C[Feature Engineering]
    C --> D[Label Encoding]
    D --> E[Train/Val Split]
    E --> F[Tokenize with BERT]
    F --> G[Create PyTorch Datasets]
    G --> H[Initialize BERT Classifier]
    H --> I[Training Loop - 3 Epochs]
    I --> J{Validation}
    J -->|Continue| I
    J -->|Complete| K[Save model.pt]
    K --> L[Save label_encoder.joblib]
    L --> M[Model Ready for Inference]
```

## Inference Pipeline

```mermaid
flowchart TD
    A[Receive PDF] --> B{File Valid?}
    B -->|No| C[Return 400 Error]
    B -->|Yes| D[Save to Temp File]
    D --> E[Extract Text PyMuPDF]
    E --> F{Text Extracted?}
    F -->|No| G[Return 400 Error]
    F -->|Yes| H[Tokenize Text]
    H --> I[Load Model from saved_model/]
    I --> J[Forward Pass]
    J --> K[Get Logits]
    K --> L[Argmax]
    L --> M[Decode with Label Encoder]
    M --> N[Return Classification]
    N --> O[Cleanup Temp File]
```

## Key Design Decisions

### 1. Model Selection
- **BERT Base Uncased**: 110M parameters, good balance of performance and speed
- **Max Length 512**: Standard BERT limit, handles most document sections
- **Dropout 0.3**: Regularization to prevent overfitting

### 2. Data Processing
- **PyMuPDF over PyPDF2**: Better text extraction quality and speed
- **BeautifulSoup for XML**: Robust parsing of structured XML documents
- **Feature engineering**: Captures data reference patterns

### 3. API Design
- **FastAPI**: Modern async framework with automatic API docs
- **Temporary file handling**: Secure file processing with cleanup
- **Error handling**: Proper HTTP status codes and error messages

### 4. Deployment Considerations
- **Model files gitignored**: Keep repository size manageable
- **Separate requirements files**: Minimal deps for inference, full deps for training
- **Health check endpoint**: Enable monitoring and container orchestration

## Performance Characteristics

### Model Performance
- Accuracy: 95%
- Primary class: 91% precision, 93% recall
- Secondary class: 99% precision, 94% recall
- Missing class: 95% precision, 99% recall

### Inference Speed
- Text extraction: ~100-500ms per PDF
- Model inference: ~50-200ms per document
- Total latency: ~150-700ms depending on PDF size

### Resource Requirements
- Model size: ~440 MB (BERT weights)
- Memory usage: ~2-4 GB during inference
- GPU recommended but not required

## Technology Stack

```mermaid
graph TB
    subgraph ML Framework
        A[PyTorch 2.1.2]
        B[Transformers 4.39.3]
    end
    
    subgraph API Framework
        C[FastAPI 0.116.1]
        D[Uvicorn 0.35.0]
    end
    
    subgraph Document Processing
        E[PyMuPDF 1.26.3]
        F[BeautifulSoup4 4.13.4]
    end
    
    subgraph Data Science
        G[Pandas 2.2.2]
        H[NumPy 1.26.4]
        I[Scikit-learn 1.3.2]
    end
    
    subgraph Utilities
        J[Joblib 1.5.1]
        K[Python 3.9+]
    end
```

## Error Handling Strategy

```mermaid
flowchart TD
    A[Request Received] --> B{File Type Valid?}
    B -->|No| C[HTTPException 400]
    B -->|Yes| D{File Exists?}
    D -->|No| E[HTTPException 404]
    D -->|Yes| F{Text Extracted?}
    F -->|No| G[HTTPException 400]
    F -->|Yes| H{Model Loaded?}
    H -->|No| I[HTTPException 500]
    H -->|Yes| J[Process]
    J --> K{Success?}
    K -->|No| L[HTTPException 500]
    K -->|Yes| M[Return 200 + Result]
```

## Future Enhancements

1. **Model Improvements**
   - Fine-tune on domain-specific data
   - Experiment with larger models (BERT-Large, RoBERTa)
   - Add confidence scores to predictions

2. **API Enhancements**
   - Batch prediction endpoint
   - Support for more file formats
   - Caching layer for repeated requests

3. **Infrastructure**
   - Docker containerization
   - Kubernetes deployment
   - Model versioning and A/B testing

4. **Monitoring**
   - Prediction logging
   - Performance metrics
   - Model drift detection

