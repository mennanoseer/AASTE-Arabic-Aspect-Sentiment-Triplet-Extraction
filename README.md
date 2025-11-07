# AASTE: Arabic Aspect-Sentiment Triplet Extraction

A PyTorch implementation of Aspect-Sentiment Triplet Extraction (ASTE) for Arabic text, based on span-based tagging schemes. This project extracts triplets of (aspect term, opinion term, sentiment) from Arabic restaurant reviews.

## Overview

This project implements a BERT-based model for Arabic Aspect-Sentiment Triplet Extraction (ASTE), which simultaneously identifies:
- **Aspect terms**: The target entities being evaluated
- **Opinion terms**: The expressions conveying sentiment
- **Sentiment polarity**: Positive, negative, or neutral sentiment

The model uses a span-based tagging approach with three different tagging schemes (1D, 2D, 3D) and is specifically designed for Arabic text processing.

## Architecture

The model consists of three main components:

1. **BERT Encoder**: Uses AraBERT for contextual Arabic text representation
2. **Span Representation**: Creates span representations from boundary and content features
3. **Triplet Classifier**: Classifies spans into aspect/opinion terms with sentiment

### Tagging Schemes

- **3D**: `{N,A}-{N,O}-{N,NEG,NEU,POS}` - Separate dimensions for aspect, opinion, and sentiment
- **2D**: `{N,O,A}-{N,NEG,NEU,POS}` - Combined aspect/opinion dimension with sentiment
- **1D**: `{N,NEG,NEU,POS,O,A}` - All labels in a single dimension

## Project Structure

```
AASTE-Arabic-Aspect-Sentiment-Triplet-Extraction/
├── models/                    # Model architecture
│   ├── __init__.py
│   └── aste_model.py         # Main BERT-based ASTE model
├── data/                      # Data loading and processing
│   ├── __init__.py
│   ├── dataset.py            # Dataset class and data loading
│   ├── vocab.py              # Vocabulary management
│   └── data_utils.py         # Data utilities and preprocessing
├── tagging/                   # Tagging schemes and inference
│   ├── __init__.py
│   ├── span_tagging.py       # Tagging scheme implementations
│   └── inference.py          # Greedy inference algorithm
├── training/                  # Training and evaluation
│   ├── __init__.py
│   ├── evaluate.py           # Evaluation metrics and functions
│   └── training_utils.py     # Training utilities
├── utils/                     # Utilities and configuration
│   ├── __init__.py
│   ├── config.py             # Argument parsing and configuration
│   └── helpers.py            # General helper functions
├── scripts/                   # Executable scripts
│   ├── run.py                # Training and evaluation pipeline
│   └── predict.py            # Inference script for predictions
├── notebooks/                 # Jupyter notebooks
│   ├── inference.ipynb       # Interactive inference notebook
│   ├── DataValidation.ipynb  # Data validation
│   └── spanTaggingExample.ipynb  # Span tagging examples
├── datasets/                  # Dataset files
│   └── ASTE-Data-V2-EMNLP2020_TRANSLATED_TO_ARABIC/
│       └── 16res/
│           ├── train_triplets.txt    # Training dataset
│           ├── dev_triplets.txt      # Validation dataset
│           └── test_triplets.txt     # Test dataset
├── outputs/                   # Model outputs and results
│   ├── models/               # Saved model checkpoints
│   ├── best_models/          # Best performing models
│   └── results/              # Experiment results
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
└── README.md                  # This file

```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/mennanoseer/AASTE-Arabic-Aspect-Sentiment-Triplet-Extraction.git
cd AASTE-Arabic-Aspect-Sentiment-Triplet-Extraction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training

Run training with default parameters:
```bash
python scripts/run.py
```

Or with custom parameters:
```bash
python scripts/run.py --dataset 16res --version 3D --hidden_dim 200 --num_epoch 100 --batch_size 16 --lr 1e-3 --bert_lr 2e-5
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/predict.py --load_model_path outputs/models/your_model.pt
```

### Interactive Inference

Use the Jupyter notebook for interactive inference:
```bash
jupyter notebook inference.ipynb
```

## Model Inference API

For programmatic inference, you can use the model directly:

```python
import torch
from transformers import AutoTokenizer
from models.aste_model import base_model
from data.vocab import load_vocab
from tagging.span_tagging import form_label_id_map, form_sentiment_id_map
from tagging.inference import extract_triplets_from_tags

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('aubmindlab/bert-base-arabertv2')
model = torch.load('outputs/best_models/16res_3D_True_best.pkl')

# Process Arabic text
sentence = "يقدم سوشي جيد حقا"
# ... preprocessing code (see notebooks/inference.ipynb for complete example)
```

## Data Format

The dataset uses the following format for triplet annotation:

```
يقدم سوشي جيد حقا .####[([1], [2], 'POS')]
```

Where:
- Text before `####`: Arabic sentence with space-separated tokens
- Text after `####`: List of triplets `[(aspect_indices, opinion_indices, sentiment)]`
- Sentiment labels: `POS` (positive), `NEG` (negative), `NEU` (neutral)

### Example Annotations

| Arabic Text | Aspect | Opinion | Sentiment |
|-------------|---------|----------|-----------|
| يقدم سوشي جيد حقا | سوشي (sushi) | جيد (good) | POS |
| الطعام رديء | الطعام (food) | رديء (bad) | NEG |
| مطعم المدينة هو الأفضل | مطعم المدينة (Almadina restaurant) | الأفضل (the best) | POS |

### Dataset Statistics

- **Domain**: Restaurant reviews (translated from English)
- **Language**: Arabic
- **Total samples**: ~850+ (varies after data preparation)
- **Annotation scheme**: Aspect-Opinion-Sentiment triplets
- **Average sentence length**: 10-15 tokens

## Configuration

### Model Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pretrained_model` | AraBERT model path | `aubmindlab/bert-base-arabertv2` |
| `--hidden_dim` | Hidden dimension size | 200 |
| `--dropout_rate` | Dropout rate | 0.5 |
| `--version` | Tagging scheme (1D/2D/3D) | 3D |
| `--span_average` | Use span averaging | False |

### Training Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--num_epoch` | Number of training epochs | 100 |
| `--batch_size` | Batch size | 16 |
| `--lr` | Learning rate for non-BERT parameters | 1e-3 |
| `--bert_lr` | Learning rate for BERT parameters | 2e-5 |
| `--l2` | L2 regularization | 0.0 |
| `--with_weight` | Use class weights | True |

## Performance

The model achieves competitive performance on the Arabic restaurant review dataset (16res):

| Tagging Scheme | Best Seed | F1 Score |
|----------------|-----------|----------|
| 3D             | 1024      | ~65-70%* |
| 2D             | 63        | ~60-65%* |
| 1D             | 270       | ~55-60%* |

*Approximate performance based on model configuration. Actual results may vary.

### Evaluation Metrics

The model is evaluated on:
- **Aspect Term Extraction (ATE)**: Precision, Recall, F1 for aspect terms
- **Opinion Term Extraction (OTE)**: Precision, Recall, F1 for opinion terms  
- **Aspect-Opinion Pair Extraction**: F1 for aspect-opinion pairs
- **Triplet Extraction**: F1 for complete (aspect, opinion, sentiment) triplets

## Key Features

### Modular Architecture
- **Organized Code Structure**: Clean separation of concerns with dedicated modules
- **Package-based Design**: Proper Python packages with `__init__.py` files
- **Easy to Extend**: Modular components make it simple to add new features
- **Well-Documented**: Comprehensive docstrings and documentation files

### Arabic Text Processing
- **Arabic Normalization**: Handles different Arabic character forms, diacritics removal
- **AraBERT Integration**: Uses pre-trained Arabic BERT for better Arabic understanding
- **Subword Handling**: Proper mapping between BERT subwords and original tokens

### Span-Based Approach
- **Boundary Features**: Uses start and end token representations
- **Span Content**: Aggregates information within spans
- **Flexible Spans**: Handles multi-word aspect and opinion terms

### Greedy Inference
- **Efficient Decoding**: Greedy algorithm for triplet extraction
- **Conflict Resolution**: Handles overlapping predictions
- **Multiple Patterns**: Supports aspect-opinion and opinion-aspect patterns


## Related Work

This implementation is inspired by:
- STAGE model - Span-based Tagging for Aspect-sentiment Triplet Extraction ([Paper](https://ojs.aaai.org/index.php/AAAI/article/view/26547), [GitHub](https://github.com/CCIIPLab/STAGE.git))
- ASTE-Data-V2 (EMNLP 2020) - Original English dataset ([GitHub](https://github.com/xuuuluuu/SemEval-Triplet-data))
- AraBERT - Arabic BERT pre-trained model
- Span-based tagging approaches for structured prediction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- AraBERT team for the pre-trained Arabic BERT model
- ASTE-Data-V2 dataset providers
- PyTorch and Transformers library developers

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainer.

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size (`--batch_size 8`)
2. **Missing data files**: Ensure train_triplets.txt and dev_triplets.txt exist in `datasets/ASTE-Data-V2-EMNLP2020_TRANSLATED_TO_ARABIC/16res/`
3. **Tokenizer issues**: Verify AraBERT model is correctly downloaded
4. **Performance issues**: Try different seeds or adjust learning rates

### System Requirements

- Python 3.7+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 2GB+ disk space for models and data

---

**Note**: This project is designed specifically for Arabic text processing. For other languages, consider adapting the normalization functions and using appropriate pre-trained models.