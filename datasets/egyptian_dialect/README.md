# Egyptian Dialect Dataset

This directory contains Arabic aspect sentiment triplet extraction data in **Egyptian dialect**, scraped from Amazon Egypt product reviews.

## Data Structure

The dataset is organized into three product categories:

```
egyptian_dialect/
├── health/           # Health & personal care products
├── fashion/          # Fashion & clothing products
├── electronics/      # Electronics & technology products
└── combined/         # All categories merged 
```

Each category contains:
- `unmerged/` - Individual data files collected by team members
- `*_merged.json` - Consolidated data file (original format)
- `train_triplets.txt` - Training set (70%)
- `dev_triplets.txt` - Development/validation set (15%)
- `test_triplets.txt` - Test set (15%)

## Data Format

### Original JSON Format
Each review in the `*_merged.json` files contains:
```json
{
    "review_text": "النص العربي للمراجعة",
    "triplets": "[([aspect_indices], [opinion_indices], 'sentiment')]",
    "star_rating": "5.0",
    "product_name": "اسم المنتج",
    "category": "Health",
    "scraped_at": "2025-12-24T23:31:22.643165",
    "product_link": "https://..."
}
```

### Triplet Text Format
The `*_triplets.txt` files use the format:
```
review_text####[([aspect_start, aspect_end], [opinion_start, opinion_end], 'sentiment')]
```

Example:
```
الغسول جه بتاريخ انتاج قديم جدا و ليه ريحة سيئة####[([0], [1, 2, 3, 4], 'NEG'), ([0], [7, 8, 9], 'NEG')]
```

## Data Split

The data has been automatically split using `scripts/prepare_egyptian_data.py` with:
- **Training**: 70%
- **Development**: 15%
- **Testing**: 15%
- **Random seed**: 42 (for reproducibility)

## Usage

### Preparing Data

To prepare individual category data:
```bash
python scripts/prepare_egyptian_data.py --category health
python scripts/prepare_egyptian_data.py --category fashion
python scripts/prepare_egyptian_data.py --category electronics
```

To merge all categories into one combined dataset:
```bash
python scripts/prepare_egyptian_data.py --category combined
```

### Training with Egyptian Dialect Data

To train the ASTE model on a single category:

```bash
python scripts/run.py --dataset egyptian_health --num_epoch 10 --batch_size 32
python scripts/run.py --dataset egyptian_fashion --num_epoch 10 --batch_size 32
python scripts/run.py --dataset egyptian_electronics --num_epoch 10 --batch_size 32
```

To train on all categories combined:
```bash
python scripts/run.py --dataset egyptian_combined --num_epoch 10 --batch_size 32
```

### Making Predictions

For individual category:
```bash
python scripts/predict.py \
    --model_path outputs/models/egyptian_health/egyptian_health_3D_weightFalse_best.pkl \
    --dataset egyptian_health \
    --batch_size 16
```

For combined dataset:
```bash
python scripts/predict.py \
    --model_path outputs/models/egyptian_combined/egyptian_combined_3D_weightFalse_best.pkl \
    --dataset egyptian_combined \
    --batch_size 16
```

## Dataset Statistics

### Health Category
- Total reviews: 899
- Training samples: 629
- Development samples: 134
- Test samples: 136

### Fashion Category
- Status: Pending team contributions

### Electronics Category
- Status: Pending team contributions

## Data Source

- **Source**: Amazon Egypt (amazon.eg)
- **Language**: Egyptian Arabic dialect
- **Collection Period**: December 2025
- **Annotation**: Aspect-Opinion-Sentiment triplets

## Notes

- The Egyptian dialect differs from Modern Standard Arabic (MSA) in vocabulary, grammar, and expressions
- Data includes informal language and colloquial expressions typical of online reviews
- Character encoding: UTF-8
- Sentiment labels: POS (positive), NEG (negative), NEU (neutral)

## Team Contributors

Data collection and annotation were divided among team members for efficient processing.

---

For questions about the dataset or to contribute additional categories, please contact the project maintainers.
