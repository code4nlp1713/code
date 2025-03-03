# Data and Code
This repository contains anonymized data and code. 

## Data-1: Human-LLM Collaborative Financial Data Annotation
We use **LLM** (`mistralai/Mixtral-8x7B-Instruct-v0.1`) to annotate [FineWeb](data) data instances, followed by manual review and correction by human experts. The final refined dataset follows the format below:

```json
{
    "id": 0,
    "uuid": "<urn:uuid:a9b25faf-6793-45cd-b73b-3869858ba063>",
    "text": "he VanEck Vectors Gold Miners ETF (NYSEArca: GDX) and the VanEck Vectors Gold Miners...",
    "label_mixtral": 1,
    "label_human": 1
}
```
Where `uuid` is the unique identifier for each document in the FineWeb dataset, `label_mixtral` is the annotation from the Mixtral model, and `label_human` is the label assigned by a human annotator. We use this data to train a financial classifier.
### Dataset Overview:
- [3,840 financial documents](https://github.com/code4nlp1713/code/blob/main/financial_human_anno_doc.json) (positive class, label: 1)
- [3,840 non-financial documents](https://github.com/code4nlp1713/code/blob/main/non_financial_human_anno_doc.json) (negative class, label: 0)

## Data-2: financial hypernyms generation.
Here are [20 samples of financial hypernym](https://github.com/code4nlp1713/code/blob/main/financial_hypernym.txt). 
We employ a 3-step approach to generate financial hypernyms
- step 1. **Financial Document Selection**: We identify financial documents by selecting all FineWeb documents with a financial score greater than 0.999.
- step 2. **We obtain high-frequency words**: We extract high-frequency words from these financial documents, excluding numerical values and commonly used words from Wikipedia.
- step 3. **We use LLMs to obtain hypernym**: We leverage LLMs to obtain hypernyms for nouns and adjectives in the extracted financial words.

