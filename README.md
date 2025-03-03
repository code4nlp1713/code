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
- step 3. **We use LLMs to obtain hypernym**: We leverage LLMs to obtain hypernyms for nouns in the extracted financial words. Here is the code snippet:
```python
from transformers import pipeline

class FinancialHypernymExtractor:
    def __init__(self):
        self.fill_mask = pipeline("fill-mask", model="roberta-base")  # Using RoBERTa-base model

    def get_financial_hypernyms(self, word, pos="noun"):
        if pos == "noun":
            masked_sentence = f"In financial context, {word} is a type of <mask>."
        else:  
            masked_sentence = f"In financial context, something {word} is <mask>."

        predictions = self.fill_mask(masked_sentence)
        hypernyms = [pred["token_str"].strip() for pred in predictions]  # Cleaning spaces
        return hypernyms

# Example usage
extractor = FinancialHypernymExtractor()
word = "equity"
financial_hypernym = extractor.get_financial_hypernyms(word)[0]  # Taking the top result
print(f"Financial Hypernym of '{word}':", financial_hypernym)
```

## Code: PPO code for optimal explanation search
### Prerequisites
Please refer to the [requirements.txt](https://github.com/code4nlp1713/code/blob/main/requirements.txt) file to install all the necessary packages, including trl (for PPO), accelerate, and others.

### Training
To run the search strategy for refinement of, e.g., 4-grade explanations via feedback from LLMs, we first need to prepare the 4-grade ground truth and then execute the code. Two important arguments to consider are:

 - 'nb_epoch' (number of epochs): This defaults to 50. In the experiment, we set it to 50. Please note that for DFS, each epoch involves evaluating the performance on 600 testing instances (in case of 4-grade experiment) after explanation rewriting, which is time costly. <span style="color: red;">**Empirically, 50 epochs take approximately 8 hours with our available computational resources**.</span>

 - '--fin': This argument determines whether to incorporate financial hypernyms. When enabled, it uses specific financial terminology to enhance the explanation rewriting.

```bash
accelerate launch --config_file config.yaml main_fin_ppo.py --nb_epoch 50 --fin 
```
