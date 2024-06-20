## Leveraging LLMs for Post-OCR Correction of Historical Newspapers
This repository contains the code for the paper [Leveraging LLMs for Post-OCR Correction of Historical Newspapers](https://aclanthology.org/2024.lt4hala-1.14/), where LLMs are adapted for a prompt-based approach to post-OCR correction. We focus on the post-OCR correction of historical English, using [BLN600](https://aclanthology.org/2024.lrec-main.219/), a parallel corpus of 19th century newspaper machine/human transcription. The dataset and fine-tuned models can be accessed from the links below.

**Data**: https://doi.org/10.15131/shef.data.25439023  
**Models**: https://huggingface.co/pykale

### Usage

`bln600.ipynb` - Notebook for loading and preparing BLN600 for model development

`bart.py` and `llama-2.py` are minimal scripts for fine-tuning BART and Llama 2 for post-OCR correction, using a YAML configuration file and a CSV file for training data with 'OCR Text' and 'Ground Truth' columns.

`results.ipynb` - Notebook for generating and examining post-OCR corrections with fine-tuned models

```(bash)
pip install -r requirements.txt
```

```(bash)
python bart.py --model {bart-base, bart-large} --config CONFIG --data DATA
```

```(bash)
python llama-2.py --model {llama-2-7b, llama-2-13b, llama-2-70b} --config CONFIG --data DATA
```

### Citation
```
@inproceedings{thomas-etal-2024-leveraging,
    title = "Leveraging {LLM}s for Post-{OCR} Correction of Historical Newspapers",
    author = "Thomas, Alan and Gaizauskas, Robert and Lu, Haiping",
    editor = "Sprugnoli, Rachele and Passarotti, Marco",
    booktitle = "Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024",
    month = "may",
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lt4hala-1.14",
    pages = "116--121",
}
```