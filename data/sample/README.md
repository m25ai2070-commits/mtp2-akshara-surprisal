# Sample dataset

`xlsum/hindi/test.jsonl.gz` and `xlsum/hindi/train.jsonl.gz` hold the Hindi XL-Sum sample used by
`python run_all.py --sample` (complete test split; first 3,000 train articles for word frequencies).
Together with `../meco/` this is enough to run every step of the pipeline end to end.

If the `xlsum/` folder is not here, create it with `python scripts/export_xlsum.py --sample` (internet needed
once), or let the sample run download Hindi XL-Sum from Hugging Face automatically.
