# Text Lens

A reproducible experiment asking whether compressed text representations preserve information useful for support-request classification. Compare a TF-IDF logistic regression baseline with a logistic regression probe on 16-dimensional truncated-SVD features.

## Run
```bash
python -m pip install -r requirements.txt
python experiment.py --text "Please reset my password" "Where is my package?"
python -m unittest -v
```
Python 3.10+ recommended. Runs on CPU with no API credentials. Results are saved to `results/metrics.json`. Use `--data requests.jsonl` to supply your own labeled data.

## Experiment design
The bundled JSONL contains 240 synthetic requests in four intents: billing, technical, account, delivery. Each record has `text`, `label`, and a phrasing `family`. A seeded group split holds out entire phrasing families. TF-IDF vocabulary, IDF weights, SVD, and classifiers are fit using training examples only. The same partition is used for both models. Report accuracy, macro F1, and a confusion matrix with explicit label order.

The split tests new wrappers around familiar issue descriptions. The underlying issue phrases appear in both partitions, so this is a deliberately easy demonstration, not evidence of generalization to new customer issues. SVD is a linear text representation, not an LLM hidden state, and classifier performance does not establish causal interpretability.

## Results and limits
The executed seed-42 run produced TF-IDF accuracy/macro F1 of 1.000/1.000 and SVD accuracy/macro F1 of 0.5625/0.5659. See `results/metrics.json` for the full locally executed run. Synthetic metrics are not production performance claims. A next experiment should use licensed real tickets, hold out issue families and customers, run multiple seeds, compare class-specific recall, and evaluate domain shift. Do not tune model choices repeatedly against the final test set.

The included tests check group isolation, metric ranges, confusion-matrix counts, and prediction outputs. Existing phrases can make this benchmark nearly perfect; the README records that limitation rather than interpreting the score as a deployable result.

## Provenance
Original code and synthetic examples created with AI coding assistance, inspired by general representation-learning concepts. No school assignment text, starter notebook, model weights, or course solutions are distributed. Review the implementation and describe your own changes accurately when using it in a portfolio. Code and bundled synthetic data use the MIT license.
