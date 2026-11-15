# PDF RAG CLI

Ask questions about a text-based PDF. The app extracts text with `pypdf`, ranks overlapping passages with local TF-IDF, and sends up to five matching passages to a configured language model. Answer citations refer to the PDF viewer's page numbers, starting at 1.

## Run locally

Run these commands from the repository root in Bash:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.main --retrieve-only "/path/to/1706.03762v7.pdf"
```

Use your own PDF, or [download the example paper](https://arxiv.org/pdf/1706.03762v7) and place it at `data/1706.03762v7.pdf` and omit the path. Try asking `What architecture does the paper propose?` and look for the Transformer in an excerpt from page 1. Type `exit` to quit.

## Enable generated answers

Set your model provider's key in the same terminal and run the CLI without `--retrieve-only`. These commands work in Bash and keep the key out of your shell history.

```bash
read -rsp "API key: " LLM_API_KEY
echo
export LLM_API_KEY
python -m src.main "/path/to/1706.03762v7.pdf"
```

The default endpoint is OpenAI's `https://api.openai.com/v1` and the default model is `gpt-4o-mini`. Set `LLM_BASE_URL` and `LLM_MODEL` for another OpenAI-compatible Chat Completions provider or model. A generated answer sends your question and retrieved PDF excerpts to that provider. With no key, the CLI displays excerpts locally.

TF-IDF ranks passages by word overlap. If a question uses different terms than the PDF, try wording closer to the document. A question with zero matches gives a fallback locally. When a question shares words with the PDF but the retrieved passages lack the answer, the model is instructed to report insufficient evidence.
