"""Terminal interface for PDF question answering."""

import argparse
from pathlib import Path

from pypdf.errors import PyPdfError

from src.llm_client import NO_ANSWER, ModelError, ask, is_configured
from src.pdf_loader import extract_pages, split_into_chunks
from src.retriever import TFIDFRetriever


DEFAULT_PDF = Path(__file__).resolve().parents[1] / "data" / "1706.03762v7.pdf"
TOP_K = 5


def _print_passages(results: list[dict]) -> None:
    print("\nRetrieved PDF passages")
    for result in results:
        page = result["pages"][0]
        print(f"\n[p. {page}]  {result['text']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask questions about a PDF.")
    parser.add_argument("pdf", type=Path, nargs="?", default=DEFAULT_PDF)
    parser.add_argument("--retrieve-only", action="store_true", help="Show passages without calling an AI model.")
    args = parser.parse_args()

    try:
        pages = extract_pages(str(args.pdf))
    except (OSError, PyPdfError, ValueError) as error:
        parser.exit(1, f"Unable to read PDF {args.pdf}: {error}\n")
    if not pages:
        parser.exit(1, "The PDF contains no extractable text. A scanned PDF may need OCR first.\n")

    chunks = split_into_chunks(pages)
    retriever = TFIDFRetriever(chunks)
    print(f"Indexed {len(chunks)} passages from {len(pages)} PDF pages with text.")

    retrieve_only = args.retrieve_only or not is_configured()
    if retrieve_only:
        print("Retrieval mode. Showing PDF excerpts without a generated answer.")
        if not args.retrieve_only:
            print("Set LLM_API_KEY to enable generated answers.")
    print('Ask a question about the PDF. Type "exit" to quit.')

    while True:
        try:
            question = input("\nQ: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.casefold() == "exit":
            return

        results = retriever.retrieve(question, top_k=TOP_K)
        if not results:
            print(f"A: {NO_ANSWER}")
            continue
        if retrieve_only:
            _print_passages(results)
            continue

        try:
            print(f"A: {ask(question, results)}")
        except ModelError as error:
            print(f"Could not generate an answer. {error}")
            _print_passages(results)
        except KeyboardInterrupt:
            print()
            return


if __name__ == "__main__":
    main()
