import arxiv
import json
import os
from tqdm import tqdm

SAVE_PATH = "data/metadata"
OUTPUT_FILE = os.path.join(
    SAVE_PATH,
    "papers_metadata.json"
)

os.makedirs(SAVE_PATH, exist_ok=True)

SEARCH_QUERIES = [
    "all:agent",
    "all:agentic",
    'all:"multi-agent"',
    'all:"llm agent"',
    'all:"tool use"',
    "all:reflexion",
    'all:"self-rag"',
    'all:"web agent"',
    'all:"autonomous agent"',
    'all:"computer use"',
    'all:"agent memory"',
    'all:"reasoning agent"',
    'all:"deep research"'
]


def load_existing_metadata():

    if not os.path.exists(OUTPUT_FILE):
        return []

    try:
        with open(
            OUTPUT_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception:
        return []


def fetch_papers():

    papers = []

    client = arxiv.Client(
        page_size=25,
        delay_seconds=5,
        num_retries=5
    )

    seen_ids = set()

    for query in SEARCH_QUERIES:

        print(f"\n{'='*60}")
        print(f"Searching: {query}")
        print(f"{'='*60}")

        search = arxiv.Search(
            query=query,
            max_results=100,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )

        try:

            for result in tqdm(client.results(search)):

                if result.published.year < 2024:
                    continue

                paper_id = result.entry_id.split("/")[-1]

                if paper_id in seen_ids:
                    continue

                seen_ids.add(paper_id)

                paper = {
                    "arxiv_id": paper_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [
                        author.name
                        for author in result.authors
                    ],
                    "published": str(result.published),
                    "pdf_url": result.pdf_url,
                    "categories": result.categories
                }

                papers.append(paper)

        except Exception as e:

            print(
                f"\nError while processing "
                f"{query}: {e}"
            )

            continue

    return papers


def merge_papers(existing, new):

    merged = {}

    for paper in existing:
        merged[paper["arxiv_id"]] = paper

    for paper in new:
        merged[paper["arxiv_id"]] = paper

    return list(merged.values())


def save_metadata(papers):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            papers,
            file,
            indent=4,
            ensure_ascii=False
        )


if __name__ == "__main__":

    print("\nLoading existing papers...")

    existing_papers = load_existing_metadata()

    print(
        f"Existing papers: "
        f"{len(existing_papers)}"
    )

    print("\nFetching new papers...")

    new_papers = fetch_papers()

    print(
        f"\nNew papers collected: "
        f"{len(new_papers)}"
    )

    final_papers = merge_papers(
        existing_papers,
        new_papers
    )

    print(
        f"\nTotal unique papers: "
        f"{len(final_papers)}"
    )

    save_metadata(final_papers)

    print(
        "\nMetadata saved successfully to:"
    )

    print(OUTPUT_FILE)