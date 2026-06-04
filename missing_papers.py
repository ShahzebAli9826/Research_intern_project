import arxiv

papers = [
    "Mem0",
    "SWE-agent",
    "OpenHands",
    "OSWorld",
    "AppWorld",
    "UI-TARS",
    "UI-TARS-2",
    "Tau"
]

client = arxiv.Client()

for title in papers:

    search = arxiv.Search(
        query=title,
        max_results=5
    )

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for result in client.results(search):

        print("Title :", result.title)
        print("ID    :", result.entry_id)
        print()