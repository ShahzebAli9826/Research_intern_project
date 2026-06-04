import json

with open(
    "data/metadata/papers_metadata.json",
    "r",
    encoding="utf-8"
) as f:
    metadata = json.load(f)

extra = [

    {
        "arxiv_id": "2504.19413v1",
        "title": "OpenHands",
        "authors": []
    },

    {
        "arxiv_id": "2405.15793v3",
        "title": "SWE-agent",
        "authors": []
    },

    {
        "arxiv_id": "2407.16741v3",
        "title": "Mem0",
        "authors": []
    },

    {
        "arxiv_id": "2407.18901v1",
        "title": "OSWorld",
        "authors": []
    },

    {
        "arxiv_id": "2404.07972v2",
        "title": "AppWorld",
        "authors": []
    },

    {
        "arxiv_id": "2501.12326v1",
        "title": "UI-TARS",
        "authors": []
    },

    {
        "arxiv_id": "2509.02544v2",
        "title": "UI-TARS-2",
        "authors": []
    },

    {
        "arxiv_id": "1310.7922v2",
        "title": "Tau",
        "authors": []
    }
]

metadata.extend(extra)

with open(
    "data/metadata/papers_metadata.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print("done")