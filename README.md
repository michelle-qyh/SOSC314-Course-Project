# SOSC314-Course-Project

## Research Question

How do different political institutions frame the policy problem of artificial intelligence regulation, and to what extent does institutional role predict the way AI is discussed?

## Data Source

The corpus is built entirely from CELLAR, the EU Publications Office repository behind EUR-Lex. Its public SPARQL endpoint provides metadata and its REST interface provides full text; neither requires registration and nothing is scraped. AI-related documents are identified by EuroVoc subject tag and by title, then filtered by a documented set of scope rules. Current inventory (retrieval of 4 September 2026): **191 in-scope documents — 80 Commission, 69 Parliament, 42 Council — out of 647 found.** CELLAR is continuously indexed, so counts drift slightly between days; the committed inventory in `data/inventory/` is the reference snapshot.

The conditions a document must meet: 1. **Institution.** Authored by the European Commission (including its DGs and executive agencies), the European Parliament (including its committees), or the Council of the EU; other EU bodies are out. 2. **Period.** Dated 1 January 2020 or later, capturing the White Paper on AI (February 2020) as the opening of the current policy cycle. 3. **Relevance.** Indexed with the EuroVoc subject *artificial intelligence*, or "artificial intelligence" in the English title. 4. **Genre.** A document type in which the institution states or negotiates a position — communications, staff documents, proposals, resolutions, opinions, amendments, conclusions, presidency notes; excluded: research publications, adopted legal acts, forwarding notes, administrative notices, minutes, factsheets. 5. **Authorship is real.** The listed institution must have written the text, not just transmitted it; register copies and works jointly attributed to several institutions (co-signed acts, joint declarations) are out. Full rules with reasons: `data/doc_types.csv`.

## Methods (planned)

The analysis moves from describing the language to measuring frames to explaining differences. We begin with descriptive text analysis — word and phrase frequencies per institution — followed by unsupervised topic modelling to identify recurring themes without defining them in advance, and distinctive-language analysis (Fightin' Words) to find the vocabulary each institution uses disproportionately often. The direct test of our hypotheses is frame classification: every paragraph is assigned to one or more predefined policy frames (economic opportunity, fundamental rights, security, labour, consumer protection, innovation, regulatory governance), with the classifier validated against a hand-coded sample before its output is used. Similarity analysis then measures how linguistically close the institutions are and whether their language converges as legislation moves from proposal to adoption, and all results are compared across legislative stages and before/after the release of ChatGPT in November 2022 as a shift in public attention. As a robustness check, we train a classifier to predict the authoring institution from text alone; its accuracy indicates whether institutional differences are systematic and serves as a diagnostic, not a finding in itself.

## Team

- Tim Maurice Steiner (tms128@duke.edu)
- Michelle, Yuhan Qiu (yq122@duke.edu)

## Repo Structure

```
SOSC314-Course-Project/
├── README.md
├── requirements.txt
├── data/
│   ├── doc_types.csv          # scope rules: type → in/out, with reasons
│   ├── doc_types.xlsx         # formatted view of the same table
│   ├── inventory/             # committed query snapshots (4 Sep 2026)
│   │   ├── works_eurovoc.csv
│   │   ├── works_title.csv
│   │   └── works_union.csv
│   ├── white_paper_2020.html  # retrieved sample document
│   └── white_paper_2020.pdf
├── notebooks/
│   ├── cellar.py              # CELLAR API helpers (SPARQL + REST)
│   ├── 01_inventory.py        # inventory pipeline → 647 found / 191 in scope
│   └── 01.Feasibility Test and Initial Exploration.ipynb
└── reports/
    └── [SOSC314] Week_2_Progress_Report_Tim_Michelle.pdf
```


## Reproduce

```bash
pip install -r requirements.txt
python notebooks/01_inventory.py   # uses cached snapshots in data/inventory/; set REFRESH = True to re-query CELLAR
```

## Reuse and licence

EU institutional documents may be reused under Commission Decision 2011/833/EU with attribution. The repository stores identifiers and source URLs and rebuilds full texts from them rather than redistributing them.
