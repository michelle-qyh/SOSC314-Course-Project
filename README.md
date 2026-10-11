# Who Defines the Problem?

**A computational analysis of how EU institutions frame artificial intelligence regulation**

SOSC314 course project, Duke Kunshan University, Fall 2026 (Prof. Markus Neumann)

## Team

- Tim Maurice Steiner (tms128@duke.edu)
- Yuhan (Michelle) Qiu (yq122@duke.edu)

## Research Question

How do different political institutions frame the policy problem of artificial intelligence regulation, and to what extent does institutional role predict the way AI is discussed?

We compare the three EU institutions that wrote about the same legislative file, the AI Act: the **European Commission** (proposes legislation), the **European Parliament** (directly elected co-legislator) and the **Council of the EU** (member-state governments, co-legislator).

## Main Findings

Most first-round differences between the institutions did not survive testing. Two did:

| Finding | Method | Estimate (95% bootstrap interval) |
|---|---|---|
| The Commission gives security more weight than the other two institutions | Frame dictionary | Commission 8.0% [6.0–10.3] vs Parliament 3.7% [2.1–5.6] and Council 3.1% [1.6–4.7] |
| The Council writes far more about the AI Act's high-risk rules | LDA topic model | Council 19.9% [11.6–29.1] vs Parliament 5.8% [2.8–9.8] |

| Hypothesis | Verdict |
|---|---|
| H1: institutions differ in their frame profiles | Partly supported: two robust differences |
| H2: Commission emphasises governance, economy, technology | Not supported: lowest on governance; its other leads overlap with the Parliament |
| H3: Parliament emphasises rights and consumer protection | Direction as expected, but not robust |
| H4: Council emphasises security and competitiveness | Contradicted: Council lowest on security; the Commission leads |

The surviving patterns fit institutional role and document genre, that is, *what* each institution writes. They do not establish institutional preferences. The Council result is a difference in content (it writes the legal text of the high-risk rules), not evidence of a governance frame.

## Data

**Sources.** All documents come from official EU repositories; nothing is scraped.

- **CELLAR**, the EU Publications Office repository behind EUR-Lex: SPARQL endpoint for metadata, REST interface for full texts.
- **Council public register**: 9 Council documents that CELLAR lists but does not host.
- **European Parliament Open Data Portal**: 5 Parliament adopted texts, matched to CELLAR records by date and EuroVoc subject tag.

**Scope rules.** A document is in scope if it meets all five conditions (full rules with reasons: `data/doc_types.csv`):

1. **Institution.** Authored by the Commission (including its DGs and executive agencies), the Parliament (including its committees) or the Council; other EU bodies are out.
2. **Period.** Dated 1 January 2020 or later, starting with the White Paper on AI (February 2020).
3. **Relevance.** Tagged with the EuroVoc subject *artificial intelligence* (concept 3030), or "artificial intelligence" in the English title.
4. **Genre.** A document type in which the institution states or negotiates a position (communications, staff documents, proposals, resolutions, opinions, amendments, conclusions, presidency notes). Excluded: research publications, adopted legal acts, forwarding notes, administrative notices, minutes, factsheets.
5. **Real authorship.** The listed institution wrote the text rather than transmitting it; register copies and co-signed acts are out.

**From inventory to corpus.**

| Stage | Commission | Parliament | Council | Total |
|---|---|---|---|---|
| In scope (inventory of 4 Sep 2026, after manual check of ambiguous types) | 78 | 69 | 39 | 186 |
| Full text retrieved | 78 | 38 | 35 | 151 |
| Analysis corpus (after relevance screen) | 75 | 36 | 28 | **139** |
| Share of in-scope documents kept | 96% | 52% | 72% | |

The inventory query found 647 AI-related documents, of which 191 passed the type-level scope rules; a manual check then excluded 5 more (2 programme evaluations in which AI is peripheral, 3 Council transmission notes). Most Parliament losses are committee opinions and amendment lists that CELLAR records but does not host, and that the Parliament website does not serve to automated requests; we did not circumvent this. The relevance screen drops documents with fewer than 0.5 explicit AI mentions per 1,000 words or fewer than 200 words.

The analysis corpus covers **139 documents, 27,732 paragraphs and 2.04 million words** (February 2020 to August 2026). CELLAR is continuously indexed, so counts drift slightly between query dates; the committed snapshot in `data/inventory/` is the reference.

**Units.** The paragraph is the unit of measurement (median 49 words); scores are averaged to documents and then to institutions, so every document counts equally regardless of length.

## Methods

Three routes to the same question, each failing in a different way:

1. **Fightin' Words (baseline).** Log-odds ratios with an informative Dirichlet prior (Monroe, Colaresi & Quinn 2008) identify the words that separate two institutions; we keep the top 20 per side by z-score. Run under seven text representations (stopword lists, bigrams, TF-IDF, minimum document frequency).
2. **Seven-frame dictionary (main measure).** The seven frames from the proposal (economic opportunity, fundamental rights, security/geopolitics, labour and social, consumer protection, technological innovation, regulatory governance), defined by 152 terms. Score = frame words per 100 words per paragraph; reported as each frame's share of the institution's total frame signal (the seven shares sum to 100).
3. **Topic models (cross-check).** LDA on raw counts and NMF on TF-IDF over 25,762 paragraphs and a 4,000-term vocabulary; K = 10 chosen from {5, 8, 10, 12} by a coherence score. No categories are imposed; topics are labelled after fitting.

**Diagnostics (Week 5).**

- *Uncertainty:* 1,000 bootstrap resamples of documents within each institution; a lead counts only if its 95% interval does not overlap the others'.
- *Term influence:* each dictionary term removed in turn; a tightened dictionary without 11 generic terms is reported alongside the original.
- *Term overlap* between frames.
- *Sensitivity:* each representation choice measured against its own reference.
- *Topic stability:* six random seeds, K = 20, harmonised stopwords, paragraph- vs document-weighting.
- *Face validity:* the 10 highest-scoring paragraphs per frame read by one coder (6 of 7 frames pass; technological innovation fails).
- *Inherited text:* paragraphs copied between institutions (17.2% of Council paragraphs); removing them moves no frame share by more than 0.5 percentage points.

**Not done.** The proposal also planned similarity analysis between institutions and change over time. We dropped both: the Council has only 5 documents before 2024, too few to compare periods, and the time went into testing whether the main differences hold. Frame classification against a hand-coded sample, also planned, remains the most important next step.

## Repository Structure

```
SOSC314-Course-Project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── corporate_body_authority_table.csv     # Publications Office authority table (institution codes)
│   ├── doc_types.csv                          # scope rules: document type → in/out, with reasons
│   ├── doc_types.xlsx                         # formatted view of the same table
│   ├── white_paper_2020.html                  # sample document from the first retrieval test
│   ├── white_paper_2020.pdf
│   ├── inventory/                             # committed query snapshots (retrieval of 4 Sep 2026)
│   │   ├── works_eurovoc.csv                  #   works found by EuroVoc tag
│   │   ├── works_title.csv                    #   works found by title
│   │   └── works_union.csv                    #   union of both, with scope decision
│   ├── corpus/                                # the analysis corpus
│   │   ├── documents.parquet                  #   139 documents, one row each (with cleaned text)
│   │   ├── paragraphs.parquet                 #   27,732 paragraphs, one row each
│   │   ├── retrieval_log.csv                  #   retrieval outcome for all 186 in-scope documents
│   │   ├── exclusions.csv                     #   documents dropped at screening, with reasons
│   │   ├── format_availability.csv            #   which CELLAR format each document offers
│   │   └── threshold_sensitivity.csv          #   sensitivity of the screening thresholds
│   ├── analysis/                              # measurement and diagnostic results
│   │   ├── frame_profile.csv                  #   frame shares by institution
│   │   ├── frame_profile_formal_position.csv  #   the same, restricted to one genre
│   │   ├── frame_profile_tightened.csv        #   the same, with 11 generic terms removed
│   │   ├── frame_profile_inherited_removed.csv #  the same, without cross-institution duplicates
│   │   ├── bootstrap_frame_profile.csv        #   95% intervals from 1,000 document resamples
│   │   ├── distinctive_terms.csv              #   Fightin' Words top terms, all specifications
│   │   ├── fw_stability.csv                   #   term turnover against the baseline
│   │   ├── sensitivity_disentangled.csv       #   each choice against its own reference
│   │   ├── leave_one_term_out.csv             #   influence of each dictionary term
│   │   ├── frame_overlap.csv                  #   terms shared between frames
│   │   ├── face_validity_summary.csv          #   verdict per frame
│   │   ├── face_validity_top_paragraphs.csv   #   the 70 paragraphs read
│   │   └── inherited_text_by_institution.csv  #   paragraphs shared verbatim across institutions
│   └── exploration/                           # working files of the Week 2 exploration
│       ├── doc_types_crosscheck.csv
│       ├── raw_works_2026-09-05.csv
│       ├── raw_works_2026-09-06.csv
│       └── figures/week2_combined.png
├── docs/                                      # project website (GitHub Pages)
│   ├── index.html                             #   single-page site
│   ├── .nojekyll                              #   serve files as they are
│   └── figures/                               #   figures used on the site (from reports/figures/)
│       ├── corpus_by_year.png
│       ├── corpus_attrition.png
│       ├── frame_profile.png                  #   panel (a) of figure1_operationalisation_week4.png
│       ├── bootstrap.png                      #   panel (a) of figure1_diagnostics_week5.png
│       ├── what_moves.png                     #   panel (b) of figure1_diagnostics_week5.png
│       ├── topics_bootstrap.png
│       └── face_validity.png
├── notebooks/
│   ├── cellar.py                              # module: CELLAR API helpers (SPARQL + REST)
│   ├── collect.py                             # module: manifestation resolution, retrieval, text extraction
│   ├── ep_portal.py                           # module: Parliament Open Data Portal recovery
│   ├── build_corpus.py                        # module: cleaning, segmentation, relevance screen
│   ├── represent.py                           # module: tokenisation and representation specifications
│   ├── measures.py                            # module: frame dictionary and Fightin' Words
│   ├── diagnostics.py                         # module: term influence, bootstrap, sensitivity
│   ├── 01_inventory.py                        # script: query → scope filter → counts (Week 2)
│   ├── 02_build_corpus.py / .ipynb            # script: inventory → analysis corpus (Week 3)
│   ├── 03_operationalisation.py / .ipynb      # script: frame scoring, distinctive terms (Week 4)
│   ├── 05_dictionary_diagnostics.py / .ipynb  # script: dictionary and Fightin' Words diagnostics (Week 5)
│   ├── 06_face_validity.py / .ipynb           # script: face validity and inherited text (Week 5)
│   ├── 07_figures.py / .ipynb                 # script: report and presentation figures from data/ (final)
│   ├── 01_feasibility_test_and_initial_exploration.ipynb      # Week 2 exploration
│   ├── 03_crosscheck_and_descriptive_stats_v1_pre_pipeline_fix.ipynb   # Week 3 coverage audit
│   ├── 03_crosscheck_and_descriptive_stats_v2_post_pipeline_fix.ipynb  # Week 3 descriptive statistics
│   ├── 04_topic_modeling.ipynb                # LDA and NMF (Week 4)
│   └── 05_topic_diagnostics.ipynb             # topic stability: seeds, K, weighting, stopwords, bootstrap (Week 5)
└── reports/
    ├── [SOSC314] Week_2_Progress_Report_Tim_Michelle.pdf
    ├── [SOSC314] Week_3_Progress_Report_Tim_Michelle.pdf
    ├── [SOSC314] Week_4_Progress_Report_Tim_Michelle.pdf
    ├── [SOSC314] Week_5_Progress_Report_Tim_Michelle.pdf
    └── figures/
        ├── figure1_pipeline_week2.png             # diagram (drawn, not generated by code)
        ├── figure2_corpus_inventory_week2.png     # 01_inventory.py
        ├── figure_institution_mapping.png         # diagram (drawn, not generated by code)
        ├── figure_retrieval_process_week3.png     # diagram (drawn, not generated by code)
        ├── figure2_corpus_attrition_week3.png     # 03_crosscheck_..._v1 notebook
        ├── figure3_doc_length_week3.png           # 03_crosscheck_..._v1 notebook
        ├── figure4_paragraphs_density_week3.png   # 03_crosscheck_..._v1 notebook
        ├── figure1_operationalisation_week4.png   # 07_figures.py
        ├── figure2_topic_models_week4.png         # 04_topic_modeling.ipynb
        ├── figure1_diagnostics_week5.png          # 07_figures.py
        ├── figure2_topic_diagnostics_week5.png    # 05_topic_diagnostics.ipynb
        ├── figure3_face_validity_week5.png        # 07_figures.py
        ├── figure_corpus_by_year_final.png        # 07_figures.py (presentation)
        └── figure_corpus_attrition_final.png      # 07_figures.py (presentation)
```

The `.py` scripts are the source of truth; the paired `.ipynb` files are notebook views of the same code (via `jupytext`). The modules are imported by the numbered scripts, are not run directly, and must keep their filenames.

## Reproduce

Requires Python 3.10+ and, for PDF text extraction during corpus building, the `pdftotext` command-line tool (part of Poppler: `brew install poppler` on macOS, `apt install poppler-utils` on Debian/Ubuntu).

```bash
pip install -r requirements.txt

# corpus (network access to CELLAR needed; skip to use the committed corpus)
python notebooks/01_inventory.py               # inventory (cached snapshots; set REFRESH = True to re-query CELLAR)
python notebooks/02_build_corpus.py            # corpus (retrieval checkpointed in data/raw/; ~3 min cold run)

# analysis (runs offline from the committed corpus, ~2 min in total)
python notebooks/03_operationalisation.py      # frame scoring, distinctive terms, approach comparison
python notebooks/05_dictionary_diagnostics.py  # term influence, overlap, bootstrap, sensitivity
python notebooks/06_face_validity.py           # face validity and inherited-text measurement
python notebooks/07_figures.py                 # report and presentation figures (EB Garamond if installed)
```

The topic models run as notebooks: `04_topic_modeling.ipynb`, then `05_topic_diagnostics.ipynb` (the six-seed stability check takes about 25 minutes).

## Reuse and Licence

EU institutional documents are © European Union and may be reused with attribution under Commission Decision 2011/833/EU. `data/corpus/` contains the cleaned full texts of the 139 documents together with their CELLAR identifiers, so every text can be traced to its source. The code in this repository may be reused with attribution to the authors.

## Useful Links

**Data infrastructure**
- [CELLAR SPARQL endpoint](https://publications.europa.eu/webapi/rdf/sparql): the public query interface used by `cellar.py`
- [Publications Office: machine access to EU law](https://eur-lex.europa.eu/content/help/data-reuse/webservice.html): EUR-Lex documentation on programmatic access
- [CELLAR / CDM ontology documentation](https://op.europa.eu/en/web/eu-vocabularies/cellar): the data model behind our queries (works, expressions, agents)
- [EuroVoc concept "artificial intelligence" (3030)](https://op.europa.eu/en/web/eu-vocabularies/concept/-/resource?uri=http://eurovoc.europa.eu/3030): the subject tag used as our first retrieval criterion
- [European Parliament Open Data Portal](https://data.europarl.europa.eu/): source of the 5 recovered Parliament texts (`ep_portal.py`)
- [Council public register](https://www.consilium.europa.eu/en/documents-publications/public-register/): source of the 9 Council documents not hosted in CELLAR

**The legislative file**
- [AI Act procedure file (2021/0106(COD))](https://oeil.secure.europarl.europa.eu/oeil/en/procedure-file?reference=2021/0106(COD)): the full legislative history in Parliament's Legislative Observatory
- [Regulation (EU) 2024/1689 (AI Act)](https://eur-lex.europa.eu/eli/reg/2024/1689/oj/eng): the adopted act on EUR-Lex

**Reuse**
- [Commission Decision 2011/833/EU](https://eur-lex.europa.eu/eli/dec/2011/833/oj): the legal basis for reusing EU documents

## References

**Framing and problem definition**
- Entman, R. M. (1993). Framing: Toward Clarification of a Fractured Paradigm. *Journal of Communication*, 43(4), 51–58.
- Rochefort, D. A., & Cobb, R. W. (eds.) (1994). *The Politics of Problem Definition: Shaping the Policy Agenda*. University Press of Kansas.

**Computational text analysis**
- Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). Latent Dirichlet Allocation. *Journal of Machine Learning Research*, 3, 993–1022.
- Efron, B., & Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Grimmer, J., & Stewart, B. M. (2013). Text as Data: The Promise and Pitfalls of Automatic Content Analysis Methods for Political Texts. *Political Analysis*, 21(3), 267–297.
- Lee, D. D., & Seung, H. S. (1999). Learning the Parts of Objects by Non-negative Matrix Factorization. *Nature*, 401, 788–791.
- Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' Words: Lexical Feature Selection and Evaluation for Identifying the Content of Political Conflict. *Political Analysis*, 16(4), 372–403.
- Roberts, M. E., Stewart, B. M., Tingley, D., et al. (2014). Structural Topic Models for Open-Ended Survey Responses. *American Journal of Political Science*, 58(4), 1064–1082.

**EU institutions and AI regulation**
- Hix, S., & Høyland, B. (2022). *The Political System of the European Union* (5th ed.). Bloomsbury.
- Princen, S. (2009). *Agenda-Setting in the European Union*. Palgrave Macmillan.
- Veale, M., & Zuiderveen Borgesius, F. (2021). Demystifying the Draft EU Artificial Intelligence Act. *Computer Law Review International*, 22(4), 97–112.

## AI Transparency

- ChatGPT 5.6 Luna was used to update the repository structure tree in this README.
- Claude Fable 5.1 was used to detect coding errors and get code recommendations in `notebooks/01_inventory.py`, `notebooks/cellar.py`, `notebooks/collect.py` and `notebooks/build_corpus.py`.
