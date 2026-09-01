# SOSC314-Course-Project

## Research Question
How do different political institutions frame the policy problem of artificial 
intelligence regulation, and to what extent does institutional role predict the way AI 
is discussed?

## Data Source
The corpus is built entirely from CELLAR, the EU Publications Office repository behind EUR-Lex. Its public SPARQL endpoint provides metadata and its REST interface provides full text; neither requires registration and nothing is scraped. AI-related documents are identified by EuroVoc subject tag and by title, then filtered by a documented set of scope rules. Current inventory (January 2020 – August 2026): 193 in-scope documents — 82 Commission, 68 Parliament, 43 Council — out of 646 found. 

## Methods (planned):
Descriptive text analysis → topic modelling → distinctive-language analysis (Fightin’ Words) → document and institution similarity → validated frame classification → change over time across legislative stages. Supervised classification of institution from text serves as a diagnostic of systematic linguistic difference, not as a headline result.

## Team
- Tim Maurice Steiner
- Michelle (Yuhan Qiu)

## Repo Structure
- `reports/` — weekly progress reports
- `notebooks/` — code
- `data/` — original and processed data
