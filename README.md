# SOSC314-Course-Project

## Research Question
How do different political institutions frame the policy problem of artificial 
intelligence regulation, and to what extent does institutional role predict the way AI 
is discussed?

## Data Source
The corpus is built entirely from CELLAR, the EU Publications Office repository behind EUR-Lex. Its public SPARQL endpoint provides metadata and its REST interface provides full text; neither requires registration and nothing is scraped. AI-related documents are identified by EuroVoc subject tag and by title, then filtered by a documented set of scope rules. Current inventory (January 2020 – August 2026): 193 in-scope documents — 82 Commission, 68 Parliament, 43 Council — out of 646 found.

The five conditions a document must meet: 	1.	Institution. Authored by the European Commission (including its DGs and executive agencies), the European Parliament (including its committees), or the Council of the EU. Other EU bodies — EESC, Committee of the Regions, EDPS, agencies — are out. 	2.	Period. Dated 1 January 2020 or later. Start chosen to capture the White Paper on AI (February 2020) as the opening of the current policy cycle. 	3.	Relevance. Indexed with the EuroVoc subject artificial intelligence, or “artificial intelligence” appears in the English title. Documents that merely mention AI in passing are not picked up, by design. 	4.	Genre. A document type in which the institution states or negotiates a position — communications, staff documents, proposals, resolutions, opinions, amendments, conclusions, presidency notes, and similar. Excluded: research publications, adopted legal acts, forwarding notes, administrative notices, minutes, factsheets. 	5.	Authorship is real. The listed institution must have written the text, not just transmitted it. Register copies of Commission documents attributed to the Council are out.

## Methods (planned):
Descriptive text analysis → topic modelling → distinctive-language analysis (Fightin’ Words) → document and institution similarity → validated frame classification → change over time across legislative stages. Supervised classification of institution from text serves as a diagnostic of systematic linguistic difference, not as a headline result.

## Team
- Tim Maurice Steiner
- Michelle (Yuhan Qiu)

## Repo Structure
- `reports/` — weekly progress reports
- `notebooks/` — code
- `data/` — original and processed data
