# SOSC314-Course-Project

## Research Question
How do different political institutions frame the policy problem of artificial 
intelligence regulation, and to what extent does institutional role predict the way AI 
is discussed?

## Data Source
The corpus is built entirely from CELLAR, the EU Publications Office repository behind EUR-Lex. Its public SPARQL endpoint provides metadata and its REST interface provides full text; neither requires registration and nothing is scraped. AI-related documents are identified by EuroVoc subject tag and by title, then filtered by a documented set of scope rules. Current inventory (January 2020 – August 2026): 193 in-scope documents — 82 Commission, 68 Parliament, 43 Council — out of 646 found.

The five conditions a document must meet: 	1.	Institution. Authored by the European Commission (including its DGs and executive agencies), the European Parliament (including its committees), or the Council of the EU. Other EU bodies — EESC, Committee of the Regions, EDPS, agencies — are out. 	2.	Period. Dated 1 January 2020 or later. Start chosen to capture the White Paper on AI (February 2020) as the opening of the current policy cycle. 	3.	Relevance. Indexed with the EuroVoc subject artificial intelligence, or “artificial intelligence” appears in the English title. Documents that merely mention AI in passing are not picked up, by design. 	4.	Genre. A document type in which the institution states or negotiates a position — communications, staff documents, proposals, resolutions, opinions, amendments, conclusions, presidency notes, and similar. Excluded: research publications, adopted legal acts, forwarding notes, administrative notices, minutes, factsheets. 	5.	Authorship is real. The listed institution must have written the text, not just transmitted it. Register copies of Commission documents attributed to the Council are out.

## Methods (planned):
The analysis moves from describing the language to measuring frames to explaining differences. We begin with descriptive text analysis — word and phrase frequencies per institution — followed by unsupervised topic modelling to identify recurring themes without defining them in advance, and distinctive-language analysis (Fightin’ Words) to find the vocabulary each institution uses disproportionately often. The direct test of our hypotheses is frame classification: every paragraph is assigned to one or more predefined policy frames (economic opportunity, fundamental rights, security, labour, consumer protection, innovation, regulatory governance), with the classifier validated against a hand-coded sample before its output is used. Similarity analysis then measures how linguistically close the institutions are and whether their language converges as legislation moves from proposal to adoption, and all results are compared across legislative stages and before/after the release of ChatGPT in November 2022 as a shift in public attention. As a robustness check, we train a classifier to predict the authoring institution from text alone; its accuracy indicates whether institutional differences are systematic and serves as a diagnostic, not a finding in itself.

## Team
- Tim Maurice Steiner (tms128@duke.edu)
- Michelle, Yuhan Qiu (yq122@duke.edu)

## Repo Structure
- `reports/` — weekly progress reports
- `notebooks/` — code
- `data/` — original and processed data

## Reuse and licence
EU institutional documents may be reused under Commission Decision 2011/833/EU with attribution. The repository stores identifiers and source URLs and rebuilds full texts from them rather than redistributing them. 
