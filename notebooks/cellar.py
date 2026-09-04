"""Helpers for querying the EU Publications Office CELLAR repository.

CELLAR exposes (a) a public SPARQL endpoint over the CDM ontology for metadata,
and (b) a REST interface for retrieving document content by identifier.
Neither requires registration.
"""
from __future__ import annotations

import io

import pandas as pd
import requests

SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"
AUTHORITY = "http://publications.europa.eu/resource/authority/"
EUROVOC_AI = "http://eurovoc.europa.eu/3030"  # EuroVoc concept: artificial intelligence
LANG_ENG = AUTHORITY + "language/ENG"

PREFIXES = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def sparql(query: str, timeout: int = 180) -> pd.DataFrame:
    """Run a SELECT query against CELLAR and return a DataFrame."""
    r = requests.get(
        SPARQL,
        params={"query": PREFIXES + query},
        headers={"Accept": "text/csv"},
        timeout=timeout,
    )
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    for c in df.columns:
        if pd.api.types.is_string_dtype(df[c]):
            df[c] = df[c].str.replace(AUTHORITY, "", regex=False)
    return df


def works_by_eurovoc(concept: str = EUROVOC_AI, since: str = "2020-01-01") -> pd.DataFrame:
    """All works tagged with a EuroVoc concept: one row per (work, agent, type, title)."""
    q = f"""
    SELECT DISTINCT ?work ?agent ?type ?date ?title WHERE {{
      ?work cdm:work_is_about_concept_eurovoc <{concept}> ;
            cdm:work_created_by_agent ?agent ;
            cdm:work_date_document ?date .
      OPTIONAL {{ ?work cdm:work_has_resource-type ?type }}
      OPTIONAL {{ ?e cdm:expression_belongs_to_work ?work ;
                    cdm:expression_uses_language <{LANG_ENG}> ;
                    cdm:expression_title ?title }}
      FILTER(?date >= "{since}"^^xsd:date)
    }}"""
    return sparql(q)


def works_by_title(phrase: str = "artificial intelligence", since: str = "2020-01-01") -> pd.DataFrame:
    """Works whose English title contains a phrase (case-insensitive)."""
    q = f"""
    SELECT DISTINCT ?work ?agent ?type ?date ?title WHERE {{
      ?e cdm:expression_belongs_to_work ?work ;
         cdm:expression_uses_language <{LANG_ENG}> ;
         cdm:expression_title ?title .
      ?work cdm:work_created_by_agent ?agent ;
            cdm:work_date_document ?date .
      OPTIONAL {{ ?work cdm:work_has_resource-type ?type }}
      FILTER(?date >= "{since}"^^xsd:date)
      FILTER(CONTAINS(LCASE(STR(?title)), "{phrase.lower()}"))
    }}"""
    return sparql(q)


def fetch_content(work_uri: str, accept: str = "application/xhtml+xml", lang: str = "eng") -> bytes:
    """Retrieve a document's content from CELLAR REST by work URI (content negotiation)."""
    r = requests.get(work_uri, headers={"Accept": accept, "Accept-Language": lang}, timeout=120)
    r.raise_for_status()
    return r.content


# Corporate-body codes attributed to the Commission family (services, DGs, agencies).
COMMISSION_BODIES = {
    "COM", "SG", "SJ", "CNECT", "RTD", "COMP", "JUST", "EMPL", "GROW", "HOME", "TRADE",
    "SANTE", "ENER", "MOVE", "EAC", "ECFIN", "FISMA", "AGRI", "ENV", "CLIMA", "DEFIS",
    "DIGIT", "JRC", "EISMEA", "EASME", "HADEA", "REA", "ERCEA", "CINEA", "EACEA", "ESTAT",
    "OLAF", "TAXUD", "MARE", "REGIO", "INTPA", "NEAR", "ECHO", "BUDG", "HR", "COMM", "OIB",
}


def institution(agent_code: str) -> str | None:
    """Map a corporate-body code to one of the three institutions, or None."""
    code = str(agent_code).split("/")[-1]
    if code == "CONSIL":
        return "Council"
    if code == "EP" or code.startswith("EP_"):
        return "Parliament"
    if code in COMMISSION_BODIES:
        return "Commission"
    return None
