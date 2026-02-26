# Zenodo (Maintainer Notes)

This file is maintainer-facing guidance for software citation quality and visibility.

## Goals

- Make software citation easy and consistent for users
- Preserve reproducibility with versioned DOIs
- Improve the chance that Google Scholar recognizes citations

## What to keep consistent

- Project name: `edlgt`
- Citation title: `edlgt: Exact Diagonalization for Lattice Gauge Theories`
- Version string: same in `pyproject.toml`, `CITATION.cff`, release tag, and Zenodo record
- Author names / ORCID

## Zenodo DOI types (important)

- Concept DOI:
  - Stable DOI for the software as a whole
  - Use for general "we used edlgt" citations
- Version DOI:
  - DOI minted for a specific release (for example `v0.1.0`)
  - Use for exact reproducibility

## Release -> Zenodo workflow

1. Prepare a tagged GitHub release (for example `v0.1.0`)
2. Ensure `CITATION.cff` and `CHANGELOG.md` are up to date
3. Publish the GitHub release
4. Wait for Zenodo to archive it and mint a version DOI
5. Check the Zenodo record metadata carefully
6. Export Zenodo BibTeX and compare with `CITATION.bib`
7. Update `CITATION.cff` with DOI metadata (recommended)
8. Update README citation section if needed

## Zenodo metadata checklist

- Title is descriptive and matches the citation title
- Authors are complete and ordered correctly
- ORCID is attached where possible
- Version matches the GitHub tag and package version
- Release date is correct
- Repository URL is correct
- License is correct (`Apache-2.0`)
- Keywords are relevant and consistent
- Description clearly states the software purpose

## Scholar visibility checklist

- In your papers, cite the software in the REFERENCES section (not only in-text)
- Use a consistent citation string across papers
- Include authors, title, year, version, DOI, and URL
- Prefer a single canonical citation format (do not alternate many variants)
- If possible, also publish a citable software paper (for example JOSS)

## Why citations may not appear in Google Scholar

- Citation appears only in the main text / acknowledgements
- PDF/reference formatting is hard for Scholar to parse
- Software/Zenodo entries are indexed less reliably than journal articles
- Multiple inconsistent citation variants are not merged

## Recommended long-term improvement

Consider publishing a short software paper (e.g. JOSS) and asking users to cite:

- the software paper (for visibility and credit)
- the Zenodo version DOI (for reproducibility)

