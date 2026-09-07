# HSBC Revision-3 public-release policy v1

Status: **FAIL-CLOSED SANITIZATION CONTRACT**  
Owner: Ha Cong Nguyen  
Date: 2026-09-08

The public release is built into a fresh directory and contains only files
selected by `scripts/hsbc_challenge/build_revision3_public_release_v1.py`.
The existing `full_quantum_eigensolver` repository remains private because its
tracked history contains IEEE-CIS-derived models, preprocessors and arrays.

## Allowed classes

- proposal/theory/audit prose and PDFs;
- source code containing no embedded row data;
- aggregate-only JSON already classified as public-candidate or used for the
  proposal's printed aggregate results, after removing private provenance,
  row tokens, and references to absent row/model payloads; and
- a generated manifest, citation file and public-bundle verifier.

## Forbidden classes

The public tree must contain no file with a data/model suffix including
`.npz`, `.npy`, `.joblib`, `.parquet`, `.csv`, `.pkl`, `.pickle`, or `.model`;
no auxiliary LaTeX build files; no Git history from the private monorepo; and
no released JSON carrying a private classification or private provenance
field.

Internal provenance is stripped from public aggregate JSON because it names or
hashes private row-bearing files. Scientific result values are not changed.
The public manifest lists included files only. It records excluded *classes*,
not the private files or their hashes.

## Release gates

1. build into an empty destination;
2. require every allow-listed source to exist;
3. reject forbidden suffixes, symlinks, oversized files, absolute local paths,
   private classifications, private provenance fields, and row/model payload
   references inside released JSON;
4. hash every included file;
5. run the public verifier in the built tree;
6. inspect the proposed Git index before commit;
7. publish only the fresh sanitized repository, never the private monorepo;
8. archive the exact tagged source bundle on Zenodo; and
9. compare GitHub-release and Zenodo-upload SHA-256 values before publication.
