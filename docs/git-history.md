# Git history: private data audit

**Status: resolved.** The history on GitHub contains no private PDFs and no generated index files.

## What happened

Early commits tracked the private corpus (`my_pdfs/`, copyrighted lecture material and a schedule
with personal names) and the generated FAISS index (`vector_db/`, which embeds the full chunk text).
The history was rewritten with [`git-filter-repo`](https://github.com/newren/git-filter-repo) and
force-pushed; the local clone that did this still has the `.git/filter-repo/` bookkeeping
directory. Commit hashes changed as a result, so any hash quoted in older notes (for example
`a68d480`) no longer exists.

## How to verify (anyone, any time)

```bash
git clone https://github.com/mmkaya-ui/kulliyatai.git && cd kulliyatai
python scripts/audit_repo.py
```

The script walks every object reachable from every ref and fails if it finds a path under
`my_pdfs/`, `vector_db*/` or `.env`, or any blob over 200 kB other than the bundled public-domain
PDFs and the font. `tests/test_repo_audit.py` runs the same check, and CI checks out the full
history (`fetch-depth: 0`) so it covers every commit.

## If you cloned before the rewrite

An old clone still contains the private blobs. Delete it and clone again; do not push from it, or
the old objects come back.
