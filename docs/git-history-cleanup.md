# Git History Cleanup Options & Trade-Offs

## Background

In commit `a68d480`, we untracked `my_pdfs/` (89 private/copyrighted files) and `vector_db/` (24 MB FAISS index containing embedded chunk text) from the working tree and added them to `.gitignore`.

However, the historical git commits still contain these blobs. The repository `.git` pack size is currently **~53 MB**.

Per repository guidelines, **git history was NOT rewritten**. If you decide to permanently purge these files from past commits, here are the two standard options and their trade-offs:

---

## Option 1: `git-filter-repo` (Recommended)

[`git-filter-repo`](https://github.com/newren/git-filter-repo) is the official tool recommended by the Git project, replacing the deprecated `git filter-branch`.

### How to run:
```bash
# 1. Install via pip
pip install git-filter-repo

# 2. Make a backup clone first!
git clone --mirror . ../kulliyatai-backup.git

# 3. Strip my_pdfs/ and vector_db/ from all historical commits
git filter-repo --path my_pdfs --path vector_db --invert-paths

# 4. Force push to remote (WARNING: destructive to remote branches)
git push origin --force --all
git push origin --force --tags
```

### Trade-offs:
- **Pros:**
  - Written in Python, thoroughly tested, handles tag and branch rewriting cleanly.
  - Automatically prunes empty commits and cleans up reflogs.
  - Leaves a clean, minimal repo (~1-2 MB instead of 53 MB).
- **Cons:**
  - **Rewrites all commit hashes** after the first commit where those files were introduced (`77db031` and earlier).
  - Anyone collaborating on this repository must delete their clone and re-clone, as rebasing old commits will reintroduce the old objects.
  - Modifies commit author dates and signatures if any were signed.

---

## Option 2: BFG Repo-Cleaner

[`BFG Repo-Cleaner`](https://rtyley.github.io/bfg-repo-cleaner/) is a fast, Scala/Java-based alternative specifically designed for stripping large files or sensitive data.

### How to run:
```bash
# 1. Download bfg.jar (requires Java runtime)
# 2. Make a bare backup clone
git clone --mirror . ../kulliyatai-bare.git
cd ../kulliyatai-bare.git

# 3. Purge folders from history
java -jar bfg.jar --delete-folders "{my_pdfs,vector_db}"

# 4. Expire reflog and prune packfiles
git reflog expire --expire=now --all && git gc --prune=now --aggressive

# 5. Force push to remote
git push --force
```

### Trade-offs:
- **Pros:**
  - Extremely fast on very large repositories.
  - Simpler command syntax for simple file/folder deletions.
- **Cons:**
  - Requires Java JRE installed on your system.
  - Protects `HEAD` by default (requires you to untrack files first, which we already did).
  - Also rewrites all commit SHAs across history.

---

## Summary Recommendation
- If this repository is only used by you or you can coordinate a fresh clone: use **`git-filter-repo`** when ready to publish publicly.
- If you prefer to avoid force-pushing and hash-invalidation right now: keep the current state (files are untracked and gitignored for all future commits).
