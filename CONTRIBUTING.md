# Developer rules

For everyone who changes this repository, human or AI. AI agents get their working
instructions from [AGENTS.md](AGENTS.md); the rules here bind them too.

## The one rule above the others

**Keep the code base clean and decluttered.** Every other rule here is read in that light.
When a rule and this one seem to disagree, this one wins, and the other rule should be fixed.

## Changing code

- Results must not change by accident. `tests/test_golden.py` pins seeded designs; a change
  that is not meant to alter results keeps it passing, and one that is regenerates the
  fixtures with `tools/make_golden.py` in the same commit and says so.
- Run `pytest -m "not slow"` before every commit, and the full suite (set
  `CRIMM_ML_TEST_MODEL`) for anything near the sampler or the model.
- Quote timings only from a dedicated machine, and say which.
- One concern per commit. A fix does not ride along with a rename.
- Code derived from another project is credited in `NOTICE` with that project's licence, in
  the same commit that adds it.

## Credit and authorship

*Decided by the owner, Truman Xu, on 2026-10-06:* AI models that write code for this project
are credited as authors, alongside the people who do. The owner delegated the detailed rules
to the first AI contributor, with one condition, the rule at the top of this page. These are
the rules. They apply equally to people and to AI models unless a rule says otherwise.

1. **Credit goes in three places, and nowhere else.**
   - **The commit.** Every commit carries its authors. An AI model that wrote a substantial
     part of a commit signs it with a trailer naming the exact model:
     `Co-Authored-By: <model name and version> <vendor's no-reply address>`, for example
     `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. This is the record of who
     wrote what; `git log` and `git blame` answer the question for any file or line.
   - **`AUTHORS.md`.** One entry per person, and one per AI model *family* (not per version,
     not per session). Add your entry with your first merged commit. Extend your list of
     areas only when you start work in a new one.
   - **The package metadata** (`authors` in `pyproject.toml`). The owner decides who is
     listed there. Nobody adds themselves.
2. **Source files carry no signatures.** No author lines, no "written by" or "generated
   with" banners, no model names, session notes, dates or emoji in code, comments,
   docstrings, test names or file names. A comment explains the code; the commit says who
   wrote it. A reviewer who finds a signature in a file removes it.
3. **Credit what landed.** A commit earns credit when it is merged. Plans, reviews,
   suggestions and reverted work are acknowledged in the pull request or the handoff log, not
   in `AUTHORS.md`.
4. **Be exact about what is yours.** Do not sign for code you ported, vendored or generated
   from another project as if it were original: say "port of", "derived from" or "generated
   with", and put the upstream project in `NOTICE`. Do not sign for a model you are not.
5. **Never remove or rewrite someone else's credit**, in a commit, in `AUTHORS.md` or in
   `NOTICE`. If an entry is wrong, raise it with the owner.
6. **Credit is not copyright, and not responsibility.** The copyright line in `NOTICE` is the
   owner's to set. Whoever merges a change answers for it, whatever wrote it; an AI
   signature is a statement of origin, so that readers know how the code came to be, and is
   never a reason to review it less.
