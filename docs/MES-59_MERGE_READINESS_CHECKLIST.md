# MES-59 merge-readiness checklist

This checklist documents the repo-facing review sweep required before a PR is
claimed merge-ready.

It exists to catch connector-delayed GitHub review feedback, especially inline
review threads that can be missed by a flat comment sweep.

## Required connector sweeps

Before merge readiness is claimed, record both connector sweeps:

1. **Immediate sweep**: inspect connector-visible PR feedback after the latest
   pushed changes and after expected CI feedback has posted.
2. **Delayed sweep**: repeat the connector review sweep shortly before merge, after
   enough time has passed for delayed review-thread and connector synchronization
   updates to arrive.

The delayed sweep is mandatory even when the immediate sweep found no action.

## Thread-aware review requirement

The sweep must inspect unresolved inline review threads. Flat top-level PR
comments are insufficient because they can omit inline thread state, stale file
context, and whether a reviewer left a thread unresolved.

If a connector view only exposes flat PR comments, use a thread-aware PR review
view before merge readiness is claimed.

## Merge-readiness gate

A PR is not merge-ready until every actionable connector-visible thread is either:

- fixed in the branch, with the fix included in the candidate commit; or
- explicitly recorded in the merge-readiness notes with the owner, rationale, and
  follow-up disposition.

Do not treat a connector sweep as complete when only top-level PR comments were
checked. Do not treat a delayed sweep as optional because the first sweep was
clean.

## Minimum readiness record

The repo-facing merge-readiness notes should include:

- candidate commit checked
- immediate connector sweep time/result
- delayed connector sweep time/result
- confirmation that unresolved inline review threads were inspected
- list of actionable threads fixed in the branch
- list of actionable threads explicitly recorded instead of fixed before merge
