Purpose: Provide stable operator-facing compatibility entrypoints during repository restructuring.
What belongs here: Thin shims, migration adapters, and minimal operator docs that delegate to maintained implementations elsewhere in the repo.
What must not appear here: Duplicated orchestration logic, live sbatch wrappers, Mirheo runtime code, generated run outputs, or operator-local scratch files.
Key rules: Every shim must point at one maintained source of truth; keep this tree safe for non-live paths unless a live route is explicitly approved and governed.
