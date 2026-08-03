# CoCC verifier security and diagnostic boundary

## Current research executor

The CoCC verifier launches a separate `python -I` process, applies an AST deny-list,
captures output and enforces an external timeout. This reduces accidental interference
and blocks a defined set of imports and calls. It is not a complete sandbox: candidate
code is still executed with `exec`, and deny-lists cannot establish general safety.

Use the current worker only for controlled local or CI research data on a machine whose
contents and credentials can tolerate compromise. Do not expose it as a public code
execution service.

## Required production isolation

A general-use executor must run every candidate in a disposable boundary with:

- no network namespace or outbound connectivity;
- read-only root filesystem and an empty size-limited writable scratch volume;
- unprivileged user with no host credentials;
- process, file-descriptor and child-process limits;
- CPU quota, wall-clock timeout and memory limit enforced outside the candidate;
- immutable worker image bound by digest;
- destruction of the container after each evaluation.

The AST safety gate remains defense in depth and must not be treated as the primary
isolation mechanism.

## Source extraction contract

The parent verifier no longer chooses the longest fenced block. It examines every
Python or unlabeled code block and selects the unique block that declares the requested
top-level function or `Solution` method. Outcomes are:

- `selected`: exactly one block declares the contract;
- `ambiguous_extraction`: more than one block declares it, or a stdin task has multiple
  candidate blocks;
- `callable_absent`: parseable candidate blocks do not declare the requested callable;
- `selected_syntax_unresolved`: one candidate block exists but cannot be parsed, so the
  worker remains responsible for the `SyntaxError` diagnosis.

## Exception audit contract

`missing_callable` is accepted only when both the worker exception class and message
match:

```text
AttributeError
callable '<identifier>' not found
```

All worker exception results preserve `exception_type` and a message truncated to 1,000
characters. The parent process revalidates the class-message-label relationship and
fails closed on inconsistent output. An internal candidate `AttributeError` is not
collapsed into `missing_callable`.
