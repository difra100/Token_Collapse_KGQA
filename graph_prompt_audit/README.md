# Graph Prompt Audit

This directory is a sidecar diagnostic layer for released graph-to-LLM methods.
It must not replace an external method's graph encoder, projector, prompt
construction, training loop, checkpoint format, or data preprocessing.

Each method adapter will capture the **projected token sequence actually passed
to the LLM**, preserve its padding mask and ordering semantics, and invoke the
method's own inference path under a token override.  This enables fresh-process
mean-token, whole-sequence swap, graph-recomputed swap, and corruption probes
without training a modified architecture.

## Multi-token reporting

For every configuration we report:

- pooled inter-example cosine and residual ratio, directly comparable with a
  single-token interface;
- token norm ratio relative to the target LLM vocabulary;
- within-example token cosine, which detects duplicated tokens within one
  prompt;
- slotwise inter-example cosine only when token slots are canonical; and
- effective rank for all valid tokens and for pooled sequences.

For node-selection methods, token order is not canonical. Their slotwise score
is intentionally omitted rather than producing an invalid matched-slot result.
