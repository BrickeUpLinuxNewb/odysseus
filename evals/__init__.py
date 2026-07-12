"""Project Inari eval harness -- the ruler everything else is measured with.

A test rig for non-deterministic model output. Three parts, per the roadmap:
a fixed task set with rubrics decided in advance (``evals/tasks/*.jsonl``),
programmatic scoring (``checkers.py`` -- no LLM judge, no vibes), and a runner
that produces a number plus its noise floor (``runner.py``).

Design constraints, in priority order:
  1. **Cheap to run.** If a sweep takes hours it stops being run, and the
     discipline dies. Sequential by default, small task sets, no heavy deps.
  2. **Programmatic scoring only.** Every task kind is machine-checkable
     (label choice, exact value, regex, JSON shape, numeric tolerance).
     Free-form generation quality is deliberately out of scope for v1.
  3. **Variance is part of the answer.** Each task runs N times; the summary
     reports mean AND spread, and ``compare`` refuses to call a delta real
     when it is inside the pooled noise floor.
  4. **Quality is not the only axis.** Every call records latency and
     throughput, so a quantization sweep sees the quality/speed trade-off.
     (VRAM is recorded by the operator per config; it cannot be measured
     portably from here.)

Import discipline: this package must import without the Odysseus app stack so
``compare`` can run on any box. Anything touching the LLM stack, the SWT loop,
or the store is imported lazily inside the specific target/command that needs
it.
"""
