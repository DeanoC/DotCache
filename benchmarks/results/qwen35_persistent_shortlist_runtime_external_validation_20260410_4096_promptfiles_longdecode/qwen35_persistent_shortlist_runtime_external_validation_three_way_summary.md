# Qwen3.5 Persistent Shortlist External Validation (Three-Way)

## Results

- hand-tuned avg abs: 0.0025901
- policy all-steps avg abs: 0.0025985
- policy mid/late-only avg abs: 0.0025985
- hand-tuned max abs: 0.0325336
- policy all-steps max abs: 0.0325336
- policy mid/late-only max abs: 0.0325336

## Read

The step gate changes routing behavior but not outcome on this external corpus:
- all-steps policy applied rate: 1.000
- mid/late-only policy applied rate: 0.750
- both policy variants produce the same replay quality to the displayed precision

So the small external regression is not coming from bootstrap alone. The runtime policy remains a useful structured prior, but this experiment does not justify switching from the hand-tuned selector yet.
