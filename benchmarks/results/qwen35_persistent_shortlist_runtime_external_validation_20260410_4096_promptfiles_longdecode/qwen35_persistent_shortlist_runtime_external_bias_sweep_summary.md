# Qwen3.5 Persistent Shortlist External Bias Sweep

## Results

- bias010_all: avg abs 0.0018026, max abs 0.0269186, policy applied 1.000
- bias010_midlate: avg abs 0.0018026, max abs 0.0269186, policy applied 0.750
- bias005_all: avg abs 0.0018027, max abs 0.0269186, policy applied 1.000
- bias005_midlate: avg abs 0.0018027, max abs 0.0269186, policy applied 0.750
- bias005_midonly: avg abs 0.0018027, max abs 0.0269186, policy applied 0.500
- bias002_all: avg abs 0.0018030, max abs 0.0269186, policy applied 1.000
- bias002_midlate: avg abs 0.0018030, max abs 0.0269186, policy applied 0.750

## Read

Bias mode was stable across the tested range, and step gating did not materially change replay quality on the unseen-family corpus.
The best measured point was the all-step bias run at weight 0.10, but the margin over 0.05 and 0.02 was tiny.
