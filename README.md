# Wearable heart-rate gap benchmark

The project's PPG quality rule is **final v2**.

- [Final v2 guide](member2_ppg/V2_WORKFLOW.md)
- [Final v2 settings](member2_ppg/results/batch_v2/config.json)
- [Final v2 batch manifest](member2_ppg/results/batch_v2/batch_manifest.json)
- [Files to commit](member2_ppg/V2_FILES.txt)

Verify the rule from the repository root:

```powershell
python member2_ppg/final_v2.py --verify-rule
```

V2 outputs are in `member2_ppg/results/batch_v2`, using the same participant
folder layout as v1. The guide records evaluation results and limitations.
V1 and earlier experiments are preserved. No imputation training is included.
