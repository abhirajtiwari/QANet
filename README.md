# SQuAD2.0

## Evaluating Model:
To run the evaluation:
```bash
python3 evaluate-v2.0.py <path_to_dev-v2.0> <path_to_predictions>
```

Sample input:
```bash
python3 evaluate-v2.0.py <path_to_dev-v2.0> <path_to_predictions>
```

<br>
Sample output:
```
{
  "exact": 64.81091552261434,
  "f1": 67.60971132981278,
  "total": 11873,
  "HasAns_exact": 59.159919028340084,
  "HasAns_f1": 64.7655368790259,
  "HasAns_total": 5928,
  "NoAns_exact": 70.4457527333894,
  "NoAns_f1": 70.4457527333894,
  "NoAns_total": 5945
}
```
