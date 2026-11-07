# Financial Capability Data (Component 05)

Source: CFPB National Financial Well-Being Survey (NFWBS) 2016 public-use file.

- Page: https://www.consumerfinance.gov/data-research/financial-well-being-survey-data/
- File: https://www.consumerfinance.gov/documents/5614/NFWBS_PUF_2016_data.csv

Download the CSV in a browser and save it as `raw/NFWBS_PUF_2016_data.csv`
(the raw file is not committed). Also keep the official user guide and codebook PDFs
from the same page for reference.

Train from the repository root:

```powershell
python -m backend.src.financial_capability.fc_run_pipeline
```

