# TODO: create a clean data setting for SDV that looks like this:
# | mg/dl | iob | hour (int) | minute (int) | day\_of\_week (int 0-6) |
# | :--- | :--- | :--- | :--- | :--- |
# | 236.0 | 0.0 | 0 | 30 | 3 |

import os
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

real_data = pd.read_csv('../data/cleaned/slow_iob/5/0.csv')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(real_data)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(real_data)

synthetic_data = synthesizer.sample(num_rows=100)

os.makedirs('../data/synthetic', exist_ok=True)
synthetic_data.to_csv('../data/synthetic/synthetic_patients_for_demo.csv', index=False)