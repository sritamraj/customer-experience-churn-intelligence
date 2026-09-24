\# Notebook Strategy



The canonical project workflow is implemented in Python and SQL scripts under `src/` and `sql/`.



Notebooks are intentionally not required for the production-style pipeline.



If notebooks are added later, they should be used for exploratory analysis, visualization, and interview-oriented investigation rather than as the source of truth for model training or final evaluation.



\## Recommended Notebook Topics



1\. Data quality and leakage exploration

2\. Customer behavior exploration

3\. Customer-experience analysis

4\. Feature distributions and temporal drift

5\. Model comparison visualization

6\. Error analysis

7\. Model explainability



\## Reproducibility Rule



The final model, calibration artifact, and final test evaluation must remain reproducible from the version-controlled Python/SQL pipeline.



Do not use notebooks to:



\* silently retrain the canonical model

\* recalibrate on the final test set

\* modify frozen artifacts

\* redefine the final target

\* report metrics that cannot be reproduced from the project scripts



Exploratory notebook results should be treated as supporting analysis, not as the canonical execution path.
