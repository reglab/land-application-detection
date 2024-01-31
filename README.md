# Land Application Detection

This repository contains code for a project trying to detect environmental violations in real-time using satellite imagery. Specifically, we detect instances of winter land application, the practice of farms spreading manure on snow-covered fields. 

## Repo organization

- `cikm`: contains code and artifacts for the paper [Detecting Environmental Violations with Satellite Imagery in
Near Real Time: Land Application under the Clean Water Act](https://arxiv.org/pdf/2208.08919.pdf), in Conference of Information and Knowledge Management (CIKM), 2022. This paper and code describes the training, tuning and analysis of the object-detection model. 

- `elpc_pipeline`: contains code and details for a pipeline to run the object detection model and output results of suspected land application events for a given set of locations. This is the pipeline deployed in the winter 2022 field trial with ELPC and WDNR, using the best model from the CIKM paper. It is also a currently functional pipeline that continues to be used by ELPC as they take on an additional piloting and testing of the model in winter 2023.

- `field_trial_2022_analysis.ipynb`: a Jupyter notebook containing analysis from the winter 2022 field trial with ELPC and WDNR. Draws data from the postgreSQL database in which the results sit.  