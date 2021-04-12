**DATE experiments**: https://github.com/naacl943/experiments

* Generate masks: create_masks.py

* AG News example: train_ag.py

* **Run an experiment on the AG News business split**:

  * pip install git+https://github.com/naacl943/simpletransformers.git

  * git clone https://github.com/naacl943/experiments.git

  * cd experiments

  * git clone  https://github.com/naacl943/datasets.git

  * cd datasets && python ag.py && bash generate_outliers_ag.sh

  * python train_ag.py
