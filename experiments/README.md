**DATE experiments**: https://github.com/bit-ml/date/tree/master/experiments

* Generate masks: create_masks.py

* AG News example: train_ag.py

* **Run an experiment on the AG News business split**:

  * pip install git+https://github.com/bit-ml/date/tree/master/experiments

  * git clone https://github.com/bit-ml/date.git

  * cd experiments

  * cd ..

  * cd datasets && python ag.py && bash generate_outliers_ag.sh

  * python train_ag.py
