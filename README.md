# DATE

* **Title**: DATE: Detecting Anomalies in Text via Self-Supervision of Transformers
* **Author**: Andrei Manolache, Florin Brad, Elena Burceanu
* **[Paper](https://aclanthology.org/2021.naacl-main.25.pdf)**: accepted at NAACL 2021, and [ICML 2021 - UDL workshop](https://sites.google.com/view/udlworkshop2021)
* **[Poster NAACL](resources/DATE_poster.pdf)**
* **[Presentation](resources/DATE_presentation.pdf)**

Leveraging deep learning models for Anomaly Detection (AD) has seen widespread use in recent years due to superior performances over traditional methods. Recent deep methods for anomalies in images learn better features of normality in an end-to-end self-supervised setting. These methods train a model to discriminate between different transformations applied to visual data and then use the output to compute an anomaly score. We use this approach for AD in text, by introducing a novel pretext task on text sequences. We learn our DATE model end-to-end, enforcing two independent and complementary self-supervision signals, one at the token-level and one at the sequence-level. Under this new task formulation, we show strong quantitative and qualitative results on the 20Newsgroups and AG News datasets. In the semi-supervised setting, we outperform state-of-the-art results by +13.5% and +6.9%, respectively (AUROC). In the unsupervised configuration, DATE surpasses all other methods even when 10% of its training data is contaminated with outliers (compared with 0% for the others).


![DATE train overview](resources/date_train.png)
![DATE test overview](resources/date_test.png)
 
 
This repo contains the official implementation of DATE.

# Please refer to it as:
```
@InProceedings{date-manolache-2021,
  title = 	 {DATE: Detecting Anomalies in Text via Self-Supervision of Transformers},
  author =       {Manolache, Andrei and Brad, Florin and Burceanu, Elena},
  booktitle = 	 {NAACL},
  year = 	 {2021}
}
```
