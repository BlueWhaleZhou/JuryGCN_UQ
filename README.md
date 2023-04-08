# JuryGCN

This is the implementation for KDD'22 paper: ["JuryGCN: Quantifying Jackknife Uncertainty on Graph Convolutional Networks"](https://dl.acm.org/doi/pdf/10.1145/3534678.3539286?casa_token=rRgnUo3vPOUAAAAA:x81aqtd5ffaxYeKlZ_OaR7G9oQ66n-2e1crUrYXklxa46JUq1xRwyh36jv68bnq7OpbK4NLdYM8f1g).

# Requirements
-Python: 3.8  
-Pytorch: 1.4.0  
-numpy: 1.19.2  
-scikit-learn: 1.1.3  
-scipy: 1.10.1  
-autograd: 1.5

# Evaluation
UQ with the application to active node classification: python uq4al.py

# Others
Please kindly cite our paper if you find it helpful to your research:

```
@inproceedings{kang2022jurygcn,
  title={JuryGCN: quantifying jackknife uncertainty on graph convolutional networks},
  author={Kang, Jian and Zhou, Qinghai and Tong, Hanghang},
  booktitle={Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={742--752},
  year={2022}
}
```
