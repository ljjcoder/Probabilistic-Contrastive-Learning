# PCL
Probabilistic Contrastive Learning for Domain Adaptation [here](https://arxiv.org/abs/2111.06021) (official Pytorch implementation). 

#### About code

We have refactored the experimental code (UDA_GVB and SSL_Flexmatch), if you have any problems, please contact us in time.

#### Something to Say:

PCL is a simple yet powerful contrastive learning loss. The full paper is answering two questions:

#### 1. Why does conventional FCL perform poorly in DA tasks?

#### 2. Why is such a simple PCL able to significantly improve performance?

We think that the core problem here is that the current FCL generally ignores the relationship between features and class weights, while PCL explicitly narrows the distance between features and class weights. Based on the above understanding, PCL can be used in all close-set learning tasks, that is, unlabeled data and labeled data belong to the same semantic space. So far, we have verified 5 tasks and we believe that PCL can bring benefits on many more tasks.
#### Therefore, we also expect researchers to verify the effectiveness of PCL in more tasks!!!!!!

#### Personal Feelings:

Readers may feel that using probability for comparative learning is as natural as breathing, and there is no difficulty. But I still want to defend it a little bit. During my own research, the process of migrating standard FCL to PCL was fraught with hardships. At first, based on the naive intuition that contrastive learning can improve feature generalization, like many methods in other fields, I applied it directly to the DA task. However, the gains obtained are very limited, which has puzzled me for a long time, and even suspected that contrastive learning is not suitable for domain adaptation tasks. It took me a long time before I realized that there was a problem with the matching of class weights and features. Of course, after this  conclusion is uttered, it is indeed very natural and not worth mentioning. But the first time I realized what the problem was, I did have a sense of finding a ray of light in an endless darkness for myself. I think this is also the charm of science.


#### If you use this code/method or find it helpful, please cite:


```
@article{li2021semantic,
  title={Semantic-aware representation learning via probability contrastive loss},
  author={Li, Junjie and Zhang, Yixin and Wang, Zilei and Tu, Keyu},
  year={2021}
}
```


 
