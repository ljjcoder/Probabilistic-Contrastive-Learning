# PCL
Probabilistic Contrastive Learning for Domain Adaptation [here](https://arxiv.org/abs/2111.06021) (official Pytorch implementation). 

#### About code

We have refactored the experimental code (UDA_GVB and SSL_Flexmatch), if you have any problems, please contact us in time.

#### Something to Say:

The full paper is answering two questions:

#### 1. Why does conventional FCL perform poorly in DA tasks?

#### 2. Why is such a simple PCL able to significantly improve performance?

We think that the core problem here is that the current FCL generally ignores the relationship between features and class weights, while PCL explicitly narrows the distance between features and class weights. Based on the above understanding, PCL can be used in all close-set learning tasks, that is, unlabeled data and labeled data belong to the same semantic space. So far, we have verified 5 tasks and we believe that PCL can bring benefits on many more tasks.
#### Therefore, we also expect researchers to verify the effectiveness of PCL in more tasks!!!!!! In addition, we also strongly recommend that relevant researchers consider the deviation of features and class weights in this type of task, and design a more elegant and efficient contrastive learning loss.

#### Personal Feelings:

Readers may feel that using probability for contrastive learning is as natural as breathing, and there is nothing difficult about it, let alone innovation. But I still want to defend it a little bit. 

During my own research period, the process of migrating standard FCL to PCL was fraught with hardships. At first, based on the naive intuition that contrastive learning can improve feature generalization, like many methods in other fields, I applied it directly to the DA task. However, the gains obtained are very limited, which has puzzled me for a long time, and even suspected that contrastive learning is not suitable for domain adaptation tasks. It took me a long time before I realized that there was a problem with the matching of class weights and features. 

However, then a key question is how to make the features approach the category weights. More importantly, what kind of indicators should we use to measure this closeness. Fortunately, We first realized that the one-hot form can be used as a suitable metric, and thus designed the concise PCL. 

Now that PCL has been presented in front of you, all this seems so natural that it is not even worth mentioning. However, when I first faced the problem, it was not obvious what was problem with FCL for DA and how to design PCL. For me, this process is more like looking for a faint fire in the vast darkness, and I doubt whether I am going in the right direction every step of the way. Luckily, I came out of the dark and found a solution that was so simple and so effective. For me, PCL is beautiful, and it makes me appreciate the charm of the ancient Chinese adage "The Greatest Truths is Concise". Therefore, I implore readers to act as if you were facing this problem for the first time, and try your best to understand the difficulties in the process.

#### If you use this code/method or find it helpful, please cite:


```
@article{li2021semantic,
  title={Semantic-aware representation learning via probability contrastive loss},
  author={Li, Junjie and Zhang, Yixin and Wang, Zilei and Tu, Keyu},
  year={2021}
}
```


 
