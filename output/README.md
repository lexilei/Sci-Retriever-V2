bow counting is much more accurate. 
Some observations:
    some datasets are unique to one task, some datasets are used for multiple tasks.
    for datasets that only appear in one subgraph (such as 'ade20k'), they are usually top 1 in word frequency
    for datasets that are used in multiple tasks, 有“concentration”的问题，i'm guessing 有一些tasks可能论文network本身比较小，所以更大的论文network的用这个词的整体frequency更大。但总体来说还是很准确。

ways to prevent this from happening:
    use number of mentions/the size of the network or other ways to measure concentration
    or simply use task+dataset as keywords