# PA228 Semestral project - Cell instance segmentation

## Method
I chose to implement the method **EmbedSeg** as described in the paper *Embedding-based Instance Segmentation in
Microscopy* (https://arxiv.org/abs/2101.10033).

The method is based on predicting offset vectors that push pixels of each cell instance towards its center (in this paper the center is represented by the medoid so as to guarantee that it lies inside the instance).
Post-processing step consists of clustering close pixels and labeling those by identical instance label.

In addition to the offset vectors, a *seediness* map and *sigma* (clustering bandwith) map are also predicted. The *seediness* map is essentially a map of probabilities, specifying whether a particular pixel a cell pixel.
To correctly cluster pixels of instace of various sizes (as is the case with cells), we also predict *sigma* map, that specifies the clustering bandwith that should be used for the instance that the pixel belongs to.

## Data preparation
For the training purposes, the needed annotations are the *instance centers*. For each segmentation annotation image of shape *(H, W)* I generated an image *INST_CENTERS* of shape *(H, W, 2)* where each pixel *(x, y)* belonging to an cell instance stores the cell instance's center in *INST_CENTERS*.
