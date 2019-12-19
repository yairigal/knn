# K-nearest-neighbors

### How to use
```python
from knn import Knn,manhattan_distance
from knn import load_set

k = Knn(k=7, distance=manhattan_distance)
train_set = load_set('/path/to/train/set/csv/file')
k.fit(train_set)
k.classify([0,1,2,3]) # enter your examples.

# Calculate n fold validation
Knn.n_fold_validation(n=0.66, train_set_path=train_set, k=17, distance=manhattan_distance)
```