# API design list

## GCN

### SparseMatmul

SparseMatul(CSR *A, Matrix *B) -> Matrix C

#### Implementation detail

Similar to the graphsum layer that has been implemented, using Gunrock load balancing policy.

### Graphsum

GraphSum(Graph g, Matrix *A) -> Maxtrix* C

#### implementation detail

This has been implemented, further troubleshooting needed

### ReLU
ReLU(Matrix *A) -> Matrix* B

#### implementation detail

Using Gunrock forEach loop to perform forward propagation, backward propagation needs to be investigated.

### Matmul
Matmul(Matrix *A, Matrix *B) -> Matrix *C

#### implementation detail
- use Gunrock forEach to implement
- use code from: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu

### CrossEntropy Loss
CrossEntropyLoss(Matrix *logits, Array *ground_truth, int num_class) -> float Loss

This function should update the gradient of Matrix A as well.
#### implementation detail
```
logits.forAll([double *log_max](double *a, int pos) -> {
  if (truth[pos / num_class] < 0) return;
  atomicMax(log_max + pos / num_class, a[pos]);
});

logits.forAll([](double *a, int pos) -> {
  if (truth[pos / num_class] < 0) return;
  a[pos] -= log_max[pos / num_class];
  atomicAdd(sum_exp + pos / num_class, exp(a[pos]));
});

sum_exp.forAll([](double *a, int pos) -> {
  if (truth[pos] < 0) return;
  atomicAdd(&loss, log(sum_exp[pos]) - logits[pos * num_class + truth[pos]]);
});
```
## GraphSAGE

