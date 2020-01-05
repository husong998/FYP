# API design list

## GCN

### SparseMatmul

#### param[in]: 

Matrix A in CSR format

Matrix B

#### param[out]

output matrix

#### Implementation detail

Similar to the graphsum layer that has been implemented

### Graphsum

#### param[in]

Graph

Matrix A

#### param[out]

output matrix

#### implementation detail

This has been implemented

### ReLU

#### param[in]

Graph

Matrix A

#### param[out]

output matrix

#### implementation detail

This has been implemented

### Matmul

#### param[in]

Matrix A

Matrix B

#### param[out]

output matrix

#### implementation detail

use code from: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu

### CrossEntropy Loss

#### param[in]

truth: ground truth label for vertices with label

Matrix A: final outpout feature matrix

#### param[out]

output matrix

#### implementation detail

Write my own cuda kernel, follow the one written in parallel-gcn

## GraphSAGE

# API design list

## GCN

### SparseMatmul

SparseMatul(CSR *A, Matrix *B) -> Matrix C

#### Implementation detail

Similar to the graphsum layer that has been implemented, using Gunrock load balancing policy.

### Graphsum

GraphSum(Graph g, Matrix *A) -> Maxtrix *C

#### implementation detail

This has been implemented, further troubleshooting needed

### ReLU
ReLU(Matrix *A) -> Matrix *B

#### implementation detail
```
A->data.forEach([](double &x) -> { x = max(0, x); });
```
### Matmul
Matmul(Matrix *A, Matrix *B, int dim) -> Matrix *C
#### implementation detail
- use Gunrock forEach to implement
```
A.forAll([](double *a, int pos) -> {
  int i = pos / dim, j = pos % dim;
  for (int k = 0; k < dim; k++) {
    atomicAdd(C + i * dim + k, a[pos] * a[j * dim + k]);
  }
});
```
- use code from: https://github.com/lzhengchun/matrix-cuda/blob/master/matrix_cuda.cu

### CrossEntropy Loss
CrossEntropyLoss(Matrix *logits, Array *ground_truth, int num_class) -> float Loss

This function should update the gradient of Matrix A as well.
#### implementation detail
```
loss = 0;

logits->data.forAll([double *log_max](double *a, int pos) -> {
  if (truth[pos / num_class] < 0) return;
  atomicAdd(&count, 1);
  atomicMax(log_max + pos / num_class, a[pos]);
});

logits->data.forAll([](double *a, int pos) -> {
  if (truth[pos / num_class] < 0) return;
  a[pos] -= log_max[pos / num_class];
  atomicAdd(sum_exp + pos / num_class, exp(a[pos]));
});

sum_exp.forAll([](double *a, int pos) -> {
  if (truth[pos] < 0) return;
  atomicAdd(&loss, log(sum_exp[pos]) - logits->data[pos * num_class + truth[pos]]);
});

if (training) {
  logits->data.forAll([](double *a, int pos) -> {
    if (truth[pos] < 0) return;
    double prob = exp(a[p]) / sum_exp[pos / num_class];
    logits->grad[pos] = prob;
  });
  
  truth.forAll([](int *a, int node_id) -> {
    if (truth[node_id] < 0) return;
    logits->grad[node_id * num_class + truth[node_id]] -= 1.0;
  });
  
  logits->grad.forEach([](double &x) -> { x /= count; });
}

return loss / count;
```
## GraphSAGE
