# API design list

## GCN

### SparseMatmul

SparseMatul(CSR *A, Matrix *B) -> Matrix C

#### param[in]: 

Matrix A in CSR format

Matrix B

#### param[out]

output matrix

#### Implementation detail

Similar to the graphsum layer that has been implemented

### Graphsum

GraphSum(Graph g, Matrix *A) -> Maxtrix C

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

