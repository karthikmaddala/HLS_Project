# Naming convention

Each directory inside the layers directory contains files for that particular layer with different set of pragmas inserted. The files are named as followed - 

`<layer_name>_<l1><n1><l2><n2>...`

where `l1` denotes the first letter corresponding to pragmas like tiling, pipelining and flattening
and `n1` denotes the factor corresponding to `l1`

Till now the pragmas inserted have unique first letters and hence are named accordingly. For example `conv2d_TP` denotes convolution with tiling and pipelining and so on. 

Conflicting first alphabets of pragmas will be resolved later as needed. 

1. Pipeline - P
2. Unroll - U
3. Tiling - T
4. Partitioning - Pa
5. Flattening - F