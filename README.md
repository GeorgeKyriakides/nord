# nord
Deep neural architecture research framework


NORD is a deep neural architecture research framework that aims to provide tools in order to 
make the implementation and comparison of neural architecture design methods fair and straightforward.

## Main concepts

- Descriptors: Making the programmatical generation of nodes and connections more straigth-forward.

- Evaluators: Evaluating the quality of descriptors' networks.

- Environments: Managing the distributed execution of Evaluators.

## Main requirements

- PyTorch and Torchvision (https://pytorch.org/)

- Horovod (https://github.com/uber/horovod)

- mpi4py (https://github.com/mpi4py/mpi4py), 
along with an MPI implementation such as (https://www.mpich.org/) or (https://www.open-mpi.org/)

- networkx (https://networkx.github.io/)
