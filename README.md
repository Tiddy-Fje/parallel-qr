## README for QR Factorization Algorithms Analysis

### Project Overview
This project was developed as part of the *HPC for Numerical Methods and Data Analysis* course at EPFL. It focuses on the numerical stability and computational performance of three QR factorization algorithms: Classical Gram-Schmidt (CGS), Cholesky QR (CQR), and Tall-Skinny QR (TSQR). The project compares these algorithms using theoretical analysis and numerical experiments on both well-conditioned and ill-conditioned matrices.

See `report/report.pdf` for a detailed analysis and discussion of the project. It covers the theoretical background, implementation details, and results of the numerical stability and runtime investigations.

### How to Run the Analysis
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Set up the environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate qr-factorization
   ```
3. Run the Python scripts to reproduce the experiments and analysis.

### Author
Tara Fjellman. For questions or collaborations, please reach out via email.

### Acknowledgments
- Professor Grigori Laura: Provided guidance and answered questions related to the project.
- Original references: The project builds on the theoretical foundations and algorithms discussed in the course materials and related literature.

### Notes
- This project adheres to EPFL guidelines for reproducible research. Ensure all code modifications are documented, and credit is given where due.
- The results and findings are based on numerical experiments conducted on the Helvetios cluster at EPFL.

### References
- [1] ‘GHS_psdef/cvxbqp1 : SuiteSparse Matrix Collection’. Accessed: Sep. 28, 2024. [Online]. Available: https://sparse.tamu.edu/GHS_psdef/cvxbqp1
- [2] ‘Helvetios’, EPFL. Accessed: Nov. 03, 2024. [Online]. Available: https://www.epfl.ch/research/facilities/scitas/hardware/helvetios/
- [3] G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H. D. Nguyen, and E. Solomonik, ‘Reconstructing Householder Vectors from Tall-Skinny QR’, in 2014 IEEE 28th International Parallel and Distributed Processing Symposium, May 2014, pp. 1159–1170. doi: 10.1109/IPDPS.2014.120.