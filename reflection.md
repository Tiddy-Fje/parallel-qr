## Context of application 
* $A$ that is ia $m\times n$ matrix with $n\approx 50$ and $m\gg n$ 
  * locally, maybe limit $m<1000$ ??
* goal is to orthogonalise the matrix
* was there a mess between W and A on slides 23 and 24
* what should made vary ??
  * number of processors 
    * average should be computed on 3-5 runs 
      * this can be done in the same excecution 
    * can go up to 32 or 64 processors 
      * are powers of two sufficient ? (2,4,8,16,32,64)
  * the initial matrix : one additional one is sufficient
    * took one from suggested website
    * can also include this in the same run
  * not the matrix size !!
  

## Question 
* what can we actually run with the same sbatch command ??
  * initially thought could do all tests with given number of cores to gain number of sbatches to run 
  * but do we also care about the 