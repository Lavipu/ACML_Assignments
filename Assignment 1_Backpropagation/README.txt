NEURAL NETWORK IMPLEMENTATION 

The present zip file contains to scripts : 
MainNN_default.ipnyb and 
WorkingCodeNN.ipynb


################################  MainNN_default.ipnyb  #################################


Execute MainNN_default.ipynb file with Jupyter Notebook. 

The script inlcudes: 
- Input values X - 8x8 identity matrix
- learning rate alpha fixed to 0.8
- Weight decay parameter fixed to 0.0001
- The following functions: 
  ---- sigmoid() function: 
       computing sigmoid function
       
  ---- d_sigmoifd() function: 
       computing the firts derivative of the sigmoid function
       
  ---- forward() function: 
       - input : input values X
       - output: activation values of output layer "output" = a3
       Performing   FORWARD PROPAGATION
       1. Activations a2 and a3 (though sigmoid function)
       2. W1 and W2 are the weights from input to hidden and from hidden to output layer 
         respectively  output
       Activation input layer  - z1 --> sigmoid(z1) = a1 = x i.e. input values
       Activation hidden layer - z2 --> sigmoid(z2) =a2
       Activation output       - z3 --> sigmoid(z3) =a3 output
       
   ---- bacwardpropagation_algo() function: 
      - input : input values X
      - output: totalError, UpdateW1, UpdateW2
      Performing: Backpropagation iterating though all samples in X
         Step 1: Forward Propagation to get the weights
         Step 2: Bacward to get activation errors in each layer and compute the weights 
                 updates 
             1. Compute Output error 
             2. delta 3 i.e. output delta : sigmoid derivative times output error
             3. delta 2 i.e. hidden delta : errors on hidden 
             4. updateW1 (resp. updateW2) : the product betwwen a1 (resp. a2) and 
                delta2 (resp delta3) 

----------------------------------------------------------------------------------------

---- while loop for convergence of the totalError for a fixed number of iterations: 
     - It updates for the weights after passing forward() and backpropagation_algo()

---- plot of convergence of the total error w.r.t. learning_rate and LAMBDA. 

########################################################################################
################################  WorkingCodeNN.ipynb  #################################

Execute  WorkingCodeNN.ipynb file with Jupyter Notebook. 

This script is intended for users that want to experiment more with the hyper-parameters 
and want to select the best combination of both. 

It consist of 3 cells, 

1.  Contains the same functions present in MainNN_default.ipynb: 
    - output a graph showing how the accumalted error of the network decreases for every 
      one of the learning rates, which can be modified by the user. 
      
2.  Contains the same functions present in MainNN_default.ipynb: 
    - calculates the accumulated errors for the learning rates of the first cell but it 
      uses a range of different weight decay parameters, lambda. 

3. Contains the same functions present in MainNN_default.ipynb: 
   - outputs a table with the metrics of the combination of both hyperparameters and 
     prints a message indicating which combination leads to a lowest error.

     



