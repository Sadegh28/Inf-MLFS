from tqdm import tqdm
import numpy as np
from .utils import mi_pairwise as rel, mi_pairwise as red

class inf_mlfs(): 
    
    def __init__(self):
        pass

        
        
        

    def rank(self, X, y, mode = 'pre_eval'):
        if mode not in ['pre_eval', 'post_eval']: 
            raise ValueError('invalid mode ==> the mode should be in [pre_eval, post_eval]')
        
        if mode == 'post_eval':
            return self.select(X, y, X.shape[1], mode)
        if mode == 'pre_eval': 
            F = list(range(X.shape[1]))
            S = []
            k = 0 
            REL = rel(X, y, message= 'Relevance Matrix')
            RED = red(X, X, message= 'Redundancy Matrix')
            A = np.array(self.__adj_matrix(REL, RED), dtype=np.float32)
            
            I = np.identity( A.shape[0] )
            r = ( 0.9/ max( np.linalg.eigvals(A) ) ) 
            A1 = I - ( r * A )
            S = np.linalg.inv( A1 ) - I
            WEIGHT = np.sum( S , axis=1 )

            RANKED = np.argsort(WEIGHT)
            RANKED = np.flip(RANKED,0)
            RANKED = RANKED.T
            WEIGHT = WEIGHT.T
            return RANKED
            

   
    
    def __adj_matrix(self,REL, RED): 
        A = [[None for i in range(len(RED))] for j in range(len(RED))]
        A = np.array(A)
        for i in range(len(RED)): 
            REL_i = sum(REL[i][:])
            RED_i = 1.0/len(RED)*sum(RED[i][:])
            for j in range(len(RED)):
                REL_j = sum(REL[j][:])
                RED_j = 1.0/len(RED)*sum(RED[j][:])
                A[i][j] = max([REL_i, REL_j]) - min([RED_i, RED_j])
        return A

                

    