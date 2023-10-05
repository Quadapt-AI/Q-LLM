import numpy as np

class LESE:
    def __init__(self, a, b, n_gram, show_matr = True):
        '''LESE: LEvenshstein Sequential Evaluation metric

        Parameters
        ----------
        a : str or list
            Reference string sequence or list of string sequence.
        b : str or list
            Hypothesis string sequence or list of string sequence.
        n_gram : int, optional
            n-gram size. The default is 3. See lese(..) sub module
        show_matr: bool
            flag to display final Levenshstein cost matrix
            
        Returns
        -------
        None
        '''
        self.a = a
        self.b = b
        self.n_gram = n_gram
        self.show_matr = show_matr
        self.lese(self.a, self.b, self.n_gram)
    

    def print_lev(self, D, n_a, n_b):
        '''Print computation matrix on console
    
        Parameters
        ----------
        D : np.array 
            Levenshstein distance computation matrix (MxN).
        n_a : int
            Length of reference sequence.
        n_b : int
            Length of hypothesis sequence.
    
        Returns
        -------
        None.
    
        '''
        for i in range(n_a + 1):
            for j in range(n_b + 1):
                print(int(D[i][j]), end=" ")
            print()
            
            
    def precision_lev_(self, D_i_j, n_a, n_b, n):
        '''Sequential Levenshstein Precision
    
        Parameters
        ----------
        D_i_j : float
            Levenshstein distance.
        n_a : int
            N-gram length of reference sequence.
        n_b : int
            N-gram length of hypothesis sequence.
        n : int
            N-gram size.
    
        Returns
        -------
        prec : float
            Precision.
    
        '''
        max_n_a_n_b = max(n_a, n_b)
        lev_seq = D_i_j//n
        prec = max(0, abs((max_n_a_n_b - lev_seq))/n_b)
        return prec
    
    
    def recall_lev_(self, D_i_j, n_a, n_b, n):
        '''Sequential Levenshstein Recall
        
        Parameters
        ----------
        D_i_j : float
            Levenshstein distance.
        n_a : int
            N-gram length of reference sequence.
        n_b : int
            N-gram length of hypothesis sequence.
        n : int
            N-gram size.
    
        Returns
        -------
        rec : float
            Recall.
        '''
        max_n_a_n_b = max(n_a, n_b)
        lev_seq = D_i_j//n 
        rec = max(0, abs((max_n_a_n_b - lev_seq))/n_a)
        return rec
    
    
    def f_score_lev_(self, D_i_j, n_a, n_b, n, beta = 1.):
        '''Sequential Levenshstein F1- score
    
        Parameters
        ----------
        D_i_j : float
            Levenshstein distance.
        n_a : int
            N-gram length of reference sequence.
        n_b : int
            N-gram length of hypothesis sequence.
        n : int
            N-gram size.
        beta : float, optional
            beta weight for balancing precision-recall. The default is 1.
    
        Returns
        -------
        float
            Precision.
        float
            Recall.
        float
            f_score.
        '''
        precision = self.precision_lev_(D_i_j, n_a, n_b, n)
        recall = self.recall_lev_(D_i_j, n_a, n_b, n)
        if precision == 0. and recall == 0.:
            return 0., 0., 0.
        else:
            f_score = ((1+ np.square(beta))*precision*recall)/(np.square(beta)*precision+recall)
        return precision, recall, f_score
    
    
    def lese(self, a, b, n = 3):
        '''LESE: LEvenshstein Sequential Evaluation metric main module
    
        Parameters
        ----------
        a : str or list
            Reference string sequence or list of string sequence.
        b : str or list
            Hypothesis string sequence or list of string sequence.
        n : int, optional
            n-gram size. The default is 3.
    
        Returns
        -------
        Tuple
            Tuple(Levenshstein distance, (precision, recall, f_1_score)).
            
        Complexity
        ----------
        Time: O(M*N)
        Space: O(M*N)
    
        '''
        if isinstance(a, str) and isinstance(b, str):
            n_a = len(a)
            n_b = len(b)
            if(len(a)//n) ==1: n = 1
            else: pass
            if(len(b)//n) ==1: n = 1
            else: pass
            n_gram_a = len(a)//n #n-gram in reference
            n_gram_b = len(b)//n #n-gram in hypothesis
            if n_b == 0:
                self.levenshstein_distance = n_a
                self.precision_, self.recall_, self.f_score_ = (0.0, 0.0, 0.0)
                return self.levenshstein_distance, self.precision_, self.recall_, self.f_score_
            elif n_a == 0:
                self.levenshstein_distance = n_b
                self.precision_, self.recall_, self.f_score_ = (0.0, 0.0, 0.0)
                return self.levenshstein_distance, self.precision_, self.recall_, self.f_score_
            else:
                pass
        elif isinstance(a, list) and isinstance(b, list):
            a = [x for x in a if not x == 'nan' if not x == ' ' if not x == '']
            b = [x for x in b if not x == 'nan' if not x == ' ' if not x == '']
            n_a = len(a)
            n_b = len(b)
            if(len(a)//n) ==1: n = 1
            else: pass
            if(len(b)//n) ==1: n = 1
            else: pass
            n_gram_a = len(a)//n
            n_gram_b = len(b)//n
            if n_b == 0:
                self.levenshstein_distance = n_a
                self.precision_, self.recall_, self.f_score_ = (0.0, 0.0, 0.0)
                return self.levenshstein_distance, self.precision_, self.recall_, self.f_score_
            elif n_a == 0:
                self.levenshstein_distance = n_b
                self.precision_, self.recall_, self.f_score_ = (0.0, 0.0, 0.0)
                return self.levenshstein_distance, self.precision_, self.recall_, self.f_score_
            else:
                pass
        
        self.D = np.zeros((n_a + 1, n_b + 1))
        # Initialising first row using the length of string/list a:
        for i in range(n_a + 1):
            self.D[i][0] = i
        
        # Initialising first column using the length of string/list b:
        for j in range(n_b + 1):
            self.D[0][j] = j
        
        for i in range(1, n_a + 1):
            for j in range(1, n_b + 1):
                if a[i-1:i+n-1] == b[j-1:j+n-1]:
                    self.D[i][j] = self.D[i - 1][j - 1]
                else:
                    # Adding 1 to account for the cost of operation
                    insertion = 1 + self.D[i][j - 1] #insertion operation
                    deletion = 1 + self.D[i - 1][j] #deletion operation
                    substitution = 1 + self.D[i - 1][j - 1] #substitution operation
                    
                    # Choosing the best option:
                    self.D[i][j] = min(insertion, deletion, substitution)
        if self.show_matr == True:
            self.print_lev(self.D, n_a, n_b)
        else:
            pass
        self.levenshstein_distance = self.D[i][j]
        self.precision_, self.recall_, self.f_score_ = self.f_score_lev_(self.D[i][j], n_gram_a, n_gram_b, n)
        return self.levenshstein_distance, self.precision_, self.recall_, self.f_score_

