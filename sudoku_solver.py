
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.sparse as scs # sparse matrix construction 
import scipy.linalg as scl # linear algebra algorithms
import scipy.optimize as sco # for minimization use
import matplotlib.pylab as plt # for visualization

def fixed_constraints(N=9):
    rowC = np.zeros(N)
    rowC[0] =1
    rowR = np.zeros(N)
    
    rowR[0] =1
    row = scl.toeplitz(rowC, rowR)
    ROW = np.kron(row, np.kron(np.ones((1,N)), np.eye(N)))
    
    colR = np.kron(np.ones((1,N)), rowC)
    col  = scl.toeplitz(rowC, colR)
    COL  = np.kron(col, np.eye(N))
    
    M = int(np.sqrt(N))
    boxC = np.zeros(M)
    boxC[0]=1
    boxR = np.kron(np.ones((1, M)), boxC) 
    box = scl.toeplitz(boxC, boxR)
    box = np.kron(np.eye(M), box)
    BOX = np.kron(box, np.block([np.eye(N), np.eye(N) ,np.eye(N)]))
    
    cell = np.eye(N**2)
    CELL = np.kron(cell, np.ones((1,N)))
    
    return scs.csr_matrix(np.block([[ROW],[COL],[BOX],[CELL]]))




# For the constraint from clues, we extract the nonzeros from the quiz string.
def clue_constraint(input_quiz, N=9):
    m = np.reshape([int(c) for c in input_quiz], (N,N))
    r, c = np.where(m.T)
    v = np.array([m[c[d],r[d]] for d in range(len(r))])
    
    table = N * c + r
    table = np.block([[table],[v-1]])
    
    # it is faster to use lil_matrix when changing the sparse structure.
    CLUE = scs.lil_matrix((len(table.T), N**3))
    for i in range(len(table.T)):
        CLUE[i,table[0,i]*N + table[1,i]] = 1
    # change back to csr_matrix.
    CLUE = CLUE.tocsr() 
    
    return CLUE


# # Function to record repeated positions and delete repeated numbers

# In[2]:


def getBlockHeads():
    row = 0
    col = 0
    block_heads = list()
    while(True):
        while(True):
            block_heads.append([row,col])
            col += 3
            if(col == 9):
                col = 0
                break
        row += 3
        if(row >= 9):
            break
    return block_heads

def getRowHeads():
    row_head = list()
    for i in range(0,9):
        row_head.append([i,0])
    return row_head

def getColHeads():
    col_head = list()
    for i in range(0,9):
        col_head.append([0,i])
    return col_head

def findRepeat(l,board):
    repeated_l = list()
    i = 0
    j = 0
    while(i<len(l)):
        j = i+1
        temp = [l[i]]
        while(j<len(l)):
            ii = board[l[i][0]][l[i][1]]
            jj = board[l[j][0]][l[j][1]]
            if(ii == jj):
                temp.append(l[j])
            j += 1
        if(len(temp)>1):
            check = True
            repeated_l.append(temp)
        i += 1
    return repeated_l


def findAllRepeat(block_heads,row_heads,col_heads,board):
    repeated_l = []

    #for block
    for row,col in block_heads:

        currentBlock = list()
        for r in range(row,row+3):
            for c in range(col,col+3):
                currentBlock.append([r,c])#one block per list

        repeated_l += findRepeat(currentBlock,board)

    #for row
    for row,col in row_heads:

        current_row = list()
        for c in range(col,col+9):
            current_row.append([row,c])#one row per list
        repeated_l += findRepeat(current_row,board)


    #for col
    for row,col in col_heads:

        current_col = list()
        for r in range(row,row+9):
            current_col.append([r,col])#one row per list
        repeated_l += findRepeat(current_col,board)

    return repeated_l

def deleteAllRepeat(repeated_l, board):
    for i in repeated_l:
        for ii in i:
            board[ii[0]][ii[1]] = 0
    return board


def deleteRepeated(board):
    block_heads = getBlockHeads()#[[0, 0], [0, 3], [0, 6], [3, 0], [3, 3], [3, 6], [6, 0], [6, 3], [6, 6]]
    row_heads = getRowHeads()#[[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]]
    col_heads = getColHeads()#[[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7], [0, 8]]
    
    repeated_l = findAllRepeat(block_heads,row_heads,col_heads,board)#[[[0, 0], [2, 1], [2, 2]], [[0, 1], [1, 1]]]
    board = deleteAllRepeat(repeated_l,board)
    return board, repeated_l


# # Weighted L1-norm minimization solver

# In[3]:


def demo_solver(quiz):
    A0 = fixed_constraints()
    A1 = clue_constraint(quiz)
    
    A = scs.vstack((A0,A1))
    A = A.toarray()
    B = np.ones(A.shape[0])


    # Because rank defficiency. We need to extract effective rank.
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    K = np.sum(s > 1e-12)
    S = np.block([np.diag(s[:K]), np.zeros((K, A.shape[0]-K))])
    A = S@vh
    B = u.T@B
    B = B[:K]

    c = np.block([ np.ones(A.shape[1]), np.ones(A.shape[1]) ])
    G = np.block([[-np.eye(A.shape[1]), np.zeros((A.shape[1], A.shape[1]))],                         [np.zeros((A.shape[1], A.shape[1])), -np.eye(A.shape[1])]])
    h = np.zeros(A.shape[1]*2)
    H = np.block([A, -A])
    b = B
    
    # Weight####################################
    L = 10
    x_pos = np.zeros((A.shape[1]))
    x_neg = np.zeros((A.shape[1]))
    x_ori = x_pos - x_neg
    tol = 1e-10
        
    for k in range(L):
        W1 = 1/(abs(x_ori) + 1)
        W  = np.block([W1,W1])
        c_w = np.matrix(c*W)
        sol = sco.linprog(c_w, G, h, H, b, method='interior-point', options={'tol':1e-6})
        
        x_news = sol.x[:A.shape[1]] - sol.x[A.shape[1]:]
            
        if np.linalg.norm(x_news - x_ori) < tol:
            break
            
        else:
            x_ori = x_news
    ##################################
    z = np.reshape(x_news, (81, 9))
    z = np.reshape(np.array([np.argmax(d)+1 for d in z]), (9,9))
    return z
##Output is an 9*9 matrix
    


# # Enhanced Solver with Warm Restart 2.0

# In[4]:


## Input is the 81-character string for the puzzle, output is the 81-character string for our solution
def solver(quiz):
    try:
        z=demo_solver(quiz)
    except:
        return quiz
    z, repeated_l=deleteRepeated(z)
    z= np.reshape(z,(1,81)) ##transfer matrix to array
    w = np.array2string(z) ##transfer array to string
    w = w[2:-2]
    w = "".join(list(w))
    w = w.replace(" ",'')
    w = w.replace(".",'')
    w = w.replace("\n",'')
    if len(repeated_l)==0:
        return w
    else:
        #######################################################warm restart step 1
        
        try:
            z=demo_solver(w)
        except:
            return w
    
    
        z, repeated_l=deleteRepeated(z)
      
   
        z=np.reshape(z,(1,81)) ##transfer matrix to array
        w = np.array2string(z) ##transfer array to string
        w = w[2:-2]
        w = "".join(list(w))
        w = w.replace(" ",'')
        w = w.replace(".",'')
        w = w.replace("\n",'')
        if len(repeated_l)==0:
            
            return w
        else:
            ####################################################warm restart step 2
           
         
       
            try:
                z=demo_solver(w)
            except:
                return w
  
            z, repeated_l=deleteRepeated(z)
          
            if len(repeated_l)==0:
               
                z=np.reshape(z,(1,81)) ##transfer matrix to array
                w = np.array2string(z) ##transfer array to string
                w = w[2:-2]
                w = "".join(list(w))
                w = w.replace(" ",'')
                w = w.replace(".",'')
                w = w.replace("\n",'')
                
                return w
            else:
                for i in range(15):        
                    #################################################warm restart step 3
                    
                    quiz_old =  np.reshape([int(c) for c in quiz], (9,9))
                    quiznew = np.ones([9,9])*(quiz_old == 0) 
                    z = quiznew*z # Delete the clues in our deleted_repeated solved puzzle
                    while(True): # Randomly select a non-repeated filled-in number as a new clue to the original quiz
                        p = int(np.random.randint(0,8,1))
                        e = int(np.random.randint(0,8,1))
                        if z[p,e] != 0:
                            quiz_old[p,e] = z[p,e]
                            break
                    quiz_old = np.reshape(quiz_old,(1,81))
                    w=np.array2string(quiz_old)
                    w = w[2:-2]
                    w = "".join(list(w))
                    w = w.replace(" ",'')
                    w = w.replace(".",'')
                    w = w.replace("\n",'')
                    try:
                        z=demo_solver(w)
                    except:
                        return w
                    w= np.reshape(z,(1,81))##transfer matrix to array
                    w = np.array2string(w) ##transfer array to string
                    w = w[2:-2]
                    w = "".join(list(w))
                    w = w.replace(" ",'')
                    w = w.replace(".",'')
                    w = w.replace("\n",'')

                    z, repeated_l=deleteRepeated(z)
                    if len(repeated_l)==0:
                        
                        return w
                    if i == 14:    
                        return w
            

