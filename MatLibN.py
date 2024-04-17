import numpy as np
'''
#A function to read the data from the files to obtain the matrix
#The file must be created in the following format, 
    #only one row must be written in a line, with elements separated by commas,
    #no comma is allowed to be placed at the end of a line
    #[1 0 0] is entered as 1,0,0
     [0 1 0]               0,1,0
     [0 0 1]               0,0,1
'''
def rd_mtrx(file_name):
    M=[]
    with open(str(file_name),'r') as file:
        for line in file:
            M+=[[]]
            fields= line.split(',')
            for num in fields:
                M[-1]+=[float(num)]
    return(M)

def tPose(M):
    if len(M.shape)==1:
        return(M)
    else:
        C=np.empty((M.shape[1],M.shape[0]))
        for i in range(len(M)):
            for j in range(len(M[i])):
                C[j,i]=M[i][j]
        return(C)

def Mat_product(X,Y,show=True):
    if len(X[0])!=len(Y):
        print('Product not defined')
    else:
        m=len(X)
        n=len(Y[0])
        Z=np.empty((m,n))
        for i in range(m):#populating the entries of the new matrix
            temp=0
            for l in range(n):
                for j in range(len(X[i])):
                    temp+=X[i][j]*Y[j][l]
                Z[i,l]=temp
                temp=0
        return(Z)

def partialPivot(mtrx,pivnum):
    piv=mtrx[pivnum,pivnum]
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    swp_nos=0
    if piv!=0:
        return(swp_nos)
    else:
        swp_nos+=1
        swap_row_ind=pivnum
        for i in range(pivnum,m):
            if mtrx[i,pivnum]>mtrx[swap_row_ind,pivnum]:
                swap_row_ind=i
        if mtrx[swap_row_ind,pivnum]==0:
            return("Unswappable zero pivot reached!")
        mtrx[pivnum,:],mtrx[swap_row_ind,:]=mtrx[swap_row_ind,:],mtrx[pivnum,:].copy()
        return(swp_nos)

def GaussJordan(mtrx):
    m=mtrx.shape[0]  #number of rows
    n=mtrx.shape[1]  #number of columns
    for i in range(m):
        pP=partialPivot(mtrx,i)
        if pP=="Unswappable zero pivot reached!":
            return("Unswappable zero pivot reached!")
        mtrx[i]=mtrx[i]/mtrx[i,i]
        for j in range(m):
            if j!=i:
                mtrx[j]+=-mtrx[j,i]*mtrx[i]
'''
A function that takes in the Augmented matrix and returs the inverse.
Note: It does not return the augmented matrix but only the inverse
'''

def inverse(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    Aug=np.empty((m,2*n))
    Aug[:,:n]=mtrx;Aug[:,n:]=np.eye(m)
    det_check=GaussJordan(Aug)
    if det_check=="Unswappable zero pivot reached!":
        return('Determinant does not exist')
    else:
        C=np.empty((m,n))
        for i in range(m):
            C[i]=Aug[i,m:]
        return(C)

'''
A function that calculates the determinant of a square matrix
'''

def Determinant(mtrx):
    m=mtrx.shape[0]  #number of rows
    n=mtrx.shape[1]   #number of columns
    temp=mtrx.copy()
    swp_nos=0
    for i in range(m):
        pP=partialPivot(temp,i)
        if pP=="Unswappable zero pivot reached!":
            return(0)
        swp_nos+=pP
        for j in range(i+1,m):
            temp[j]+=-temp[j,i]*temp[i]/temp[i,i]
    val=1
    for pivnum in range(m):
        val*=temp[pivnum,pivnum]
    return((-1)**(swp_nos)*val)

'''
LU decompostion(Doolittle/l_{ii}=1)
#Takes a matrix input and decomposes the input into a lower triangular matrix and an upper triangluar matrix
#The original matrix is LOST
'''
def LUdecomp_Doolittle(mtrx):
    m=mtrx.shape[0]         #number of rows
    n=mtrx.shape[1]         #number of columns 
    temp=mtrx
    for k in range(m):
        #This section will check if the decompostion is possible
        #And if required will use partial pivoting to attempt factorisation
        if Determinant(temp[0:k+1,0:k+1])==0:
            pP=partialPivot(temp,k) 
            if pP=="Unswappable zero pivot reached!":
                return("LU Decomposition not possible")
    #This section calculates the value of the elements in L and U       
    for i in range(1,m):                 #choosing row number
        for j in range(m):               #choosing column number
            if i<=j:                     #Calculating u_{ij}
                temp=0
                for k in range(i):
                    temp+=mtrx[i,k]*mtrx[k,j]
                mtrx[i,j]=mtrx[i,j]-temp #replacing the original matrix with the calculated value
            elif i>j:                    #Calculating l_{ij}
                temp=0
                for k in range(j):
                    temp+=mtrx[i,k]*mtrx[k,j]
                mtrx[i,j]=(mtrx[i,j]-temp)/mtrx[j,j]


'''
LU decompostion(Crout/u_{jj}=1)
#Takes a matrix input and decomposes the input into a lower triangular matrix and an upper triangluar matrix
#This section operates just like the previous one with a minor difference in the formula
#The original matrix is LOST
'''
def LUdecomp_Crout(mtrx):
    m=mtrx.shape[0]  #number of rows
    n=mtrx.shape[1]  #number of columns
    for k in range(m):
        #This section will check if the decompostion is possible
        #And if required will use partial pivoting to attempt factorisation
        if Determinant(mtrx[0:k+1,0:k+1])==0:
            pP=partialPivot(mtrx,k) 
            if pP=="Unswappable zero pivot reached!":
                return("LU Decomposition not possible")
    #This section caclulates the value of the elements in L and U
    for j in range(1,m):                           #choosing column number
        for i in range(m):                         #choosing row number
            if i>=j:                               #Calculating u_{ij}
                temp=0
                for k in range(j):
                    temp+=mtrx[i,k]*mtrx[k,j]
                mtrx[i,j]=mtrx[i,j]-temp            #Notice the difference in formula
            if i<j:                                 #calculating l_{ij}
                temp=0
                for k in range(i):
                    temp+=mtrx[i,k]*mtrx[k,j]
                mtrx[i,j]=(mtrx[i,j]-temp)/mtrx[i,i]#Notice the difference in formula

'''
Cholesky decompostion
#Only works for Hermitian and positive definite matrices
#To compute square root, 'math' library has been imported
#Takes a matrix input and decomposes the input into a lower triangular matrix and an upper triangluar matrix
#This section operates just like the previous ones with a minor difference in the formula
#The original matrix is LOST
'''

def CholDecomp(mtrx):
    m=mtrx.shape[0]         #number of rows
    n=mtrx.shape[1]         #number of columns
    for k in range(m):
        #This section will check if the decompostion is possible
        #And if required will use partial pivoting to attempt factorisation
        if Determinant(mtrx[0:k+1,0:k+1])==0:
            pP=partialPivot(mtrx,k) 
            if pP=="Unswappable zero pivot reached!":
                return("LU Decomposition not possible")
    #This section caclulates the value of the elements in L and U
    #we dont need to calculate every combination of i,j since the matrix is symmetric
    for i in range(m):
        for j in range(i,m):
            if i==j:                   #diagonal elements
                temp=0
                for k in range(i):
                    temp+=mtrx[i,k]**2
                mtrx[i,i]=np.sqrt(mtrx[i,i]-temp)
            if i<j:                    #calculating off diagonal elements
                temp=0
                for k in range(i):
                    temp+=mtrx[i,k]*mtrx[k,j]
                mtrx[i,j]=(mtrx[i,j]-temp)/mtrx[i,i]
                mtrx[j,i]=mtrx[i,j]    #since the matrix is symmetric


'''
backward and forward substitution for Doolitlle's method
'''

def backsubs_Doolittle(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    X=np.empty((m,1))
    for i in range(m-1,-1,-1):#moving backwards
        temp=0
        for j in range(i+1,m):
            temp+=mtrx[i,j]*X[j][0]
        X[i,0]=(mtrx[i,-1]-temp)/mtrx[i,i]
    return(X)
def forsubs_Doolittle(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    X=np.empty((m,1))
    for i in range(m):#moving forward
        temp=0
        for j in range(i):
            temp+=mtrx[i,j]*X[j][0]
        X[i,0]=(mtrx[i,-1]-temp)
    return(X)



'''
backward and forward substitution for Crout's method
'''

def backsubs_Crout(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    X=np.empty((m,1))
    for i in range(m-1,-1,-1):#moving backwards
        temp=0
        for j in range(i+1,m):
            temp+=mtrx[i,j]*X[j][0]
        X[i,0]=mtrx[i,-1]-temp
    return(X)
def forsubs_Crout(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    X=np.empty((m,1))
    for i in range(m):#moving forward
        temp=0
        for j in range(i):
            temp+=mtrx[i,j]*X[j][0]
        X[i,0]=(mtrx[i,-1]-temp)/mtrx[i,i]
    return(X)


'''
backward and forward substitution for Cholesky decomposition
'''
def backsubs_chol(mtrx):
    m=mtrx.shape[0]         #number of rows
    n=mtrx.shape[1]         #number of columns
    X=np.empty((m,1))
    for i in range(m-1,-1,-1):#moving backwards
        temp=0
        for j in range(i+1,m):
            temp+=mtrx[i,j]*X[j][0]
        X[i,0]=(mtrx[i,-1]-temp)/mtrx[i,i]
    return(X)
def forsubs_chol(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    X=np.empty((m,1))
    for i in range(m):
        temp=0
        for j in range(i):#moving forward
            temp+=mtrx[i,j]*X[j][0]
        X[i,0]=(mtrx[i,-1]-temp)/mtrx[i,i]
    return(X)


'''
Solve a system of equation using LU decomposition(Doolittle method)
AX=b
#takes A and b(separately) as input and returns X
#Original matrices are preserved
'''

def solve_system_Doolittle(A,b):
    m=A.shape[0]   #number of rows
    n=A.shape[0]   #number of columns
    LUb=np.empty((m,n+1))
    LUb[:,:-1]=A
    LUb[:,-1]=b.reshape(b.size)
    check=LUdecomp_Doolittle(LUb)#LU decompostion
    if check=="LU Decomposition not possible":
        return("No unique solution exists")
    y=forsubs_Doolittle(LUb)#forward substitution to solve Ly=b
    Uy=np.empty((m,n+1))
    Uy[:,:-1]=LUb[:,:-1]
    Uy[:,-1]=y.reshape(y.size)
    x=backsubs_Doolittle(Uy)#backward substitution to solve Ux=y
    return(x)
'''
Solve a system of equation using LU decomposition(Crout's method)
AX=b
#takes A and b(separately) as input and returns X
#Original matrices are preserved
'''

def solve_system_Crout(A,b):
    m=A.shape[0]   #number of rows
    n=A.shape[0]   #number of columns
    LUb=np.empty((m,n+1))
    LUb[:,:-1]=A
    LUb[:,-1]=b.reshape(b.size)
    check=LUdecomp_Crout(LUb)
    if check=="LU Decomposition not possible":
        return("No unique solution exists")
    y=forsubs_Crout(LUb)#forward substitution to solve Ly=b
    Uy=np.empty((m,n+1))
    Uy[:,:-1]=LUb[:,:-1]
    Uy[:,-1]=y.reshape(y.size)
    x=backsubs_Crout(Uy)#backward substitution to solve Ux=y
    return(x)

'''
Solve a system of equation using Cholesky decomposition
AX=b
#A is hermitian and positive definite
#takes A and b(separately) as input and returns X
#Original matrices are preserved
'''

def solve_system_chol(A,b):
    m=A.shape[0]   #number of rows
    n=A.shape[1]   #number of columns
    LUb=np.empty((m,n+1))
    LUb[:,:-1]=A
    LUb[:,-1]=b.reshape(b.size)
    check=CholDecomp(LUb)
    y=forsubs_chol(LUb)#forward substitution to solve Ly=b
    Uy=np.empty((m,n+1))
    Uy[:,:-1]=LUb[:,:-1]
    Uy[:,-1]=y.reshape(y.size)
    x=backsubs_chol(Uy)#backward substitution to solve UX=y
    return(x)

'''
Inverse using LU decomposition
#Calculates inverse by solving AX=I for each row of I
#Takes only the matrix (not augmented) as input and returns the inverse
#original matrix may be LOST
'''

def LUinverse(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    I=np.eye(m)           #Indentity matrix
    inv=np.empty((m,n))
    for i in range(m):
        temp=solve_system_Doolittle(mtrx,I[:,i].reshape(m))#Matrix([[ele[i]] for ele in I])
        inv[:,i]=temp.reshape(temp.size)#inv+=temp.Transpose().lol
    return(inv)


'''
Inverse using Cholesky decomposition
#Calculates inverse by solving AX=I for each row of I
#Takes only the matrix (not augmented) as input and returns the inverse
#original matrix is preserved
'''

def CholInverse(mtrx):
    m=mtrx.shape[0]   #number of rows
    n=mtrx.shape[1]   #number of columns
    I=np.eye(m)           #Indentity matrix
    inv=np.empty((m,n))
    for i in range(m):
        temp=solve_system_chol(mtrx,I[:,i].reshape(m))
        inv[:,i]=temp.reshape(temp.size)#inv+=temp.Transpose().lol
    return(inv)

################################
#Iterative methods for Matrices#
################################

def JacobiSolve(A,b,eps,x0=None):
    '''
    Solve Ax=b----(1) type of equations using Jacobi method
    -------------------
    inputs
        A : 2d square numpy array
            same as in eqn.(1)
        b : 2d column numpy array
            same as in eqn.(1)
        eps : float
            tolerance 
    Outputs
        x2 : 2d vertical float array
            solution 'x' in eqn.(1)
    '''
    if x0:
        x1=x0
    else:
        x1=np.zeros(b.size)
    while True:
        x2=np.zeros(b.size)
        for i in range(len(x1)):
            x2[i]=(b[i]-sum([A[i,l]*x1[l]*(i!=l) for l in range(b.size)]))/A[i,i]
        if sum((x2-x1)**2)<eps:
            break
        else:
            x1=x2
    return(x2)

def GaussSeidelSolve(A,b,eps,x0=['wow']):
    '''
    Solve Ax=b----(1) type of equations using Gauss-Seidel method
    -------------------
    inputs
        A : 2d square numpy array
            same as in eqn.(1)
        b : 2d column numpy array
            same as in eqn.(1)
        eps : float
            tolerance 
    Outputs
        x2 : 2d vertical float array
            solution 'x' in eqn.(1)
    '''
    if x0==['wow']:
        x=np.zeros(b.size)
    else:
        x=x0
    while True:
        Sum=0
        for i in range(len(x)):
            temp=(b[i]-sum([A[i,j]*x[j] for j in range(i)])-sum([A[i,j]*x[j] for j in range(i+1,b.size)]))/A[i,i]
            Sum+=(temp-x[i])**2
            x[i]=temp
        if Sum<eps:
            break
    return(x)


#################################
#Sparse Systems
#############################

def ConjGrad(A,b,eps,x0='no'):
    '''
    Solve Ax=b----(1) type of equations using Gauss-Seidel method
    -------------------
    inputs
        A : 2d square numpy array
            same as in eqn.(1)
        b : 2d column numpy array
            same as in eqn.(1)
        eps : float
            tolerance 
        x0 : 2d column numpy array
            initial guess value
    Outputs
        x : 2d vertical float array
            solution 'x' in eqn.(1)
    '''
    if x0=='no':
        x=np.zeros(b.size).reshape((b.size,1))
    else:
        x=x0
    #print(x)
    r=b-Mat_product(A,x)
    d=r
    i=1
    while i<=b.size:
        #print(f"i={i}")
        al=Mat_product(tPose(r),r)[0,0]/Mat_product(tPose(r),Mat_product(A,d))[0,0]
        x=x+al*d
        r=r-al*Mat_product(A,d)
        #print(x)
        if sum(r**2)<eps:
            break
        else:
            beta=-Mat_product(tPose(r),Mat_product(A,d))[0,0]/Mat_product(tPose(d),Mat_product(A,d))[0,0]
            d=r+beta*d
            i=i+1
    return(x)
        

def ConjGradinv(A,eps):
    '''
    Solve AB=I----(1) type of equations using Gauss-Seidel method
    -------------------
    inputs
        A : 2d square numpy array
            same as in eqn.(1)
        eps : float
            tolerance 
    Outputs
        B : 2d vertical float array
            solution 'B' in eqn.(1)
    '''
    x=np.zeros(A.shape)
    for j in range(A.shape[0]):
        b=np.zeros((A.shape[0],1))
        b[j,0]=1
        x[:,j]=ConjGrad(A,b,eps).reshape(A.shape[0])
    return(x)
        

#################################
#          Eigensystems         #
#################################

def PowerMethod(A,x0='wow',eps=0.0001):
    '''
    Find the largest eigenvalue and the corresponding eigenvector using the power method
    -------------------
    inputs
        A : 2d square numpy array
            Matrix whose eigenvalues are to be computed
        x0 : column vector
            guess column vector(optional)
        eps : float
            tolerance 
    Outputs
        (eVal,eVec) : 2-tuple
            largest eigenvalue and the corresponding eigenvector  
    '''
    if x0=='wow':
        x0=np.random.rand(A.shape[0],1)
    x=x0
    y=x
    l=1
    k=1
    while True:
        x=Mat_product(A,x)
        l0=1/Mat_product(tPose(x),y)[0,0]
        x=Mat_product(A,x)
        l0=l0*Mat_product(tPose(x),y)[0,0]
        if abs(l-l0)<eps:
            k+=1
            l=l0
            break
        else:
            l=l0
    return(l,x/np.sqrt(np.sum(x**2)))


def GramSchmidt(A):
    '''
    Use the Gram-Schmidt orthonormalisation procedure to construct a orthogonal matrix out of A
    -------------------
    inputs
        A : 2d square numpy array
            Matrix which is to orthogonalised
    Outputs
        Q : 2d square numpy array
            orthogonal matrix
    '''
    Q=np.empty(A.shape)
    temp=np.zeros(A.shape)
    for i in range(Q.shape[0]):
        Q[:,i][:,np.newaxis]=Mat_product(np.eye(A.shape[0])-temp,A[:,i][:,np.newaxis])
        Q[:,i]=Q[:,i]/np.sqrt(sum(Q[:,i]**2))
        temp+=Mat_product(Q[:,i][:,np.newaxis],tPose(Q[:,i][:,np.newaxis]))/Mat_product(tPose(Q[:,i][:,np.newaxis]),Q[:,i][:,np.newaxis])
    return(Q)

def QRfac(A):
    '''
    Use the Gram-Schmidt orthonormalisation procedure to factorize A=QR, where Q is orthogonal and R is upper triangular
    -------------------
    inputs
        A : 2d square numpy array
            Matrix which is to be factorize
    Outputs
        (Q,R) : tuple of 2d square numpy arrays
            Q and R as explained above
    '''
    Q=GramSchmidt(A)
    R=Mat_product(tPose(Q),A)    
    return(Q,R)

def eigenQR(A,N=1000):
    '''
    Compute Eigenvalues using QR-decomposition
    -------------------
    inputs
        A : 2d square numpy array
            Matrix whose eigenvalues are to be computed
        N : Number of iterations
    Outputs
        (eVals,eVecs) : 2-tuple containing all eigenvalues and respective eigenvectors(in columns)
    '''
    temp=np.eye(A.shape[0])
    for _ in range(N):
        Q,R=QRfac(A)
        A=Mat_product(R,Q)
        temp=Mat_product(temp,Q)
    return(np.diag(A),temp)