import math
#Custom error for impossible bracketing
class NoBracket(Exception):
    pass

'''
#Bracketing algorithm
#input: function form, any Bracket, beta(optional)
#output: bracket with f(a)*f(b)<0
'''
def bracket(f,a,b,beta=1.5):
    #implementing bracketing algorithm
    for i in range(12):
        if f(a)*f(b)<0:
            return((a,b))
        if abs(f(a))<abs(f(b)):
            a=a-beta*(b-a)
        elif abs(f(a))>abs(f(b)):
            b=b+beta*(b-a)
    #if bracketing fails
    if f(a)*f(b)>0:
        raise NoBracket("The function has no bracket in the vicinity,either choose a better",
            "bracket or check the function")

'''
#Finding roots using bisection method
#takes function form, bracket, precision as input
#gives root as output
#If required, it can also provide the iteration data(i.e., x_i with respect to iteration number) by passing a 
 relevant argument
'''
def bisection(f,a,b,err=10**(-5),iter_data=False):
    #i and x_i track the value of the ith iteration, N is the counter
    i=[]
    x_i=[]
    N=1
    #Implementing Bisection method
    while (b-a)/2>err:
        i.append(N)
        c=(a+b)/2
        if f(c)*f(a)<0:
            b=c
        elif f(c)*f(b)<0:
            a=c
        else:
            return("Bad bracket or function")
        x_i.append(c)
        N+=1
    if iter_data==True:
        return((i,x_i))
    return((a+b)/2)

'''
#Regula falsi method for finding root
#Takes function form, bracket, precision as input
#Provides root as output within certain precision
#If required, it can also provide the iteration data(i.e., x_i with respect to iteration number) by passing a 
 relevant argument
'''
def regula_falsi(f,a,b,err=10**(-5),iter_data=False):
    #i and x_i track the value of the ith iteration, N is the counter
    i=[]
    x_i=[]
    N=1
    #Implementing Regula Falsi method
    c=b-(b-a)*f(b)/(f(b)-f(a))
    while True:
        i.append(N)
        c_bef=c
        c=b-(b-a)*f(b)/(f(b)-f(a))    
        if f(a)*f(c)<0:
            b=c
            a=a
        elif f(a)*f(c)>0:
            b=b
            a=c
        x_i.append(c)
        N+=1
        if abs(c-c_bef)<err and abs(f(c))<err:
            if iter_data==True:
                return((i,x_i))
            return(c)


'''
#Symmetric derivative
#takes function form, point to be evaluated as input
#returns first derivative
'''
def sym_derivative(f,a,h=10**(-3)):
    return((f(a+h)-f(a-h))/(2*h))
'''
#symmetric double derivative
#takes function, point to be evaluated as input 
#returns double derivative
'''
def sym_dderivative(f,a,h=10**(-3)):
    return((f(a+h)+f(a-h)-2*f(a))/(h**2))
'''
#Solving Equation using Newton-Raphson method
#Takes function form and an initial guess as input
#Provides root as output within certain precision
#If required, it can also provide the iteration data(i.e., x_i with respect to iteration number) by passing a 
 relevant argument
'''
def newton_raphson(f,x_0,err=10**(-5),iter_data=False):
    #i and x_i track the value of the ith iteration, N is the counter
    i=[0]
    x_i=[x_0]
    N=0
    #This if condtional will take care if the guess is exactly the root
    if abs(f(x_0))==0:
        return(x_0)
    #Using Newton Raphson algorithm
    x=x_0-f(x_0)/sym_derivative(f,x_0,err*10**(-3))
    while True:
        N+=1
        i.append(N)
        x_bef=x
        x=x-f(x)/sym_derivative(f,x)
        x_i.append(x)
        if abs(x-x_bef)<err and abs(f(x))<err:
            if iter_data==True:
                return((i,x_i))
            return(x)

'''
Synthetic division 
#Takes the coefficient of the polynomial as input in the following format:
    f(x)=a_n*x^n+a_n-1*x^n-1+.....+a_0 is input as [a_n,a_n-1,....,a_0]
#Takes the extreme indices of the above list to be considered
#Takes the divisor:='div', i.e., (x-div)
#Returns the resulting coefficient list from which the coefficients can be read off as prescribed above
'''
def synth_div(coeff,start_index,end_index,div):
    for i in range(start_index+1,end_index+1):
        temp=coeff[i-1]
        coeff[i]=coeff[i]+div*temp
    return(coeff)
'''
Laguerre's method
#Takes Coefficient list, degree of the polynomial, guess as input
#returns root and deflated polynomial coefficient vector
'''
def Laguerre(coeff,deg,a,err=10**(-5)):
    #Creating the polynomial out of the coefficients
    f=lambda x: sum([coeff[i]*x**(deg-i) for i in range(deg+1)])
    n=1  
    # Implementing Laguerre's Method 
    while True:  
        if abs(f(a))<err:
            a=newton_raphson(f,a,err)
            synth_div(coeff,0,deg,a)
            return(a)
        else:
            G=sym_derivative(f,a)/f(a)
            H=G**2-sym_dderivative(f,a)/f(a)
            a=a-deg/max(G+math.sqrt((deg-1)*(deg*H-G**2)),G-math.sqrt((deg-1)*(deg*H-G**2)))
            n+=1      
'''
#This is the actual function which is to be called to solve an equation completely
#Takes coefficient list, degree, initial guess and precision as input
#returns a list containing the roots upto the required precision
'''
def LaguerreSolve(coeff,deg,a,err=10**(-5)):
    roots=[]
    while deg>1:
        roots.append(Laguerre(coeff,deg,a,err))
        deg=deg-1   
    roots.append(-coeff[1]/coeff[0]) 
    return(roots)

def fixedPoint(g,eps,x0):
    '''
    solves x=g(x) using fixed point method
    --------------------
    g : RHS of the equation
    eps : tolerance for loop termination
    x0 : initial guess
    '''
    x_p=x0
    x_n=g(x_p)
    while abs(x_n-x_p)>eps:
        x_p=x_n
        x_n=g(x_p)
    return(x_n)