import matplotlib.pyplot as plt
import math
import numpy as np
import Root_find_lib as rofl
#####################################
def ode_forwardeuler(f,x_0,y_0,h=0.01,N=100):
    '''
    solve ode using forward euler's method
    ----------------------------------
    solves ordinary differential equations of the form

    dy/dx = f(y(x),x)-------------(i)

    input: f(y(x),x),initial x, initial y

    output: set of (x,y) that satisfy the (i)
    '''
    x=[x_0]
    y=[y_0]
    for i in range(N):
        y.append(y[-1]+h*f(y[-1],x[-1]))
        x.append(x[-1]+h)
    return(x,y)
###################################################
def ode_backwardeuler(f,x_0,y_0,h=0.01,N=100):
    '''
    solve ode using backward euler's method
    ----------------------------------
    solves ordinary differential equations of the form

    dy/dx = f(y(x),x)-------------(i)

    input: f(y(x),x),initial x, initial y

    output: set of (x,y) that satisfy the (i)
    '''
    x=[x_0]
    y=[y_0]
    for i in range(N):
        x.append(x[-1]+h)
        yNR=rofl.newton_raphson(lambda ynr:ynr-y[-1]-h*f(ynr,x[-1]),y[-1])
        y.append(y[-1]+h*f(yNR,x[-1]))
    return(x,y)
########################################################3
def predCorr(f,x_0,y_0,h=0.01,N=100):
    """
    predictor-corrector method
    -----------------------------------
    solves ordinary differential equations of the form

    dy/dx = f(y(x),x)-------------(i)

    finds k1(prediction) and k2(correction) and takes their average to find next point

    input: f(y(x),x),initial x, initial y

    output: set of (x,y) that satisfy the (i)
    """
    x=[x_0]
    y=[y_0]
    for i in range(N):
        k1=h*f(y[-1],x[-1])
        yp=y[-1]+k1
        k2=h*f(yp,x[-1]+h)
        x.append(x[-1]+h)
        y.append(y[-1]+(k1+k2)/2)
    return(x,y)
#########################################################
def RK2_solve(f,x_0,y_0,h=0.01,N=100):
    """
    Runge-Kutta Method second order
    -----------------------------------
    solves ordinary differential equations of the form

    dy/dx = f(y(x),x)-------------(i)

    finds k1 and k2 and uses k2 to find the next point

    input: f(y(x),x),initial x, initial y

    output: set of (x,y) that satisfy the (i)
    """
    x=[x_0]
    y=[y_0]
    for i in range(N):
        k1=h*f(y[-1],x[-1])
        k2=h*f(y[-1]+k1/2,x[-1]+h/2)
        x.append(x[-1]+h)
        y.append(y[-1]+k2)
    return(x,y)
###################################################
def RK4_solve(f,x_0,y_0,h=0.01,N=100):
    """
    Runge-Kutta Method fourth order
    -----------------------------------
    solves ordinary differential equations of the form

    dy/dx = f(y(x),x)-------------(i)

    finds k1,k2,k3 and k4 to calculates the next point

    input: f(y(x),x),initial x, initial y

    output: set of (x,y) that satisfy the (i)
    """
    x=[x_0]
    y=[y_0]
    for i in range(N):
        k1=h*f(y[-1],x[-1])
        k2=h*f(y[-1]+k1/2,x[-1]+h/2)
        k3=h*f(y[-1]+k2/2,x[-1]+h/2)
        k4=h*f(y[-1]+k3,x[-1]+h)
        x.append(x[-1]+h)
        y.append(y[-1]+(k1+2*k2+2*k3+k4)/6)
    return(x,y)
######################################################
def RK4_solve_system(F,x,Y_0,h=0.01,N=100):
    """
    RK4 method for solving system of differential equation with one independent variable
    --------------------------------------------------
    solves system of differential equation of the form(capital means column matrix)

    dY/dx=F(Y,x)---------------------(1)

    Finds K1,K2,K3,K4 and calculates the next point

    input: F: column matrix containing all the functions
              e.g. [f1,f2......,fn]
                  where each fi takes input as fi([y_0,y_1.....y_n],x)
           x : initial value of the independent variable
           Y_0: column matrix containing all initial values of the dependent variables
              e.g. [[y_1_0],[y_2_0]........,[y_n_0]]
    output: list of lists containing the set of values that satisfy (1)
            e.g.  [[y_1_0,y_1_1,..............y_1_N],
                   [y_2_0,y_2_1,..............y_2_N],
                   .
                   .
                   [y_n_0,y_n_1,..............y_n_N],
                   [x_0,x_1,....................x_N]]
    
    """
    sol=Y_0+[[x]]
    n=len(sol)-1
    for _ in range(N):
       K1=[h*f_i([sol[k][-1] for k in range(n)],sol[-1][-1]) for f_i in F]
       K2=[h*f_i([sol[k][-1]+K1[k]/2 for k in range(n)],sol[-1][-1]+h/2) for f_i in F]
       K3=[h*f_i([sol[k][-1]+K2[k]/2 for k in range(n)],sol[-1][-1]+h/2) for f_i in F]
       K4=[h*f_i([sol[k][-1]+K3[k] for k in range(n)],sol[-1][-1]+h) for f_i in F]
       sol[-1].append(sol[-1][-1]+h)
       for i in range(n):
           sol[i].append(sol[i][-1]+(K1[i]+2*K2[i]+2*K3[i]+K4[i])/6)
    return(sol)
    
######################################################################
def bvp_shoot2(f,a,b,bvl,bvh,g1,g2,h=0.01,N=100,tol=0.0001):
    """ 
    Shooting method to solve 2nd order Ordinary BVP
    -------------------------------------------
    To solve d2y/dx2=f(x,y,y')---------(1)
    
    we reduce it into two equations

    dy/dx=z----------------(2)

    dz/dx=f(x,y,z)-------------(3)
    
    Then using a guess value of slope at lower boundary point, A solution is 'shot' at the upper boundary value
    Then using interpolation of the two guesses supplied, a possible solution to (1) is obtained that might 
    satisfy the boundary conditions

    input: function form as in (3) , lower bound, upper bound, boundary value at lower bound,boundary value at upper bound
    guess 1, guess 2

    Output: set of points satisfying (1) and the boundary conditions

    """
    m=int((b-a)/h)
    if m>N:
        N=m+10
    F=[lambda Y,x:Y[1],lambda Y,x:f(x,Y[0],Y[1])]
    Y_01=[[bvl],[g1]]
    Y_02=[[bvl],[g2]]
    x=a
    S1=RK4_solve_system(F,x,Y_01,h,N)
    S2=RK4_solve_system(F,x,Y_02,h,N)
    d1=bvh-S1[0][m]
    d2=bvh-S2[0][m]
    if d1>0 and d2<0:
        if abs(d1)<tol:
            plt.plot(S1[2],S1[0],label='Solution')
            return(S1)
        elif abs(d2)<tol:
            plt.plot(S2[2],S2[0],label='Solution')
            return(S2)
        yZh=S2[0][m]
        Zh=g2
        yZl=S1[0][m]
        Zl=g1
    elif d1<0 and d2>0:
        if abs(d1)<tol:
            plt.plot(S1[2],S1[0],label='Solution')
            return(S1)
        elif abs(d2)<tol:
            plt.plot(S2[2],S2[0],label='Solution')
            return(S2)
        yZh=S1[0][m]
        Zh=g1
        yZl=S2[0][m]
        Zl=g2
    else:
        if abs(d1)<tol:
            plt.plot(S1[2],S1[0],label='Solution')
            return(S1)
        elif abs(d2)<tol:
            plt.plot(S2[2],S2[0],label='Solution')
            return(S2)
        print("The guesses are not good enough")
        return(S1,S2)
    g=Zl+(Zh-Zl)*(bvh-yZl)/(yZh-yZl)
    Y=[[bvl],[g]]
    S=RK4_solve_system(F,x,Y,h,N)
    plt.plot(S1[2],S1[0],label='Guess 1')
    plt.plot(S2[2],S2[0],label='Guess 2')
    plt.plot(S[2],S[0],label='Solution')
    return(S,S1,S2)