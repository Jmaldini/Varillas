import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

gama=1
k=0.001
h=0.001
lamb=(gama*k)/(h**2)
alfa= lambda t:t*0
beta= lambda t:t*0
f= lambda x: np.sin(np.pi*x)
x=np.linspace(0,1, int((1/h)+1))
diag=[1+lamb for i in range(len(x)-2)]
diag_u=[-0.5*lamb for i in range(len(x)-3)]
diag_d=[-0.5*lamb for i in range(len(x)-3)]
data=[diag,diag_u,diag_d]
A=diags(data,[0,1,-1]).toarray()
diag_01=[1-lamb for i in range(len(x)-2)]
diag_u1=[0.5*lamb for i in range(len(x)-3)]
diag_d1=[0.5*lamb for i in range(len(x)-3)]
data1=[diag_01,diag_u1,diag_d1]
B=diags(data1,[0,1,-1]).toarray()
u0=np.transpose(np.array([list(map(f,x[1:-1]))]))
plt.plot(x[1:-1],u0, label="Condición inicial")

for j in range (1,2):
    b1=[lamb*alfa(j+1)]
    b=[lamb*alfa(j)]
    for i in range(len(x)-4):
        b1.append(0)
        b.append(0)
    b1.append(lamb*beta(j+1))
    b1=np.array([b]).T
    b.append(lamb*beta(j))
    b=np.array([b]).T
    C=np.dot(B,u0)
    u1=np.dot(np.linalg.inv(A),C)
    u0=u1
    
plt.plot(x[1:-1],u1, label="Aproximación")
plt.legend()