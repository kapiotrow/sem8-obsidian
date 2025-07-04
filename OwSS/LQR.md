## Nieskończony horyzont czasowy
System opisany liniowym, stacjonarnym równaniem stanu:
$$
\dot{x}(t) = Ax(t) + Bu(t), \; t>0, \; x(0)=x_{0}
$$
Minimalizowany **wskaźnik jakości:**
$$
Q(u) = \frac{1}{2}\int_{0}^{\infty} (x(t)^T W x(t) + u(t)^T R u(t))dt
$$
Gdzie:
$$
W=W^T \ge 0, \; R=R^T>0
$$

Należy sprawdzić warunki **sterowalności i obserwowalności** (do istnienia rozwiązania potrzebna jest stabilizowalność i wykrywalność, ale sprawdzamy warunki silniejsze):
	**Sterowalność:**
$$
rank([B \; AB]) = n
$$
	**Obserwowalność:**
$$
rank(\begin{bmatrix} W \\ WA \end{bmatrix}) = n
$$
Gdzie $n$ to wymiar macierzy $A$.

**Równanie Ricattiego:**
$$
0 = KBR^{-1}B^TK - KA - A^TK - W
$$
**Optymalna wartość wskaźnika jakości:**
$$
Q(u_T, x_0) = \frac{1}{2} x_{0}^T K(0)x_0
$$
Równanie **regulatora optymalnego:**
$$
u(t, x_0) = -R^{-1}B^T Kx(t, x_0)
$$

## Skończony horyzont czasowy
System opisany liniowym, stacjonarnym równaniem stanu:
$$
\dot{x}(t) = Ax(t) + Bu(t), \; t>0, \; x(0)=x_{0}
$$
Minimalizowany **wskaźnik jakości:**
$$
Q(u) = \frac{1}{2}x(T)^T Px(T) + \frac{1}{2}\int_{0}^{\infty} (x(t)^T W x(t) + u(t)^T R u(t))dt
$$
Gdzie:
$$
P=P^T \ge 0, \; W=W^T \ge 0, \; R=R^T>0
$$
Należy sprawdzić warunki **sterowalności i obserwowalności** (do istnienia rozwiązania potrzebna jest stabilizowalność i wykrywalność, ale sprawdzamy warunki silniejsze):
	**Sterowalność:**
$$
rank([B \; AB]) = n
$$
	**Wykrywalność:**
$$
rank(\begin{bmatrix} W \\ WA \end{bmatrix}) = n
$$
Gdzie $n$ to wymiar macierzy $A$.

**Równanie Ricattiego:**
$$
\dot{K} = KBR^{-1}B^TK - KA - A^TK - W, \; K(T)=P
$$
Gdzie $K=K^T>=0$.

**Optymalna wartość wskaźnika jakości:**
$$
Q(u) = \frac{1}{2}x_{0}^T K(0) x_{0}
$$
**Regulator optymalny:**
$$
u(t) = -R^-1 B^T Kx
$$