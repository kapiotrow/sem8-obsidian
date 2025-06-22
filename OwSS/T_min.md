###### Rozdział 5 STEROWANIE MINIMALNOCZASOWE W SYSTEMIE LINIOWYM STACJONARNYM

System opisany liniowym, stacjonarnym równaniem stanu:
$$
\dot{x}(t) = Ax(t) + Bu(t), \; t>0, \; x(0)=x_{0}, \; x(T) = x_f 
$$
Oraz sterowanie jest ograniczone:
$$
u_i(t) \le u_{max}
$$
Wskaźnik jakości strerowania:
$$
Q(u, T) = T
$$
Hamiltonian:
$$
H(\psi(t), x(t), u(t)) = \psi(t)^T(Ax(t)+Bu(t))
$$
$$
\dot\psi(t) = − A^T\psi(t)
$$

$$

$$
### Rozwiązanie dla bang bang
Z rozwiązania ogólnego liniowego układu równań różniczkowych
$$
x(t) = e^{-At}x(0) + \int_{0}^{t}e^{A(t-\tau)}Bu(t) \; d\tau
$$

Podstawiając czas końcowy i czas przełączenia dla u typu bang-bang:
$$
x(T) = e^{-AT}x(0) + e^{AT}\int_{0}^{T}e^{A(-\tau)}Bu(t) \; d\tau
$$
$$
x_f = e^{-AT}x(0) + e^{AT}(\int_{0}^{t_s}e^{A(-\tau)}Bu_0 \; d\tau + \int_{t_s}^{T}e^{A(-\tau)}Bu_1 \; d\tau)
$$
Jeżeli $x_f = 0$:
$$
0 = x(0) + (\int_{0}^{t_s}e^{A(-\tau)}Bu_0 \; d\tau + \int_{t_s}^{T}e^{A(-\tau)}Bu_1 \; d\tau)
$$
