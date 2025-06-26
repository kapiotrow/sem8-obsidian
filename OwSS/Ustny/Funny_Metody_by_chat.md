
## 1. Bezpośrednie metody obliczeniowe: Optymalizacja czasów przełączeń

### 🧠 Zasada działania

Rozważamy sterowanie optymalne, w którym funkcja sterująca przyjmuje wartości krańcowe:

$$ u(t) \in \{u_{\min}, u_{\max}\}$$ Sterowanie tego typu nazywa się **brzegowym** lub **bang-bang**. Decyzjami nie są wartości funkcji $u(t)$, ale **czasy przełączeń** $$\tau = (\tau_1, \tau_2, \dots, \tau_M)$$, gdzie $u$ zmienia się z $u_{\min}$ na $u_{\max}$ lub odwrotnie. System dynamiczny ma postać:

$$
\dot{x}(t) = f_0(x(t)) + f_1(x(t))u(t), \quad x(0) = x_0
$$
Funkcjaprzełączająca: Funkcja przełączająca:
$$
\varphi(t) = \psi^T(t) f_1(x(t))
$$
Warunekmaksimum: Warunek maksimum:
$$
u(t) = \begin{cases}  
u_{\max} & \text{gdy } \varphi(t) < 0 \  
u_{\min} & \text{gdy } \varphi(t) > 0  
\end{cases}$$

### ⚙️ Algorytm

1. **Inicjalizacja**:
    
    - wybierz $u_0 \in {u_{\min}, u_{\max}}$
        
    - ustaw liczbę przełączeń $M = 0$
        
2. **Optymalizacja**:
    
    - dla danego $M$ minimalizuj wskaźnik Σ(u0,τ1,…,τM)\Sigma(u_0, \tau_1, \dots, \tau_M) np. metodą BFGS.
        
    - wylicz gradient:
        $$
        ∂Σ∂τi=H(x(τi−),ψ(τi−),ui−1)−H(x(τi+),ψ(τi+),ui)\frac{\partial \Sigma}{\partial \tau_i} = H(x(\tau_i^-), \psi(\tau_i^-), u_{i-1}) - H(x(\tau_i^+), \psi(\tau_i^+), u_i) $$
    - bierz kierunek $d$ i wyznacz maksymalny krok $s_{\max}$ spełniający warunki nierówności.
        
3. **Warunek sprawdzający**:
    
    - czy sterowanie spełnia warunek maksimum?
        
        - jeśli nie: generacja szpilkowa
            
        - jeśli tak: koniec
            
4. **Generacja szpilkowa**:
    
    - dodaj nowe przełączenie w miejscu minimum $$2u(t)−umin⁡−umax⁡∣⋅∣φ(t)∣|2u(t) - u_{\min} - u_{\max}| \cdot |\varphi(t)$$
        
5. **Redukcja**:
    
    - usuń przełączenie, jeśli nie wnosi poprawy lub leży na brzegu dopuszczalnego obszaru
        

### ✅ Zalety

- przejrzysta struktura,
    
- możliwość użycia efektywnych metod optymalizacji (BFGS, L-BFGS-B).
    

### ❌ Wady

- wymaga dobrej inicjalizacji,
    
- trudności przy optymalizacji liczby przełączeń.
    

---

## 2. Prosta pośrednia metoda strzałów (Rozdz. 12)

### 🧠 Zasada działania

Rozwiązujemy dwupunktowy problem brzegowy dla układu kanonicznego:
$$
z˙(t)=F(z(t))=[f(x(t),u(t))−∂H∂x(x(t),u(t),ψ(t))],z(0)=[x0p]\dot{z}(t) = F(z(t)) = \begin{bmatrix} f(x(t), u(t)) \\ -\frac{\partial H}{\partial x}(x(t), u(t), \psi(t)) \end{bmatrix}, \quad z(0) = \begin{bmatrix}x_0 \\ p\end{bmatrix}
$$
Warunek końcowy:
$$
ψ(T)+∇q(x(T))=0⇒Φ(p)=0\psi(T) + \nabla q(x(T)) = 0 \quad \Rightarrow \Phi(p) = 0
$$
### ⚙️ Algorytm

1. **Zainicjuj** $p = p_0$
    
2. **Iteracja Newtona**:
    $$
    p(k+1)=p(k)−[Φ′(p(k))]−1Φ(p(k))p^{(k+1)} = p^{(k)} - [\Phi'(p^{(k)})]^{-1} \Phi(p^{(k)})
    $$
3. **Obliczenie pochodnej**:
    
    - podejście wariacyjne: integracja układu
        $$
        ξ˙(t)=∂F∂z(z(t))ξ(t),ξ(0)=I\dot{\xi}(t) = \frac{\partial F}{\partial z}(z(t)) \xi(t), \quad \xi(0) = I
        $$
    - podejście sprzężone:
        $$
        η˙(t)=−(∂F∂z(z(t)))Tη(t),η(T)=[∇q(x(T))I]\dot{\eta}(t) = -\left(\frac{\partial F}{\partial z}(z(t))\right)^T \eta(t), \quad \eta(T) = \begin{bmatrix} \nabla q(x(T)) \\ I \end{bmatrix}
        $$
    - wtedy: $Φ′(p)=ηT(0)\Phi'(p) = \eta^T(0)$
        
4. **Zbieżność**: gdy $|\Phi(p)| < \varepsilon$
    

### ✅ Zalety

- szybka zbieżność blisko optimum
    

### ❌ Wady

- wymaga dobrej inicjalizacji $p$
    
- czułość na błędy numeryczne
    

---

## 3. Pośrednia metoda strzałów wielokrotnych (Rozdz. 12)

### 🧠 Zasada działania

Zamiast jednego długiego problemu brzegowego, dzieli się przedział $[0, T]$ na $N$ mniejszych i rozwiązuje układy lokalnie z warunkami dopasowania:
$$
zi(ti+)=zi+1(ti+)z_i(t_i^+) = z_{i+1}(t_i^+)
$$
### ⚙️ Algorytm

1. **Podział czasu**: $t_0 = 0 < t_1 < \dots < t_N = T$
    
2. **Na każdym przedziale**:
    
    - solve: $\dot{z}_i = F(z_i),\ z_i(t_i) = \sigma_i$
        
3. **Warunki dopasowania**: $\sigma_{i+1} = z_i(t_{i+1})$
    
4. **Rozwiązywanie całego układu** jako systemu równań algebraicznych metodą Newtona.
    

### ✅ Zalety

- rozszerzony obszar zbieżności
    
- lepsza stabilność numeryczna
    

### ❌ Wady

- wysoki koszt obliczeniowy
    

---

## 4. Metoda Borelowskiego – Minimalnoczasowa (Rozdz. 13)

### 🧠 Zasada działania

Dla systemu z ograniczonym sterowaniem:
$$
x˙(t)=f0(x(t))+f1(x(t))u(t),u(t)∈[−umax⁡,umax⁡]\dot{x}(t) = f_0(x(t)) + f_1(x(t))u(t), \quad u(t) \in [-u_{\max}, u_{\max}]
$$
Hamiltonian:
$$
H(x,u,ψ)=ψTf0(x)+ψTf1(x)uH(x, u, \psi) = \psi^T f_0(x) + \psi^T f_1(x) u
$$
Sterowanie brzegowe:
$$
u(t) = u_{\max} \cdot \text{sgn}(\psi^T f_1(x))$$ ### ⚙️ Algorytm (wariant najprostszy) 1. **Założenie liczby przełączeń $k = n - 1$** 2. **Zmienna decyzyjna**: wektor czasów przełączeń $\tau = (\tau_1, \dots, \tau_k)$ 3. **Równanie dopasowania**:
$$
\Phi(\tau) = x(T; \tau) - x_f = 0
$$
4.∗∗MetodaNewtona∗∗:4. **Metoda Newtona**:
$$
\tau^{(i+1)} = \tau^{(i)} - [\Phi'(\tau^{(i)})]^{-1} \Phi(\tau^{(i)})
$$
5. **Wyznaczenie pochodnych**: przez układy wariacyjne lub scałkowanie macierzy fundamentalnej wstecz. ### ✅ Zalety - przystosowana do zadań minimalnoczasowych - pozwala uwzględnić warunek maksimum ### ❌ Wady - niestabilność numeryczna - metoda działa dobrze tylko przy dobrej inicjalizacji --- ## 5. Generacje szpilkowe i redukcje przełączeń ### Generacje szpilkowe - dodanie przełączeń w miejscach, gdzie funkcja przełączająca silnie narusza warunek maksimum:
$$
\varphi(t) = \psi^T(t) f_1(x(t)) \not= 0
$$
- nowe czasy przełączeń dodaje się w parze (np. $\tau_k = \theta$, $\tau_{k+1} = \theta$) ### Redukcje przełączeń - usunięcie przełączenia, gdy: - nie poprawia wskaźnika - jest nieaktywny w sensie warunku maksimum --- W razie potrzeby mogę dopisać przykłady numeryczne lub kod implementacyjny w Pythonie lub MATLABie.