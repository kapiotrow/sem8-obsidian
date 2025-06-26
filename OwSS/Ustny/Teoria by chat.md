# Teoria sterowania i optymalizacja – notatki

## 1. Jak formułuje się problem optymalizacji dynamicznej?

Problem optymalizacji dynamicznej formułuje się jako poszukiwanie takiej trajektorii stanu $$x(t)$$ i sterowania $$u(t)$$, która minimalizuje wskaźnik jakości, przy zachowaniu dynamiki układu i ograniczeń:

$$
\min_{u(\cdot)} J = \Phi(x(T)) + \int_{t_0}^{T} L(x(t), u(t), t)\,dt
$$

przy ograniczeniach:

$$
\dot{x}(t) = f(x(t), u(t), t), \quad x(t_0) = x_0
$$

$$
u(t) \in \mathcal{U}, \quad x(t) \in \mathcal{X}
$$

---

## 2. Co to jest sterowanie optymalne?

Sterowanie optymalne to funkcja sterowania $$u^*(t)$$, dla której odpowiadająca trajektoria stanu $$x^*(t)$$ optymalizuje wskaźnik jakości:

$$
J(x^*(\cdot), u^*(\cdot)) = \min_{u(\cdot) \in \mathcal{U}} J(x(\cdot), u(\cdot))
$$

---

## 3. Główne elementy analizy problemu optymalizacji dynamicznej

1. **Funkcja celu**: $$J = \Phi(x(T)) + \int L\,dt$$  
2. **Równania stanu**: $$\dot{x} = f(x, u, t)$$  
3. **Ograniczenia**:  
    $$u(t)\in \mathcal{U}$$$$x(t)\in \mathcal{X}$$  
   - warunki początkowe i końcowe  
4. **Zasady optymalności** (np. zasada maksimum Pontriagina)  
5. **Metody obliczeniowe** (bezpośrednie, pośrednie)

---

## 4. Zasada maksimum Pontriagina:
### Ustalony stan początkowy, ustalony horyzont, swobodny stan końcowy

- Dane:
  $$
  x(t_0)=x_0,\quad x(T)\ \text{swobodne}
  $$

  $$
  J = \int_{t_0}^{T} L(x,u,t)\,dt
  $$

- Hamiltonian:
  $$
  \mathcal{H}(x, u, \psi, t) = L(x,u,t) + \psi^T f(x,u,t)
  $$

- Równania:
  $$
  \dot{x}^* = \frac{\partial \mathcal{H}}{\partial \psi}, \quad \dot{\psi} = -\frac{\partial \mathcal{H}}{\partial x}
  $$

- Warunek maksimum:
  $$
  \mathcal{H}(x^*, u^*, \psi^*, t) = \max_{u\in\mathcal{U}} \mathcal{H}(x^*,u,\psi^*,t)
  $$

- Warunek brzegowy:
  $$
  \psi(T) = 0
  $$

---

## 5. Sprowadzenie problemu do postaci kanonicznej

- Wprowadzenie pomocniczej zmiennej:
  $$
  \dot{z}(t) = L(x,u,t),\quad z(t_0) = 0
  $$

- Rozszerzony wektor stanu:
  $$
  \tilde{x}(t) = \begin{pmatrix} x(t) \\ z(t) \end{pmatrix}
  $$

- Nowy wskaźnik jakości:
  $$
  J = \Phi(x(T)) + z(T)
  $$

---

## 6. Zasada maksimum: ustalony stan początkowy, swobodny horyzont i końcowy

- Dodatkowy warunek:
  $$
  \mathcal{H}(x^*(T),u^*(T),\psi^*(T),T) = 0
  $$

- Pozostałe zasady jak w punkcie 4.

---

## 7. Zasada maksimum: ogólne ograniczenia na stan początkowy i końcowy

- Warunki:
  $$
  h_0(x(t_0)) = 0,\quad h_T(x(T)) = 0
  $$

- Warunki brzegowe dla $$\psi$$:
  $$
  \psi(t_0) = \mu_0^T \frac{\partial h_0}{\partial x}(x(t_0)), \quad
  \psi(T) = -\mu_T^T \frac{\partial h_T}{\partial x}(x(T))
  $$

---

## 8. Zasada maksimum dla problemu kanonicznego

Stosuje się do rozszerzonego układu $$\tilde{x}=(x,z)$$, z rozszerzoną zmienną sprzężoną $$\psi=(\psi_x,\psi_z)$$.

---

## 9. Problem minimalnoczasowy – ogólne sformułowanie

- Wskaźnik:
  $$
  J = \int_{t_0}^{T} 1\,dt = T - t_0 \to\min
  $$

- Hamiltonian:
  $$
  \mathcal{H} = 1 + \psi^T f(x,u)
  $$

- Warunek maksimum:
  $$
  u^* = \arg\max_{u\in\mathcal{U}} \psi^T f(x,u)
  $$

---

## 10. Problem minimalnoczasowy – system liniowy stacjonarny

- System:
  $$
  \dot{x} = A x + B u, \quad u\in \{u:\|u\|_\infty\le1\}
  $$

- Zestaw twierdzeń:
  1. Istnienie sterowania optymalnego  
  2. Zasada maksimum  
  3. Charakter sterowania bang–bang  
  4. Ograniczenie liczby przełączeń  
  5. Warunki kontynuacji  
  6. Stabilność rozwiązania

---

## 11. Twierdzenie o liczbie przełączeń w sterowaniu minimalnoczasowym

Dla układu wymiaru $$n$$ – maksymalnie $$n-1$$ przełączeń sterowania.

---

## 12. Synteza regulatora minimalnoczasowego

Wyznaczanie strategii sterowania $$u(t)$$ według struktury bang–bang i momentów przełączeń.

---

## 13. Problem liniowo-kwadratowy – horyzont skończony

- Wskaźnik:
  $$
  J = x^T(T)F x(T) + \int_{t_0}^{T}(x^T Q x + u^T R u)\,dt
  $$

- System:
  $$
  \dot{x}=Ax+Bu
  $$

- Sterowanie:
  $$
  u^*(t) = -R^{-1}B^T P(t)x(t)
  $$

- Równanie Riccatiego:
  $$
  \dot{P} = -A^T P - P A + P B R^{-1}B^T P - Q,\quad P(T)=F
  $$

---

## 14. Problem liniowo-kwadratowy – horyzont nieskończony

- Stacjonarne ARE:
  $$
  A^T P + P A - P B R^{-1}B^T P + Q = 0
  $$

- Sterowanie:
  $$
  u^* = -R^{-1}B^T P x
  $$

---

## 15. Stabilizacja systemu nieliniowego – zastosowanie LQR

- Liniaryzacja wokół $$x_e$$:
  $$
  \dot{\delta x} = A_e \delta x + B_e \delta u
  $$

- Sterowanie:
  $$
  u = -K \delta x
  $$

---

## 16. Optymalizacja czasów przełączeń

W problemach typu **bang–bang**, sterowanie przyjmuje wartości skrajne (np. $$u = \pm u_{\text{max}}$$), a decyzje sprowadzają się do wyznaczenia **czasów przełączeń**.

### Postać sterowania:

$$
u(t) = \begin{cases}
u_{\text{max}}, & t \in [t_0, t_1) \cup [t_2, t_3) \cup \dots \\
-u_{\text{max}}, & t \in [t_1, t_2) \cup [t_3, t_4) \cup \dots
\end{cases}
$$

### Zmiennymi decyzyjnymi są:
$$
\{t_1, t_2, \dots, t_N\}
$$

### Metody:
- Gradient czasów przełączeń
- Zagnieżdżone iteracje: optymalizacja $$t_k$$
 - integracja
- Hybrydowe podejścia strukturalne

**Zalety:** mała liczba zmiennych.  
**Wady:** brak gładkości, trudność w inicjalizacji.

---

## 17. Bezpośrednie metody obliczeniowe

Dyskretne podejście do problemu sterowania optymalnego:

### Kroki:
1. Siatka czasowa: $$t_0, t_1, \dots, t_N$$
2. Aproksymacja:
   - schodkowa (stała)
   - łamana (liniowa)
   - wielomianowa
3. Zamiana równań stanu na ograniczenia

### NLP:
$$
\min_{x_k, u_k} \sum_{k=0}^{N-1} L(x_k, u_k, t_k)\cdot\Delta t + \Phi(x_N)
$$

przy:
$$
x_{k+1} = f_d(x_k, u_k, t_k), \quad x_0 = x_{\text{start}}
$$

**Algorytmy:** SQP, IPOPT, metoda kolokacji.

---
## 18. Metoda aproksymacji schodkowej

$$
u(t)\approx \text{const}\quad \text{na }[t_k,t_{k+1})
$$

---

## 19. Metoda aproksymacji łamaną

$$
u(t)\approx a_k t + b_k\quad \text{na }[t_k,t_{k+1}]
$$

---

## 20. Metoda kolokacji bezpośredniej

Metoda kolokacji jest jedną z najważniejszych bezpośrednich metod optymalizacji w teorii sterowania. Polega na aproksymacji trajektorii stanu i sterowania za pomocą funkcji bazowych, np. wielomianów Lagrange’a, oraz sprowadzeniu problemu sterowania optymalnego do dużego, rzadkiego problemu optymalizacji nieliniowej (NLP).

### Aproksymacja:

Stan i sterowanie są reprezentowane jako interpolacje wielomianowe:

$$
x(t) \approx \sum_{i=0}^{d} x_i \, \ell_i(t), \quad u(t) \approx \sum_{i=0}^{d} u_i \, \ell_i(t)
$$

gdzie $$\ell_i(t)$$ to wielomiany Lagrange’a wyznaczone w węzłach kolokacyjnych.

### Warunki kolokacyjne:

W punktach kolokacyjnych $$t_c$$ wymusza się spełnienie równań stanu:

$$
\dot{x}(t_c) = f(x(t_c), u(t_c), t_c)
$$

Dla każdej trajektorii między $$t_k$$ i $$t_{k+1}$$ generowane są algebraiczne ograniczenia wynikające z równań różniczkowych.

### Uwagi praktyczne:

- Często stosuje się **pseudospektralne kolokacje** (np. punkty Gaussa–Lobatto), które zapewniają wysoką dokładność globalną przy niewielkiej liczbie węzłów.
- Problem końcowy jest przekształcany w **sparsyfikowany NLP** i rozwiązywany za pomocą solverów (np. IPOPT, SNOPT).

**Zalety:**
- Wysoka dokładność aproksymacji
- Jednolity schemat dla systemów liniowych i nieliniowych

**Wady:**
- Duży rozmiar NLP
- Potrzeba zaawansowanych solverów i dobrej inicjalizacji

---

## 21. Pośrednie metody obliczeniowe sterowania optymalnego

Pośrednie metody wynikają z teorii warunków koniecznych optymalności, takich jak **zasada maksimum Pontriagina**. Rozwiązanie problemu optymalnego sterowania sprowadza się do rozwiązania układu równań różniczkowych z warunkami brzegowymi (BVP).

### Układ warunków koniecznych:

$$
\begin{cases}
\dot{x}(t) = \frac{\partial \mathcal{H}}{\partial \psi}, \\
\dot{\psi}(t) = -\frac{\partial \mathcal{H}}{\partial x}, \\
u^*(t) = \arg\max_{u \in \mathcal{U}} \mathcal{H}(x(t), u, \psi(t), t)
\end{cases}
$$

gdzie $$\mathcal{H}$$ to Hamiltonian problemu, a $$\psi(t)$$ to zmienna sprzężona (kosztat).

### Warunki brzegowe:

Zależą od formy problemu. Dla ustalonego stanu początkowego i swobodnego końcowego np.:

$$
x(t_0) = x_0, \quad \psi(T) = 0
$$

lub bardziej ogólne warunki wynikające z ograniczeń na końcowy stan lub czas.

### Metoda numeryczna – shooting:

- **Single shooting**: zgadujemy wartość $$\psi(t_0)$$ i integrujemy układ równań. Warunki końcowe służą do korekty zgadywanej wartości.
- Czułość: metoda jest silnie niestabilna dla długich horyzontów lub sztywnych układów.

**Zalety:**
- Wysoka dokładność teoretyczna
- Naturalne odwzorowanie struktury optymalności

**Wady:**
- Problemy ze zbieżnością i stabilnością
- Wymaga precyzyjnej inicjalizacji i dobrego solvera BVP

---

## 22. Pośrednia metoda strzałów wielokrotnych (multiple shooting)

Metoda multiple shooting jest rozszerzeniem single shooting, zaprojektowanym w celu poprawy stabilności numerycznej i zbieżności przy rozwiązywaniu układów warunków koniecznych (BVP).

### Idea:

1. Dzielimy horyzont czasowy na $$N$$ podprzedziałów: $$[t_0, t_1], [t_1, t_2], \dots, [t_{N-1}, T]$$
2. Na każdym przedziale wyznaczamy niezależnie:
   - początkowy stan $$x_k$$
   - trajektorię na przedziale poprzez całkowanie równań stanu
3. Wymuszamy **ciągłość trajektorii** pomiędzy przedziałami:

$$
x_{k+1}^{\text{start}} = x_k^{\text{end}} \quad \text{dla } k=0,\dots,N-1
$$

Dodatkowo można również nałożyć warunki na $$\psi$$, ale w wielu implementacjach zmienne sprzężone są eliminowane przez zastosowanie równań sensitivności.

### NLP:

Problem optymalizacji zawiera zmienne decyzyjne:
- $$x_k$$ (początkowy stan każdego segmentu)
- $$u(t)$$ (parametry sterowania)
- oraz ograniczenia dopasowujące stany na granicach segmentów

**Zalety:**
- Poprawa stabilności numerycznej
- Lepsze możliwości inicjalizacji
- Równoległość obliczeń na segmentach

**Wady:**
- Zwiększony rozmiar problemu NLP
- Wysoka złożoność przy dużej liczbie segmentów

---

