## Swobodny stan końcowy i ustalony horyzont
Rozważamy problem sterowania dla systemu opisanego równaniem stanu
$$
\dot{x}(t) = f(x(t), \; u(t)), \; x(0) = x_0, \; t \in [0,\;T]
$$
Sterowanie jest w tym problemie jedyną zmienną decyzyjną. Sterowania **dopuszczalne** są funkcjami przedziałami ciągłymi. O $f$ zakładamy, że jest ciągła, ma pochodną ze względu na pierwszy argument i pochodna ta jest ciągła.
Dla każdego sterowania dopuszczalnego $u$ istnieje dokładnie jedno rozwiązanie równania stanu spełniające warunek początkowy. Zakładamy, że każde takie rozwiązanie jest określone w całym przedziale $[0, T]$. 
Wskaźnik jakości jest określony formułą
$$
Q(u) = q(x(T))
$$
Zakładamy, że $q$ jest różniczkowalna w sposób ciągły. 
Sterowanie dopuszczalne $\hat{u}$ nazywamy optymalnym, jeśli dla każdego sterowania zachodzi $Q(\hat{u}) \le Q(u)$.

Funkcja sprzężona definiowana jest jako rozwiązanie liniowego równania różniczkowego nazywanego sprzężonym:
$$
\dot{\psi}(t) = - \partial_1 f(x(t), u(t)) \psi(t)
$$
z warunkiem końcowym
$$
\psi(T) = -\partial q(x(T))
$$
Definiujemy Hamiltonian
$$
H(\psi(t), x(t), u(t)) = \psi^T f(x(t), u(t))
$$
Równanie sprzężona można zapisać także w postaci:
$$
\dot{\psi}(t) = -\partial_2 H(\psi(t), x(t), u(t))
$$
### Zasada maksimum Pontriagina
Dla $\hat{u}$ będącego sterowaniem optymalnym, $\hat{x}$ będącego optymalną trajektorią stanu wygenerowaną przez $\hat{u}$ i $\hat{\psi}$ będącego odpowiednią trajektorią sprzężoną, dla każdego $t \in [0, T]$ zachodzi 
$$
H(\hat{\psi(t)}, \hat{x(t)}, \hat{u(t)}) \ge H(\hat{\psi(t)}, \hat{x(t)}, v(t))
$$
czyli Hamiltonian jest maksymalizowany przez sterowanie optymalne.

## Całka we wskaźniku jakości
W zastosowaniach często spotyka się wskaźnik jakości zawierający wyrażenie całkowe:
$$
Q(u) = q(x(T)) + \int_0 ^T p(x(t), u(t))dt
$$
Załóżmy, że $p$ i $\partial_1 p$ są ciągłe. Przez proste zwiększenie wymiaru stanu problem z całkowym wskaźnikiem sprowadza się do zadania sformułowanego w poprzednim punkcie. Wprowadza się dodatkową współrzędna stanu:
$$
\dot{x}_{n+1}(t) = p(x(t), u(t)), \; x_{n+1}(0) = 0
$$
Wskaźnik całkowy możemy teraz zapisać w postaci
$$
Q(u) = q(x(T)) + x_{n+1}(T)
$$

## Zadanie ze swobodnym horyzontem
Rozważamy zadanie, gdzie horyzont $T \ge 0$ nie jest ustalony i obok sterowania pełni rolę zmiennej decyzyjnej. Wskaźnik jakości zależy jawnie od stanu końcowego i horyzontu
$$
Q(u, T) = q(x(T), T)
$$
Uzupełnimy warunki zasady maksimum wnioskiem wynikającym z optymalności horyzontu. W tym celu przedłużymy najpierw sterowanie $\hat{u}$ poza horyzont $\hat{T}$ w taki sposób, aby było ciągłe w punkcie $\hat{T}$. Dodatkowy warunek ma wtedy postać
$$
\partial_2 Q(\hat{u}, \hat{T}) = 0
$$
Podstawiając $\partial_1 q(\hat{x}(T), \hat{T}) = -\hat{\psi}(\hat{T})$ otrzymujemy
$$
H(\hat{\psi}, \hat{x}(\hat{T}), \hat{u}(\hat{T})) = \partial_2 q(\hat{x}(\hat{T}), \hat{T})
$$

## Problem kanoniczny
Nakładamy następujące ograniczenia na zmienne decyzyjne:
* $t_f \ge t_0$
* $g(t_0, x_0, t_f, x_f) \le 0$
* $h(t_0, x_0, t_f, x_f)=0$
* $x_0 = x(t_0)$
* $x_f = x(t_f)$
Minimalizowany jest wskaźnik jakości
$$
Q(u, t_0, x_0, t_f, x_f) = q(t_0, x_0, t_f, x_f)
$$
Definiujemy Hamiltonian:
$$
H(\psi, x, u, t) = \psi^T f(x, u, t)
$$
Zakładamy, że $(\hat{u}, \hat{t_0}, \hat{x_0}, \hat{t_f}, \hat{x_f})$ jest decyzją optymalną. 
Warunki transwersalności:
$$
\hat{\psi}(\hat{t_0}) = \partial_{x_{0}} (\lambda q + \mu^T g + \rho^T h)
$$
$$
\hat{\psi}(\hat{t_f}) = -\partial_{x_{f}} (\lambda q + \mu^T g + \rho^T h)
$$
$$
H[\hat{t_0}] = -\partial_{t_{0}} (\lambda q + \mu^T g + \rho^T h)
$$
$$
H[\hat{t_f}] = \partial_{t_{f}} (\lambda q + \mu^T g + \rho^T h)
$$
Jeśli w zadaniu nie ma ograniczeń nierównościowych, to pomija się wyraz $\mu^T q$.
Jeśli w zadaniu nie ma ograniczeń równościowych, to pomija się wyraz $\rho^T h$.
Jeśli w zadaniu nie ma ani ograniczeń nierównościowych, ani równościowych, to można przyjąć $\lambda = 1$.

Warunek nieujemności:
$$
\lambda \ge 0, \; \mu \ge 0
$$
Warunek nietrywialności:
$$
\lambda + ||\mu|| + ||\rho|| > 0
$$
