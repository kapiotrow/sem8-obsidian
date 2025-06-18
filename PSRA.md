### Omów budowę części procesorowej układu ARM - APU
**APU** (Accelerated Processing Unit) - zintegrowany układ scalony integrujący CPU, GPU, rzadziej także NPU, DSP.
<img src="Pasted image 20250618102900.png" alt="Image" width="500">

Zalety:
* wydajność
* niski pobór mocy
* tańsza produkcja
* efektywna komunikacja między CPU a GPU.

### Co to jest system SoC, podaj różnice w stosunku do architektury opartej o integrację na poziomie PCB
**SoC (System-on-a-Chip)** to układ scalony, który integruje **wiele komponentów systemu komputerowego** w **jednym chipie**. Typowy SoC zawiera:
- **CPU** – procesor centralny (np. ARM Cortex),
- **GPU** – procesor graficzny,
- **RAM** (czasem tylko kontroler RAM),
- **modemy** (np. LTE, Wi-Fi, Bluetooth),
- **kontrolery wejścia/wyjścia** (USB, HDMI, audio, itp.),
- **NPU / AI engine**, DSP i inne specjalizowane jednostki.

| SoC                                                    | PCB                                                            |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| Wszystkie główne komponenty w jednym układzie scalonym | Komponenty w oddzielnych chipach na wspólnej plytce drukowanej |
| Bardzo mały rozmiar                                    | Większy rozmiar                                                |
| Niski pobór mocy                                       | Zwykle wyższy pobór mocy                                       |
| Brak możliwości wymiany komponentów                    | Możliwość wymiany komponentów, elastyczność, rozbudowywalność  |
| Niższy koszt produkcji przy większej skali             | Potencjalnie wyższy koszt produkcji                            |
| Użycie: smartfony, laptopy, IoT, embedded              | PC, serwery                                                    |

### Omów podstawowe komponenty części programowalnej (PL) układu Zynq
* **CLB** (Configurable Logic Block) - LUT, rejestry, multipleksery
* **DSP** (Digital Signal Processor)
* **BRAM** - wbudowana pamięć blokowa o dużej przepustowości
* **CMT** (Clock Management Tiles) - PLL, MMCM
* **Interfejsy AXI**
* **DMA** (Direct Memory Access) - sterowniki pozwalające na bezpośredni dostęp do pamięci (bez angażowania procesora)
* **IO Blocks**

### Wskaż podstawową różnicę pomiędzy układami homogenicznymi i heterogenicznym, wymień ich wady i zalety.
**Układy homogeniczne** składają się z identycznych elementów obliczeniowych. 
Zalety:
* łatwe zrównoleglanie procesów, gdzie każdy wątek jest taki sam i jest ich odpowiednia liczba 
* łatwe rozdzielanie wątków pomiędzy elementy obliczeniowe (bo są takie same)
* ogólnie łatwe projektowanie, implementacja i debugging
Wady:
* wszystkie elementy są takie same, żaden nie specjalizuje się w konkretnym typie obliczeń
**Układy heterogeniczne** składają się z elementów obliczeniowych różnego rodzaju.
Zalety:
* możliwość przydzielenia zadań dostosowanych do zróżnicowanych zdolności elementów obliczeniowych
* większa elastyczność
Wady:
* zadanie rozdzielenia zadań pomiędzy elementy obliczeniowe jest trudne
* ogólnie projektowanie, implementacja i debuggowanie jest trudne

### Przedstaw cechy charakterystyczne układów FPGA i procesorów CPU
**FPGA**:
* elastyczność struktury
* duże możliwości zrównoleglania obliczeń
* dużo IO
* deterministyczne opóźnienie
* trudniejszy w projektowaniu, implementacji, debuggowaniu
* trudność implementacji algorytmów bardzo zależnych od danych
* niższy pobór mocy
**CPU:**
* sekwencyjne przetwarzanie danych
* ograniczony zbiorem instrukcji 
* wysoka dostępność narzędzi 
* niższa elastyczność

### Omów Xilinx AI engines dostępne w układach Versal/ACAP
AI Engines to dwuwymiarowa macierz składająca się z dziesiątek lub setek jednostek AI Engine. Pojedyncza jednostka AI Engine składa się z procesora typu Very Long Instruction Word, Single Instruction Multiple Data, zaprojektowanego z myślą o zastosowaniach ML i zaawansowanego przetwarzania sygnałów. Każda jednostka posiada także pamięć programu, pamięć danych, procesor RISC i różne interfejsy łączące ją z innymi jednostkami AI Engine oraz pozostałymi elementami SoC.
AI Engines wspierają obliczenia na INT8/16/32, CINT16/32 i FP32. Mają także dedykowane elementy sprzętowe dla obliczeń FFT.

### Omów podstawowe wyzwanie związane z przetwarzaniem strumienia o rozdzielczości 4K/UHD
Obraz o takiej rozdzielczości to przede wszystkim bardzo dużo danych. W przypadku działania w rygorze czasowym (np. systemy czasu rzeczywistego), przetworzenie takiej liczby pikseli stanowi problem.
W FPGA istnieją sprzętowe ograniczenia prędkości zegara i ilości dostępnych zasobów logicznych, dlatego zaprojektowanie architektury przetwarzającej klatki obrazu o rozdzielczości 4K wymaga zastosowania przetwarzania potokowo-równoległego (przetwarzanie czysto potokowe jest zbyt wolne). W takim trybie, piksele przetwarzane są w kilku (dwóch, może czterech) równoległych potokach, co pozwala na osiągnięcie zadowalających FPS.
Zastosowanie przetwarzania potokowo-równoległego powoduje problemy przy stosowaniu algorytmów używających kontekstu (kontekst trzeba albo współdzielić między potokami, albo dublować, a na FPGA nie ma dużo pamięci). Kolejnym wyzwaniem jest synchronizacja potoków, w przypadku gdy nie są one identyczne. 

### Omów możliwości komunikacji pomiędzy częścią programowalną, a procesorem ARM
Przede wszystkim **AXI** (Advanced eXtensible Interface) - standard w komunikacji pomiędzy PS a PL. Cechuje się wysoką przepustowością. 
* w pełnej wersji obsługuje **pipelining** (może wysłać wiele instrukcji zanim wróci odpowiedź), max szerokość szyny danych to 1024 bity, 5 kanałów
* w wersji Lite ma cztery kanały, jest prostszy, instrukcje stanowią podzbiór pełnej wersji
* w wersji Stream nie ma burstów, dane są pakowane w pakiety, ramki i streamy, nie ma ograniczenia na długość przesyłanych danych (przesył może być ciągły); nie ma sygnałów adresowych
* AXI DMA Controller

### Omów różnice pomiędzy Zynq, Zynq US+ oraz Veral/ACAP
**Zynq:**
* jeden procesor
* starszy, tańszy, wystarczający do wielu rozwiązań
**Zynq Ultrascale+:**
* APU
* Real-Time PU
* droższy, nowszy, mocniejszy
**AMD Versal:**
* najnowsza architektura z trzech omawianych
* NoC (Network on Chip) - bardzo szybka, zintegrowana ścieżka przesyłu danych pomiędzy PL a PS oraz kontrolerem pamięci DDR4 
* AI Engines
* Dual-Core APU
* Dual-Core RT PU

### Wymień jakie potencjalne problemy mogą pojawić się w przypadku przeniesienia części obliczeń do koprocesora sprzętowego
Przy przenoszeniu obliczeń pomiędzy jednostkami należy w pierwszej kolejności wziąć pod uwagę koszt przesyłu danych pomiędzy jednostkami - operacje dostępu do pamięci są często czasochłonne, magistrale mają ograniczoną przepustowość.


### Omów potencjalne zastosowanie FPGA/Zynq jako platformy obliczeniowej dla drona (zalety/wady) - przykład to zaprezentowane aplikacje lądowania
FPGA dobrze nadaje się do zaawansowanego przetwarzania sygnałów (przede wszystkim danych wizyjnych, ale też do fuzji z innymi czujnikami) ze względu na duże zasoby IO, możliwość przetwarzania potokowego i rónoległego, DSP wbudowane w fabrica i dużą elastyczność. Układy FPGA są też (potencjalnie) mniej mocożerne.
Wyrzucenie cięższych obliczeń z CPU na FPGA powoduje zwolnienie zasobów CPU, które może być potrzebne do celów zarządzania misją. 


______________________________________________________________
Omów wyzwania związane z implementacją sprzętową algorytmów segmentacji obiektów 
Omów wyzwania związane z implementacją sprzętową algorytmów (FPGA, potokowa) algorytmów indeksacji obiektów pierwszoplanowych (wskaż do czego może ew. przydać się kompresja)
Omów wyzwania związane ze sprzętową implementacją algorytmów (FPGA, potokowa) algorytmu HOG 
Omów wyzwania związane z implementacją sprzętową algorytmów (FPGA, potokowa) Lucasa-Kanade
Omów wyzwania związane z implementacją sprzętową algorytmów segmentacji obiektów pierwszoplanowych (wskaż do czego może ew. przydać się kompresja). 
Omów wyzwania związane z implementacją sprzętową algorytmów (FPGA, potokowa) stereowizja

 

 

 