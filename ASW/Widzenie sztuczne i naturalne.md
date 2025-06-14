## Wstęp
**Światło widzialne** - 380-780nm.

Jedno z podstawowych praw kolorymetrii mówi, że **trzy dowolne barwy tworzą układ barw podstawowych, jeśli zsumowanie świateł o dwóch barwach z rozpatrywanej trójki nie może doprowadzić do uzyskania mieszaniny odtwarzającej światło o trzeciej barwie.**

Podstawowym narzędziem do digitalizacji sygnału pochodzącego z kamery, TV lub magnetowidu (XD) jest karta wideo - **frame grabber**.

## RGB
* typowo, każda barwa reprezentowana przez jeden **bajt** informacji; trzy bajty na punkt
* model addytywny

## CMY, CMYK
* subtraktywne mieszanie barw
* pokazuje, co należy odjąć od koloru białego, by uzyskać docelowy kolor
* barwy CMY są dopełnieniem barw RGB do światła białego
* K = min(C, M, Y)

## HSI, HSV
* każdą barwę można jednoznacznie opisać za pomocą trzech składowych: Hue, Saturation, Value/Brightness/Luminosity
* H - barwy czyste; zamknięty ciąg barw, który tworzą barwy widmowe i nasycone purpury
* barwy dwóch świateł, które zsumowane dają światło o przyjętej umownie barwie białej, nazywamy **dopełniającymi**
* **barwy widmowe** możemy określić przez długość fali promieniowania monochromatycznego, **purpury** zaś poprzez długości fali odpowiadającej barwie widmowej dopełniającej
* **barwy czyste** to barwy nasycone; zmniejszenie nasycenia powoduje uzyskanie barwy nienasyconej; w wyniku całkowitego odbarwienia uzyskujemy barwę białą
![[Pasted image 20250524150414.png]]

## YUV
* **Zasada odpowiedniości** wymaga, by jeden z trzech sygnałów obrazu telewizji barwnej był identyczny z sygnałem telewizji monochromatycznej; sygnał ten zwany jest **luminancja**
* poza luminancją transmitowane były dwa różnicowe sygnały chrominancji
* przykładowy format YUV 4:2:2 - cztery bity informacji o jasności i po dwa bity informacji składowych różnicowych (ludzkie oko jest bardziej czułe na zmianę jasności)

## Zmysł Wzroku
![[Pasted image 20250524154106.png]]

## Widzenie automatyczne
Akwizycja --> przetwarzanie --> analiza --> rozpoznanie i semantyczna interpretacja

### Akwizycja
* kamery wizyjne, kamwidy
* CCD (cgarge-couple device) - układ ze sprzężeniem ładunkowym
	* układ wielu elementów światłoczułych, z których każdy rejestruje, a następnie pozwala odczytać sygnał elektryczny proporcjonalny do ilości padającego na niego promieniowania
	* Boyle i Smith, pierwszy sensor 19069, nagroda Nobla 2009
	* w cyfrowych aparatach fotograficznych najczęściej stosowane filtry barwne (filtry Bayera)
	* pierwsze matryce rozwijane w celu obserwacji kosmosu
	* zasada działania: kiedy powierzchnia matrycy CCD jest oświetlona, uwolnione zostają nośniki, które gromadzą się w kondensatorach (efekt fotoelektryczny wewnętrzny); nośniki te zostają przesunięte w miarowych impulsach elektrycznych oraz zostają przeliczone przez obwód, który wyłapuje nośniki z każdego elementu światłoczułego, przekazuje je do kondensatorów, mierzy, wzmacnia napięcie i opróżnia kondensatory; ilość zebranych nośników w pewnym przedziale czasu zależy od natężenia światła
	![[Pasted image 20250524162802.png]]
	
* CMOS (complementary metal-oxide semiconductor (1963/1993)
	* matryca składa się z milionów elementów o następującej budowie:
		* element światłoczuły działający na zasadzie fotodiody
		* wzmacniacz sygnału
		* przetwornik analogowo-cyfrowy
		* mikrosoczewka
		* filtr barwny (najczęściej filtr Bayera)
	* PPS (passive pixel sensor) - zawiera tylko element światłoczuły i dwa tranzystory sterujące; duży poziom szumów
	* APS (active pixel sensor) - dodano wzmacniacz, co zmniejszyło wpływ szumów
	* SoC (system on chip) - wbudowany przetwornik ADC
	* CoC (camera on chip) - często stosowane w telefonach, są wyposażone we własny procesor sygnałowy
	* zalety:
		* małe zakłócenia w przesyłaniu danych w związku z małą odległością fotodioda-ADC
		* niski koszt produkcji 
		* niski pobór mocy
		* szybki odczyt (bez potrzeby zaciemniania matrycy jak w CCD)
		* łatwe resetowanie (elektroniczna migawka)
		* możliwość odczytu wybranych pikseli (wykorzystywane przy ustawianiu ostrości)
	* wady:
		* mniejsza światłoczułość w porównaniu do CCD (część matrycy nie jest światłoczuła)
		* większy prąd ciemny
		* każdy piksel ma własny wzmacniacz, co powoduje różne wskazania poszczególnych pikseli przy tym samym oświetleniu
	* zakrzywione matryce CMOS:
		* można je łączyć z płaskimi obiektywami o większej aperturze, co pozwala na większy dopływ światła do matrycy
		* promienie świetlne padają prosto na fotodiody znajdujące się na brzegach 
		* siła, którą działano na czujnik w celu zakrzywienia, zmienia pasmo wzbronione poszczególnych diod, zmniejszając tym samym szum spowodowany prądem ciemnym
		* prostsza konstrukcja obiektywu

## Obraz jako funkcja
W technice wizyjnej trójwymiarowy obraz analogowy jest rzutowany na światłoczułą płaszczyznę przetwornika optoelektrycznego. Obraz ten jest reprezentowany przez dwuwymiarową funkcję intensywności oświetlenia *L(x, y)*.
Argumenty *x, y* opisują powierzchniowe współrzędne punktu obrazu, zaś wartość funkcji określona jest przez poziom jasności obrazu - luminanjcę.
Obrazy poddaje się procesowi **dyskretyzacji** (próbkowanie i kwantowanie).

