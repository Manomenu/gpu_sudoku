# gpu_sudoku
Aplikacja do rozwiązywania na cpu i gpu sudoku. Celem jest przedstawienie wydajności fragmentów algorytmu na cpu i porównanie go z cpu.

# Pomysł
Jeśli mogę ustalić wartość w danej komórce to mogę pozbyć się tej wartości w pozostałych komórkach zajdujących się
w tym samym bloku 3x3, kolumnie i wierszu nakładając na nie ograniczenie uniemożliwające nadanie takiej samej wartości.

Zastosowanie ograniczenia może odkryć kolejne pole, w którym mogę ustalić wartość co pociąga za sobą możliwość 
nałożenia kolejnych ograniczeń. Powtarzanie tej procedury wraz z tworzeniem kopii planszy w niejednoznaczych stanach planszy
doprowadzi mnie do całkowitego rozwiąznia planszy.

# Implementacja algorytmu
## Założenia i logika
1) Plansza to tablica komórek długości 81.
2) Blok 81 wątków utożsamię z jedną planszą (tj. 1 wątek na jedną komórkę).
3) Mam strukturę trzech 9 elementowych tablic 16-bitowych intów, w bitach 1-9 trzymam informację o nałożonych ograniczeniach kolejno na wiersze, kolumny, bloki.
   Przykład:
     w komórce nie mogę wstawić wartości 1, 2, 3 , 9 => komórka ma wartość `0000 0001 1111 0000`
4) Jeśli flaga informująca o tym, że rozwiązanie już jest została ustawiona to bloki kończą działanie.
5) Każdy działający wątek w bloku aktualizuje ograniczenia w strukturze ograniczeń.
   - Jeśli po aktualizacji gdzieś jest zero opcji na wstawienie cyfry to ustaw flagę w bloku, że zakończył pracę i można skorzystać z zasobów, które zajmował.
   - Jeśli po aktualizacji w komórce jest tylko jedna opcja na cyfrę to wstaw tą wartość w aktualizowane miejsce w planszy i ustaw wątek jako niedziałający.
   - Jeśli po aktualizacji w komórce jest więcej niż jedna opcja na cyfrę to nic nie rób.
   - Jeśli po aktualizacji w żadnej komórce nie nastąpiła zmiana ograniczeń w strukturze to:
      - Wybierz komórkę z najmniejszą liczbą dostępnych wartości, jeśli jest takich wiele to wybierz tą z najmniejszym id przykładowo.
      - 1-szą wartość wstaw do tej komórki i kontunułuj dla tej wersji stołu obliczenia w tym bloku.
      - Dla pozostałych wartości stwórz kopie stołów z powstawianymi dostępnymi cyframi w wybraną niejednoznaczą komórkę w niepracujących blokach Ustaw te bloki jako pracujące.
      - Wywołaj pracę wszystkich bloków.
   - Jeśli po aktualizacji wszystkie wątki są niedziałające, a były wprowadzone zmiany w poprzedniej iteracji to znaczy, że mamy rozwiązanie.
     Skopiuj je do miejsca na rozwiązanie, oznacz flagą jakąś, że już jest rozwiązanie.
6) Wczytwane jest `BOARDS_NUM` plansz z pliku wejściowego, każda wczytana plansza jest rozwiązywana zaraz po tym, jak poprzednia zostanie rozwiązana.


