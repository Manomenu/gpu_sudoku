# gpu_sudoku
An application for solving sudoku using cpu and gpu. The central point is to compare solver implementations on datasets composed of different number of boards.

# Pomysł
Jeśli mogę ustalić wartość w danej komórce to mogę pozbyć się tej wartości w pozostałych komórkach zajdujących się
w tym samym bloku 3x3, kolumnie i wierszu nakładając na nie ograniczenie uniemożliwające nadanie takiej samej wartości.

Zastosowanie ograniczenia może odkryć kolejne pole, w którym mogę ustalić wartość co pociąga za sobą możliwość 
nałożenia kolejnych ograniczeń. Powtarzanie tej procedury może doprowadzić mnie do całkowitego rozwiąznia planszy lub ustalić 
część pól.

# Implementacja algorytmu
## Założenia i logika
1) Plansza to tablica komórek (ozn. X) długości 81.
2) Blok 81 wątków utożsamię z jedną planszą (tj. jeden wątek na jedną X)
3) X to int 16-bitowy, w bitach 1-9 trzymam informację o nałożonych ograniczeniach.
   Przykład:
     w X nie mogę wstawić wartości 1, 2, 3 , 9 => X ma wartość `0000 0001 1111 0000`
4) Każdy wątek czyta swoją X, jeśli tylko jeden bit ma wartość `true` to wątek skończy pracę, wiadomo konkretnie która   
   wartość może być przypisana.
   Wpp. wątek przegląda odpowiadający wiersz, kolumnę i blok `aktualizując ograniczenia w danej komórce`. Jeśli możliwości    
   uzupełnienia komórek zostaną zredukowane w jakiejkolwiek X to ponownie uruchamiamy niezakończone wątki dla    
   zaktualizowanej planszy.
   W przypadku, w którym żaden wątek nie zaktualizuje już planszy, a nie wszystkie wątki zostaną zakończone to plansza jest
   nierozwiązywalna lub nasza strategia nie pasuje do danej planszy. Wtedy uruchamiamy algorytm brutalny dla obecnego stanu
   planszy.

## Aktualizacja ograniczeń w komórce
Do aktualizacji ograniczeń
