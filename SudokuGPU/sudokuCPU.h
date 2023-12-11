#pragma once

extern "C" __declspec(dllexport) int sudokuCPU(char *game, char *solution);
// Zwracany wynik
//   1 : znaleziono 1 rozwiazanie
//   0 : nie znaleziono rozwizan
//  -1 : nieprawidlowy znak na wejsciu
//  -2 : konflikt na planszy poczatkowej
//  -3 : znaleziono zbyt duzo rozwiazan
