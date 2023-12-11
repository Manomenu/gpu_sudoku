//
////#include "cuda_runtime.h"
////#include "device_launch_parameters.h"
//
//#include <stdio.h>
//
//#include "sudokuCPU.h"
//
//#include <iostream>
//#include <iomanip>
//#include <chrono>
//
//const int N = 3;
//const int N2 = N*N;
//
//void print(char *solution);
//
//using namespace std;
//using namespace std::chrono;
//
//int main()
//{
//    char *data = (char*)"090700860031005020806000000007050006000307000500010700000000109020600350054008070";
//    char solution[81]; 
//    print(data);
//    int res = sudokuCPU(data,solution);
//    auto ts = high_resolution_clock::now();
//    res = sudokuCPU(data,solution);
//    auto te = high_resolution_clock::now();
//    cout << "Time:    " << setw(7) << 0.001*duration_cast<microseconds>(te-ts).count() << " nsec" << endl ;
//    print(solution);
//    return 0;
//}
//
//void print(char *solution)
//    {
//    for ( int i=0, ij=0 ; i<N2 ; ++i )
//        {
//        if ( i%N==0 ) printf("\n");
//        for ( int j=0 ; j<N2 ; ++j,++ij )
//            {
//            if ( j%N==0 ) printf(" |");
//            printf("  %c",solution[ij]);
//            }
//        printf(" |\n");
//        }
//    printf("\n");
//    }
