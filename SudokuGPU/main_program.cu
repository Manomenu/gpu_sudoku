#include "device_launch_parameters.h"
#include "cuda_utils.cuh"
#include "config.cuh"
#include "boards_loader.cuh"

#pragma region Global_variables

cudaError_t cudaStatus;

__device__ int mutex = 0;


#pragma endregion

#pragma region Function_definitions

#ifdef DEBUG
void printBoard(board_t& board)
{
    for (int y = 0; y < BOARD_SIZE; ++y)
    {
        for (int x = 0; x < BOARD_SIZE; ++x)
        {
            std::cout << char(board.cells[x + y * BOARD_SIZE] + '0') << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}
#endif

__device__ int numberOfDigits(uint a);

__device__ uchar onePosToDigit(uint a);

__device__ bool hasOneTrueBit(uint a);

__device__ bool differsAtOneBit(uint a, uint b);

__device__ int forkBoard(board_t* boards, board_t& board, int tid, constraints_t& constraints, int* forkedBoardsCount, int* nextBoardId, int& squareToFork, bool& canForkBeMade, int& boardId);

__device__ int updateConstraints(board_t& board, int tid, constraints_t& constraints);

__device__ int fillBoard(board_t& board, int tid, constraints_t& constraints, int& isThreadProgressing, int& isSquareSet, int& mutexBoard);

__global__ void solveBoardDFSKernel(board_t* boards, uchar* solvedBoardStatus);

__global__ void solveBoardBFSKernel(board_t* boards, uchar* solvedBoardStatus, int* forkedBoardsCount, int* nextBoardId);

void solveBoard(board_t* dev_boards, uchar* dev_solvedBoardStatus, int* dev_forkedBoardsCount, int* dev_nextBoardId, board_t* solvedBoard);

void solveBoards(board_t* inputBoards, board_t* solvedBoards);

#pragma endregion

int main()
{
    board_t inputBoards[BOARDS_NUM];
    board_t solvedBoards[BOARDS_NUM];

    #pragma region Cuda_setup

    cudaStatus = cudaDeviceReset();
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaSetDevice(0);
    validateCudaStatus(cudaStatus);

    #pragma endregion

    BoardsLoader::loadBoardsFromBinary(inputBoards);

    solveBoards(inputBoards, solvedBoards);

    #ifdef DEBUG 
        printBoard(solvedBoards[0]);
    #endif

    BoardsLoader::saveBoardsToTxt(solvedBoards);

    return 0;
}

#pragma region Function_implementations

__global__ void solveBoardDFSKernel(board_t* boards, uchar* solvedBoardStatus)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int isThreadProgressing = 1;
    int isBlockProgressing = 1;
    int isThreadInvalid = 0;
    int isSquareSet = 0;
    int isBlockInvalid = 0;
    int squaresSet = 0;
    board_t& solutionBoard = *(boards + 2 * BOARDS_BATCH_SIZE);

    __shared__ constraints_t constraints;
    __shared__ board_t board;
    __shared__ board_t starterBoard;
    __shared__ int decisionSquares[81];
    __shared__ int decisionsTaken;
    __shared__ int backtrackingDone;
    __shared__ int decisionDone;
    __shared__ int currentDecisionSquare;
    __shared__ int mutexBoard;

    if (tid == 0)
    {
        mutexBoard = INT_MAX;
        memset(&constraints, 0, sizeof(constraints_t)); // idk why but it somehow stays with 1022 in each cell before previous solution!
        decisionsTaken = 0;
    }

    starterBoard[tid] = (boards + bid)->cells[tid];
    decisionSquares[tid] = -1;
    __syncthreads();

    while (*solvedBoardStatus != BoardStatus::Solved) // try to solve board
    {
        board[tid] = starterBoard[tid];
        __syncthreads();

        if (tid == 0) memset(constraints.rows, 0, sizeof(constraints.rows));
        if (tid == 1) memset(constraints.columns, 0, sizeof(constraints.columns));
        if (tid == 2) memset(constraints.blocks, 0, sizeof(constraints.blocks));
        __syncthreads();

        // try to solve current starter board
        isBlockProgressing = 1;
        while (isBlockProgressing)
        {
            isSquareSet = 0;
            isBlockProgressing = 0; 
            squaresSet = 0;
            isBlockInvalid = 0;

            isThreadProgressing = updateConstraints(board, tid, constraints);
            isBlockProgressing = __syncthreads_or(isThreadProgressing);

            isThreadInvalid = fillBoard(board, tid, constraints, isThreadProgressing, isSquareSet, mutexBoard);
            isBlockInvalid = __syncthreads_or(isThreadInvalid);
            isBlockProgressing = __syncthreads_or(isThreadProgressing) | isBlockProgressing;
            squaresSet = __syncthreads_count(isSquareSet);

            if (squaresSet == BOARD_SIZE * BOARD_SIZE) // jesli mamy rozwiazanie
            {
                solutionBoard[tid] = board[tid];
                if (tid == 0) *solvedBoardStatus = BoardStatus::Solved;
                __syncthreads();

                return;
            }

            if (isBlockInvalid) // jesli zly wybor zostal dokonany wczesniej                 // popraw wybor i zacznij od nowa
            {
                if (decisionsTaken == 0) return;

                if (tid == 0) backtrackingDone = 0;
                __syncthreads();

                while (backtrackingDone == 0)
                {
                    if (decisionsTaken == 1 && starterBoard[decisionSquares[decisionsTaken]] >= 9) return; // nie udalo sie cofnac

                    if (starterBoard[decisionSquares[decisionsTaken]] < 9) // moge sprobowac kolejna wartosc w ostatniej decyzji
                    {
                        if (tid == 0) starterBoard[decisionSquares[decisionsTaken]]++;
                        __syncthreads();

                        board[tid] = starterBoard[tid];
                        __syncthreads();
                        board[decisionSquares[decisionsTaken]] = 0;
                        __syncthreads();

                        if (tid == 0) memset(constraints.rows, 0, sizeof(constraints.rows));
                        if (tid == 1) memset(constraints.columns, 0, sizeof(constraints.columns));
                        if (tid == 2) memset(constraints.blocks, 0, sizeof(constraints.blocks));
                        __syncthreads();

                        updateConstraints(board, tid, constraints);
                        __syncthreads();

                        int takenDigits;
                        

                        if (tid == 0)
                        {
                            takenDigits = (constraints.rows[decisionSquares[decisionsTaken] / 9]
                                | constraints.columns[decisionSquares[decisionsTaken] % 9]
                                | constraints.blocks[decisionSquares[decisionsTaken] / 9 / 3 * 3 + decisionSquares[decisionsTaken] % 9 / 3]);

                            if ((takenDigits | 1 << starterBoard[decisionSquares[decisionsTaken]]) > takenDigits)
                            {
                                backtrackingDone = 1;
                            }
                        }
                        __syncthreads();
                    }

                    if (backtrackingDone == 0 && starterBoard[decisionSquares[decisionsTaken]] >= 9)
                    {
                        if (tid == 0)
                        {
                            starterBoard[decisionSquares[decisionsTaken]] = 0;
                            decisionSquares[decisionsTaken] = -1;
                            decisionsTaken--;
                        }
                        __syncthreads();

                        if (decisionsTaken == 0) return;
                    }
                }
                break;
            }

            if (!isBlockProgressing) // jesli trzeba dokonac kolejnego wyboru
            {
                if (tid == 0)
                {
                    decisionsTaken++;
                    decisionDone = 0;
                    currentDecisionSquare = INT_MAX;
                }
                __syncthreads();

                if (board[tid] == 0) atomicMin(&currentDecisionSquare, tid);
                __syncthreads();

                if (currentDecisionSquare == INT_MAX) return; // should never happen!

                if (tid == 0)
                {
                    decisionSquares[decisionsTaken] = currentDecisionSquare;
                    starterBoard[currentDecisionSquare]++;
                }
                __syncthreads();

                // ustal dostepne wartosci dla nowego pola dycyzji
                board[tid] = starterBoard[tid];
                if (tid == currentDecisionSquare) board[tid] = 0;
                __syncthreads();

                if (tid == 0) memset(constraints.rows, 0, sizeof(constraints.rows));
                if (tid == 1) memset(constraints.columns, 0, sizeof(constraints.columns));
                if (tid == 2) memset(constraints.blocks, 0, sizeof(constraints.blocks));
                __syncthreads();

                updateConstraints(board, tid, constraints);
                __syncthreads();

                uint availableDigits;

                if (tid == 0)
                {
                   availableDigits = ~(constraints.rows[currentDecisionSquare / 9]
                        | constraints.columns[currentDecisionSquare % 9]
                        | constraints.blocks[currentDecisionSquare / 9 / 3 * 3 + currentDecisionSquare % 9 / 3]) & 0x000003FE;
                }
                __syncthreads();

                while (decisionDone == 0) // wybierz poprawna wartosc dla decyzji (najmniejsza dostepna)
                {
                    if (tid == 0)
                    {
                        if ((availableDigits & (1u << starterBoard[currentDecisionSquare])) > 0)
                        {
                            decisionDone = 1;
                        }
                        else
                        {
                            starterBoard[currentDecisionSquare]++; 
                        }
                    }
                    __syncthreads();
                }
                break;
            }
        }
    }
}

//__global__ void solveBoardDFSKernelx(board_t* boards, uchar* solvedBoardStatus)
//{
//    int      tid = threadIdx.x;
//    int      bid = blockIdx.x;
//    int      isThreadProgressing;
//    int      isThreadInvalid;
//    int      isSquareSet;
//    board_t& solutionBoard = *(boards + 2 * BOARDS_BATCH_SIZE);
//
//    int           isBlockProgressing;
//    __shared__ int           isBlockInvalid;
//    __shared__ int           squaresSet;
//    __shared__ int           new_deciosion_square;
//    __shared__ int           boardId;
//    __shared__ bool          canForkBeMade;
//    __shared__ constraints_t constraints;
//    __shared__ board_t       board;
//
//    __shared__ board_t        starterBoard;
//    __shared__ int            decision_depth;
//    //__shared__ short decisions_value[81];
//    __shared__ short decisions_squares[81];
//    __shared__ int decision_estabilished;
//
//    if (tid == 0)
//    {
//        memset(&constraints, 0, sizeof(constraints_t)); // idk why but it somehow stays with 1022 in each cell before previous solution!
//        isBlockProgressing = 1;
//        new_deciosion_square = INT_MAX;
//        canForkBeMade = true;
//        decision_depth = -1;
//        decision_estabilished = 0;
//    }
//
//    board[tid] = (boards + bid)->cells[tid];
//    starterBoard[tid] = board[tid];
//    //decisions_value[tid] = 0;
//    decisions_squares[tid] = -1;
//    __syncthreads();
//
//    while (true)
//    {
//        while (isBlockProgressing)
//        {
//            if (tid == 0) isBlockProgressing = 0;
//            __syncthreads();
//
//            #pragma region standard
//
//            isThreadProgressing = updateConstraints(board, tid, constraints);
//            isBlockProgressing = __syncthreads_or(isThreadProgressing);
//
//            isThreadInvalid = fillBoard(board, tid, constraints, isThreadProgressing, isSquareSet);
//
//
//            isBlockInvalid = __syncthreads_or(isThreadInvalid);
//            isBlockProgressing = __syncthreads_or(isThreadProgressing);
//            squaresSet = __syncthreads_count(isSquareSet);
//
//            if (squaresSet == BOARD_SIZE * BOARD_SIZE)
//            {
//                solutionBoard[tid] = board[tid];
//                if (tid == 0) *solvedBoardStatus = BoardStatus::Solved;
//                __syncthreads();
//
//                return;
//            }
//
//#pragma endregion
//
//            if (isBlockInvalid)
//            {
//                while (decision_estabilished == 0)
//                {
//                    // no solution
//                    if (decision_depth == 0 && starterBoard[decisions_squares[decision_depth]] == 9 || decision_depth == -1) return;
//
//                    // can change last decision
//                    else if (starterBoard[decisions_squares[decision_depth]] < 9) // might update last decision
//                    {
//                        int next_decision = ++starterBoard[decisions_squares[decision_depth]];
//                        __syncthreads();
//
//                        if (tid == 0)
//                        {
//                            starterBoard[decisions_squares[decision_depth]] = 0;
//                        }
//                        __syncthreads();
//
//                        #pragma region clear_constraints
//                        if (tid < 9)
//                        {
//                            constraints.rows[tid] = 0;
//                        }
//                        else if (tid < 18)
//                        {
//                            constraints.blocks[tid - 9] = 0;
//                        }
//                        else if (tid < 27)
//                        {
//                            constraints.columns[tid - 18] = 0;
//                        }
//                        __syncthreads();
//#pragma endregion
//
//                        updateConstraints(starterBoard, tid, constraints);
//                        __syncthreads();
//
//                        starterBoard[decisions_squares[decision_depth]] = next_decision;
//                        __syncthreads();
//
//                        int takenDigits = constraints.rows[decisions_squares[decision_depth] / 9]
//                            | constraints.columns[decisions_squares[decision_depth] % 9]
//                            | constraints.blocks[decisions_squares[decision_depth] / 9 / 3 * 3 + decisions_squares[decision_depth] % 9 / 3];
//                        __syncthreads();
//
//                        if ((takenDigits & (1 << next_decision)) > takenDigits)
//                        {
//                            decision_estabilished = 1;
//                            board[tid] = starterBoard[tid];
//                        }
//                        __syncthreads();
//                    }
//
//                    // back off decision
//                    else
//                    {
//                        if (tid == 0)
//                        {
//                            starterBoard[decisions_squares[decision_depth]] = 0;
//                            decisions_squares[decision_depth] = -1;
//                            decision_depth--;
//                        }
//                        __syncthreads();
//                    }
//
//                }
//                
//                #pragma region clear_constr
//                if (tid < 9)
//                {
//                    constraints.rows[tid] = 0;
//                }
//                else if (tid < 18)
//                {
//                    constraints.blocks[tid - 9] = 0;
//                }
//                else if (tid < 27)
//                {
//                    constraints.columns[tid - 18] = 0;
//                }
//#pragma endregion
//                if (tid == 0)
//                {
//                    decision_estabilished = 0;
//                    isBlockInvalid = 0;
//                    
//                }
//                isBlockProgressing = 1;
//                __syncthreads();
//            }
//        }
//
//        // make new decision
//
//        isBlockProgressing = 1;
//
//        if (board[tid] == 0) atomicMin(&new_deciosion_square, tid);
//        __syncthreads();
//
//        if (tid == 0)
//        {
//            decisions_squares[++decision_depth] = new_deciosion_square;
//        }
//        __syncthreads();
//
//        // find fist valid decision
//        int availableDigits = ~(constraints.rows[new_deciosion_square / 9]
//            | constraints.columns[new_deciosion_square % 9]
//            | constraints.blocks[new_deciosion_square / 9 / 3 * 3 + new_deciosion_square % 9 / 3]) & 0x000003FE;
//        __syncthreads();
//
//        if (availableDigits & 1 << 1)
//        {
//            if (tid == new_deciosion_square) 
//            {
//                starterBoard[tid] = 1;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 2)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 2;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 3)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 3;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 4)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 4;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 5)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 5;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 6)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 6;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 7)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 7;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 8)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 8;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//        else if (availableDigits & 1 << 9)
//        {
//            if (tid == new_deciosion_square)
//            {
//                starterBoard[tid] = 9;
//                board[tid] = starterBoard[tid];
//            }
//            __syncthreads();
//        }
//
//        if (tid == 0)
//        {
//            isBlockInvalid = 0;
//            
//        }
//        isBlockProgressing = 1;
//        __syncthreads();
//    }
//}

__global__ void solveBoardBFSKernel(board_t* boards, uchar* solvedBoardStatus, int* forkedBoardsCount, int* nextBoardId)
{
    int      tid = threadIdx.x;
    int      bid = blockIdx.x;
    int      isThreadProgressing;
    int      isThreadInvalid;
    int      isSquareSet;
    int      isBlockProgressing;
    board_t& solutionBoard = *(boards + 2 * BOARDS_BATCH_SIZE);

    
    __shared__ int           isBlockInvalid;
    __shared__ int           squaresSet;
    __shared__ int           squareToFork;
    __shared__ int           boardId;
    __shared__ bool          canForkBeMade;
    __shared__ constraints_t constraints;
    __shared__ board_t       board;
    __shared__ int           mutexBoard;

    if (tid == 0)
    {
        memset(&constraints, 0, sizeof(constraints_t)); // idk why but it somehow stays with 1022 in each cell before previous solution!
        
        mutexBoard = INT_MAX;
        squareToFork = INT_MAX;
        canForkBeMade = true;
    }

    isBlockProgressing = 1;
    board[tid] = (boards + bid)->cells[tid];
    __syncthreads();

    while (isBlockProgressing)
    {
        if (tid == 0) isBlockProgressing = 0;
        __syncthreads();

        isThreadProgressing = updateConstraints(board, tid, constraints);
        isBlockProgressing  = __syncthreads_or(isThreadProgressing);
    
        isThreadInvalid     = fillBoard(board, tid, constraints, isThreadProgressing, isSquareSet, mutexBoard);
        isBlockInvalid      = __syncthreads_or(isThreadInvalid);
        isBlockProgressing  = __syncthreads_or(isThreadProgressing);
        squaresSet          = __syncthreads_count(isSquareSet);

        if (squaresSet == BOARD_SIZE * BOARD_SIZE)
        {
            solutionBoard[tid] = board[tid];
            if (tid == 0) *solvedBoardStatus = BoardStatus::Solved;
            __syncthreads();

            return;
        }

        if (isBlockInvalid)
        {
            if (tid == 0)
            {
                bool locked = true;
                while (locked) {
                    while (atomicCAS(&mutex, 0, 1) != 0);
                    locked = false;
                }

                atomicSub(forkedBoardsCount, 1);

                mutex = 0;
            }
            __syncthreads();

            return;
        }
    }

    forkBoard(boards, board, tid, constraints, forkedBoardsCount, nextBoardId, squareToFork, canForkBeMade, boardId);
    __syncthreads();

    return;
}

__device__ int forkBoard(board_t* boards, board_t& board, int tid, constraints_t& constraints, int* forkedBoardsCount, int* nextBoardId, int& squareToFork, bool& canForkBeMade, int& boardId)
{
    uint availableDigits;
    uint digit;
    int  digitsCount;
    bool locked = true;
    board_t* forkedBoard;

    if (board[tid] == 0) atomicMin(&squareToFork, tid);
    __syncthreads();

    availableDigits = ~(constraints.rows[squareToFork / 9]
        | constraints.columns[squareToFork % 9]
        | constraints.blocks[squareToFork / 9 / 3 * 3 + squareToFork % 9 / 3]) & 0x000003FE;

    if (tid == 0)
    {
        digitsCount = numberOfDigits(availableDigits);
        
        while (locked) {
            while (atomicCAS(&mutex, 0, 1) != 0);
            locked = false;
        }

        // Critical section
        if (*forkedBoardsCount + digitsCount - 1 > BOARDS_BATCH_SIZE) 
        {
            canForkBeMade = false; 
        }
        else
        {
            *forkedBoardsCount += digitsCount - 1;
            canForkBeMade = true;

            //debug
            if (*forkedBoardsCount == 0)
            {
                int errxd = 3;
                errxd = 4;
            }
        }

        mutex = 0;
    }
    int ku = canForkBeMade ? 1:0;
    __syncthreads();

    if (canForkBeMade == false)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        __syncthreads();

        return;
    }

    if (availableDigits & 1 << 1)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == 0) forkedBoard->cells[squareToFork] = 1;
        __syncthreads();
    }
    if (availableDigits & 1 << 2)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 2;
        __syncthreads();
    }
    if (availableDigits & 1 << 3)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 3;
        __syncthreads();
    }
    if (availableDigits & 1 << 4)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 4;
        __syncthreads();
    }
    if (availableDigits & 1 << 5)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 5;
        __syncthreads();
    }
    if (availableDigits & 1 << 6)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 6;
        __syncthreads();
    }
    if (availableDigits & 1 << 7)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 7;
        __syncthreads();
    }
    if (availableDigits & 1 << 8)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        if (boardId >= 400)
        {
            int nie = 3;
            nie = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 8;
        __syncthreads();
    }
    if (availableDigits & 1 << 9)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        //debug
        if (boardId >= 400)
        {
            int a = 3;
            a = 4;
        }

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 9;
        __syncthreads();
    }
}

void solveBoard(board_t* dev_boards, uchar* dev_solvedBoardStatus, int* dev_forkedBoardsCount, int* dev_nextBoardId, board_t* solvedBoard)
{
    int     kernelIter = 0;
    int     forkedBoardsCount = 1;
    int     nextBoardId;
    uchar   solvedBoardStatus;

    board_t xd[BOARDS_BATCH_SIZE * 2 + 1];

    // forking & solving board
    do
    {
        if (forkedBoardsCount == 0)
        {
            int aaa = 3;
            aaa = 4;
        }
        solveBoardBFSKernel<<<forkedBoardsCount, THREADS_IN_BLOCK_NUM>>>(dev_boards, dev_solvedBoardStatus, dev_forkedBoardsCount, dev_nextBoardId);

        cudaStatus = cudaGetLastError();
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaDeviceSynchronize();
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemcpy(&solvedBoardStatus, dev_solvedBoardStatus, sizeof(uchar), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

        if (solvedBoardStatus == BoardStatus::Solved)
        {
            cudaStatus = cudaMemcpy(solvedBoard, dev_boards + 2 * BOARDS_BATCH_SIZE, sizeof(board_t), cudaMemcpyDeviceToHost);
            validateCudaStatus(cudaStatus);

            return;
        }

        cudaStatus = cudaMemcpy(&forkedBoardsCount, dev_forkedBoardsCount, sizeof(int), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemcpy(&nextBoardId, dev_nextBoardId, sizeof(int), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

        // might be problematic, consider changing it to standard memcpy
        cudaStatus = cudaMemcpy(dev_boards, dev_boards + nextBoardId - forkedBoardsCount, sizeof(board_t) * forkedBoardsCount, cudaMemcpyDeviceToDevice);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_nextBoardId, forkedBoardsCount, 1);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemcpy(xd, dev_boards, sizeof(board_t) * (2 * BOARDS_BATCH_SIZE + 1), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

    } while (forkedBoardsCount <= BOARDS_BATCH_SIZE - 8 && kernelIter++ <= MAX_KERNEL_ITERATIONS);    // I will still have place for at least one full fork 
                                                                                                        // from one square of required within my forked boards batch
    // solving board
    solveBoardDFSKernel<<<forkedBoardsCount, THREADS_IN_BLOCK_NUM>>>(dev_boards, dev_solvedBoardStatus);

    cudaStatus = cudaGetLastError();
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaDeviceSynchronize();
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaMemcpy(&solvedBoardStatus, dev_solvedBoardStatus, sizeof(uchar), cudaMemcpyDeviceToHost);
    validateCudaStatus(cudaStatus);

    if (solvedBoardStatus == BoardStatus::Solved)
    {
        cudaStatus = cudaMemcpy(solvedBoard, dev_boards + 2 * BOARDS_BATCH_SIZE, sizeof(board_t), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);
    }
    else
    {
        memset(solvedBoard, 0, sizeof(board_t));
    }

    return;
}

void solveBoards(board_t* inputBoards, board_t* solvedBoards)
{
    board_t*    dev_boards; // goal is to have BOARDS_BATCH_SIZE boards one after another there and place for solvedBoard at the end
    uchar*      dev_solvedBoardStatus;
    int*        dev_forkedBoardsCount;
    int*        dev_nextBoardId;
    
    #pragma region Device_memory_allocation

    cudaStatus = cudaMalloc((void**)&dev_boards, sizeof(board_t) * (2 * BOARDS_BATCH_SIZE + 1));
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_solvedBoardStatus, sizeof(board_t));
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_forkedBoardsCount, sizeof(int));
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_nextBoardId, sizeof(int));
    validateCudaStatus(cudaStatus);

    #pragma endregion

    for (int i = 0; i < BOARDS_NUM; ++i)
    {
        #pragma region Data_for_solving_sudoku_preparation

        cudaStatus = cudaMemcpy(dev_boards, inputBoards + i, sizeof(board_t), cudaMemcpyHostToDevice);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_solvedBoardStatus, BoardStatus::Available, sizeof(uchar));
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_forkedBoardsCount, 1, 1);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_nextBoardId, 1, 1);
        validateCudaStatus(cudaStatus);

        #pragma endregion

        solveBoard(dev_boards, dev_solvedBoardStatus, dev_forkedBoardsCount, dev_nextBoardId, solvedBoards + i);
    }

    cudaFree(dev_boards);
    cudaFree(dev_solvedBoardStatus);
    cudaFree(dev_forkedBoardsCount);
    cudaFree(dev_nextBoardId);
}

__device__ int updateConstraints(board_t& board, int tid, constraints_t& constraints)
{
    int isBlockProgressing = 0;
    uint old_value;
    int  stillProgressing = 1;
    
    while (stillProgressing)
    {
        stillProgressing = 0;


        old_value = constraints.rows[tid / 9];
        atomicOr(&(constraints.rows[tid / 9]), (1u << board[tid] & ~1u));
        __syncthreads();
        if (old_value < constraints.rows[tid / 9])
        {
            isBlockProgressing = 1;
            stillProgressing = 1;
        }

        old_value = constraints.columns[tid % 9];
        atomicOr(&(constraints.columns[tid % 9]), (1u << board[tid] & ~1u));
        __syncthreads();
        if (old_value < constraints.columns[tid % 9])
        {
            isBlockProgressing = 1;
            stillProgressing = 1;
        }

        old_value = constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3];
        atomicOr(&(constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]), (1u << board[tid] & ~1u));
        __syncthreads();
        if (old_value < constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3])
        {
            isBlockProgressing = 1;
            stillProgressing = 1;
        }

        stillProgressing = __syncthreads_or(stillProgressing);
    }

    return isBlockProgressing;
}

__device__ int fillBoard(board_t& board, int tid, constraints_t& constraints, int& isThreadProgressing, int& isSquareSet, int& square)
{
    int isThreadInvalid = 0;
    uint    availableDigits = 0;
    uchar digit = 12;

    square = 82;
    isThreadProgressing = 0;
    isSquareSet = 0;



    

    for (int i = 0; i < 81; i++)
    {
        if (tid == i)
        {
            if (board[tid] == 0)
            {
                availableDigits =
                    0x000003FE &
                    (~(constraints.rows[tid / 9]
                        | constraints.columns[tid % 9]
                        | constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]));
                digit = onePosToDigit(availableDigits);

                if (digit > 0 && digit < 10)
                {
                    board[tid] = digit;

                    atomicOr(&(constraints.rows[tid / 9]), (1u << board[tid] & ~1u));
                    atomicOr(&(constraints.columns[tid % 9]), (1u << board[tid] & ~1u));
                    atomicOr(&(constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]), (1u << board[tid] & ~1u));

                    isThreadProgressing = 1;
                    isSquareSet = 1;
                }

                if (availableDigits == 0)
                {
                    isThreadInvalid = 1;
                }
            }
            else
            {
                isSquareSet = 1;
            }
        }

        __syncthreads();
    }

    return isThreadInvalid;

    /*isSquareSet = 0;
    int     isBoardInvalid = 0;
    uint    availableDigits;
    uchar   digit;
    int locked = true;

    isThreadProgressing = 0;
    isSquareSet = 0;


    availableDigits =
        0x000003FE &
        (~(constraints.rows[tid / 9]
            | constraints.columns[tid % 9]
            | constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]));

    digit = onePosToDigit(availableDigits);

    
    int min;
    
    if (digit > 0 && digit < 10 && board[tid] == 0)
    {
        min = atomicMin(&mutexBoard, tid);
    }
    __syncthreads();

    if (digit == 0)
    {
        isBoardInvalid = 1;
        mutexBoard = 0;

        return isBoardInvalid;
    }

    if (board[tid] > 0)
    {
        isSquareSet = 1;

        return isBoardInvalid;
    }

    if (min == INT_MAX) return 0;

    

    
     if (digit < 10 && tid == min)
    {

        atomicOr(&(constraints.rows[tid / 9]), (1u << board[tid] & ~1u));
        atomicOr(&(constraints.columns[tid % 9]), (1u << board[tid] & ~1u));
        atomicOr(&(constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]), (1u << board[tid] & ~1u));

        isThreadProgressing = 1;
        isSquareSet = 1;
    }
    mutexBoard = 0;


    return isBoardInvalid;*/
}

__device__ uchar onePosToDigit(uint a) // O(1)
{
    switch (a)
    {
    case 0: return 0;
    case 1 << 0: return 10; // should not happen, throw cuda error if occured
    case 1 << 1: return 1;
    case 1 << 2: return 2;
    case 1 << 3: return 3;
    case 1 << 4: return 4;
    case 1 << 5: return 5;
    case 1 << 6: return 6;
    case 1 << 7: return 7;
    case 1 << 8: return 8;
    case 1 << 9: return 9;
    default: return 11;
    }
}

__device__ bool hasOneTrueBit(uint a)
{
    return a && !(a & (a - 1));
}

__device__ bool differsAtOneBit(uint a, uint b)
{
    return hasOneTrueBit(a ^ b);
}

__device__ int numberOfDigits(uint n)
{
    int count = 0;
    while (n != 0) {
        n = n & (n - 1);
        count++;
    }

    return count;
}

#pragma endregion
