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

__device__ int fillBoard(board_t& board, int tid, constraints_t& constraints, int& isThreadProgressing, int& isSquareSet, int* takenValues);

__global__ void solveBoardDFSKernel(board_t* boards, uchar* solvedBoardStatus);

__global__ void solveBoardBFSKernel(board_t* boards, uchar* solvedBoardStatus, int* forkedBoardsCount, int* nextBoardId);

void solveBoard(board_t* dev_boards, uchar* dev_solvedBoardStatus, int* dev_forkedBoardsCount, int* dev_nextBoardId, board_t* solvedBoard, 
    std::chrono::milliseconds& copyMemoryDuration, std::chrono::milliseconds& dfsDuration, std::chrono::milliseconds& bfsDuration);

void solveBoards(board_t* inputBoards, board_t* solvedBoards);

#ifdef KOMPUTER_SZKOLA

void solveBoardsCPU(board_t* inputBoards, board_t* solvedBoards);

#endif

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

    #pragma region Reading_input_data

    auto start = std::chrono::high_resolution_clock::now();

    BoardsLoader::loadBoardsFromBinary(inputBoards);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "reading input data: " << std::setw(7) << 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " nsec" << std::endl;

    #pragma endregion

    std::cout << "\n" << BOARDS_NUM << " boards\n";

    #pragma region Whole_computing_gpu

    start = std::chrono::high_resolution_clock::now();
    
    solveBoards(inputBoards, solvedBoards);
    
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "gpu - whole computing time: " << std::setw(7) << 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " nsec" << std::endl;

    #pragma endregion

    #ifdef KOMPUTER_SZKOLA

    #pragma region Whole_computing_cpu

    start = std::chrono::high_resolution_clock::now();

    solveBoardsCPU(inputBoards, solvedBoards);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "cpu - whole computing time: " << std::setw(7) << 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " nsec" << std::endl;

    #pragma endregion

    #endif // KOMPUTER_SZKOLA

    #ifdef DEBUG 
    printBoard(solvedBoards[0]);
    #endif

    #pragma region Saving_results_to_file

    start = std::chrono::high_resolution_clock::now();

    BoardsLoader::saveBoardsToTxt(solvedBoards);

    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "\n" << "writing output data: " << std::setw(7) << 0.001 * std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << " nsec" << std::endl;

    #pragma endregion

    return 0;
}

#pragma region Function_implementations

#ifdef KOMPUTER_SZKOLA

void solveBoardsCPU(board_t* inputBoards, board_t* solvedBoards)
{
    for (int i = 0; i < BOARDS_NUM; ++i)
    {
        sudokuCPU((char*)inputBoards->cells, (char*)(solvedBoards + i)->cells);
    }
}

#endif

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
    __shared__ int takenValues[10];

    if (tid == 0)
    {
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

            isThreadInvalid = fillBoard(board, tid, constraints, isThreadProgressing, isSquareSet, takenValues);
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

                if (currentDecisionSquare == INT_MAX) return;

                if (tid == 0)
                {
                    decisionSquares[decisionsTaken] = currentDecisionSquare;
                    starterBoard[currentDecisionSquare]++;
                }
                __syncthreads();

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
    __shared__ int           takenValues[10];

    if (tid == 0)
    {
        memset(&constraints, 0, sizeof(constraints_t)); // idk why but it somehow stays with 1022 in each cell before previous solution!
        
        squareToFork = INT_MAX;
        canForkBeMade = true;
    }

    isBlockProgressing = 1;
    board[tid] = (boards + bid)->cells[tid];
    __syncthreads();

    while (isBlockProgressing)
    {
        isThreadProgressing = updateConstraints(board, tid, constraints);
        isBlockProgressing  = __syncthreads_or(isThreadProgressing);
    
        isThreadInvalid     = fillBoard(board, tid, constraints, isThreadProgressing, isSquareSet, takenValues);
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
        }

        mutex = 0;
    }
    __syncthreads();

    if (canForkBeMade == false)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        __syncthreads();

        return;
    }

    if (availableDigits & 1 << 1)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == 0) forkedBoard->cells[squareToFork] = 1;
        __syncthreads();
    }
    if (availableDigits & 1 << 2)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 2;
        __syncthreads();
    }
    if (availableDigits & 1 << 3)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 3;
        __syncthreads();
    }
    if (availableDigits & 1 << 4)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 4;
        __syncthreads();
    }
    if (availableDigits & 1 << 5)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 5;
        __syncthreads();
    }
    if (availableDigits & 1 << 6)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 6;
        __syncthreads();
    }
    if (availableDigits & 1 << 7)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 7;
        __syncthreads();
    }
    if (availableDigits & 1 << 8)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 8;
        __syncthreads();
    }
    if (availableDigits & 1 << 9)
    {
        if (tid == 0) boardId = atomicAdd(nextBoardId, 1);
        __syncthreads();

        forkedBoard = boards + boardId;

        forkedBoard->cells[tid] = board[tid];
        if (tid == squareToFork) forkedBoard->cells[tid] = 9;
        __syncthreads();
    }
}

void solveBoard(board_t* dev_boards, uchar* dev_solvedBoardStatus, int* dev_forkedBoardsCount, int* dev_nextBoardId, board_t* solvedBoard, 
    std::chrono::milliseconds& copyMemoryDuration, std::chrono::milliseconds& dfsDuration, std::chrono::milliseconds& bfsDuration)
{
    int     kernelIter = 0;
    int     forkedBoardsCount = 1;
    int     nextBoardId;
    uchar   solvedBoardStatus;
    std::chrono::steady_clock::time_point start, end;

    do
    {
        #pragma region BFS

        start = std::chrono::high_resolution_clock::now();

        solveBoardBFSKernel<<<forkedBoardsCount, THREADS_IN_BLOCK_NUM>>>(dev_boards, dev_solvedBoardStatus, dev_forkedBoardsCount, dev_nextBoardId);

        cudaStatus = cudaGetLastError();
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaDeviceSynchronize();
        validateCudaStatus(cudaStatus);

        end = std::chrono::high_resolution_clock::now();
        bfsDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        #pragma endregion

        start = std::chrono::high_resolution_clock::now();

        cudaStatus = cudaMemcpy(&solvedBoardStatus, dev_solvedBoardStatus, sizeof(uchar), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

        end = std::chrono::high_resolution_clock::now();
        copyMemoryDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        if (solvedBoardStatus == BoardStatus::Solved)
        {
            start = std::chrono::high_resolution_clock::now();

            cudaStatus = cudaMemcpy(solvedBoard, dev_boards + 2 * BOARDS_BATCH_SIZE, sizeof(board_t), cudaMemcpyDeviceToHost);
            validateCudaStatus(cudaStatus);

            end = std::chrono::high_resolution_clock::now();
            copyMemoryDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

            return;
        }

        start = std::chrono::high_resolution_clock::now();

        cudaStatus = cudaMemcpy(&forkedBoardsCount, dev_forkedBoardsCount, sizeof(int), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemcpy(&nextBoardId, dev_nextBoardId, sizeof(int), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);

        end = std::chrono::high_resolution_clock::now();
        copyMemoryDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        cudaStatus = cudaMemcpy(dev_boards, dev_boards + nextBoardId - forkedBoardsCount, sizeof(board_t) * forkedBoardsCount, cudaMemcpyDeviceToDevice);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_nextBoardId, forkedBoardsCount, 1);
        validateCudaStatus(cudaStatus);

    } while (forkedBoardsCount <= BOARDS_BATCH_SIZE - 8 && kernelIter++ <= MAX_KERNEL_ITERATIONS);  
    
    #pragma region DFS

    start = std::chrono::high_resolution_clock::now();

    solveBoardDFSKernel<<<forkedBoardsCount, THREADS_IN_BLOCK_NUM>>>(dev_boards, dev_solvedBoardStatus);

    cudaStatus = cudaGetLastError();
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaDeviceSynchronize();
    validateCudaStatus(cudaStatus);

    end = std::chrono::high_resolution_clock::now();
    dfsDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    #pragma endregion

    start = std::chrono::high_resolution_clock::now();

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

    end = std::chrono::high_resolution_clock::now();
    copyMemoryDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    return;
}

void solveBoards(board_t* inputBoards, board_t* solvedBoards)
{
    board_t*    dev_boards; // goal is to have BOARDS_BATCH_SIZE boards one after another there and place for solvedBoard at the end
    uchar*      dev_solvedBoardStatus;
    int*        dev_forkedBoardsCount;
    int*        dev_nextBoardId;
    std::chrono::milliseconds copyMemoryDuration(0), bfsDuration(0), dfsDuration(0);
    std::chrono::steady_clock::time_point start, end;
    
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

        start = std::chrono::high_resolution_clock::now();

        cudaStatus = cudaMemcpy(dev_boards, inputBoards + i, sizeof(board_t), cudaMemcpyHostToDevice);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_solvedBoardStatus, BoardStatus::Available, sizeof(uchar));
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_forkedBoardsCount, 1, 1);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_nextBoardId, 1, 1);
        validateCudaStatus(cudaStatus);

        end = std::chrono::high_resolution_clock::now();
        copyMemoryDuration += std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        #pragma endregion

        solveBoard(dev_boards, dev_solvedBoardStatus, dev_forkedBoardsCount, dev_nextBoardId, solvedBoards + i, copyMemoryDuration, dfsDuration, bfsDuration);
    }

    std::cout << "gpu - copying data to CPU: " << std::setw(7) << 0.001 * copyMemoryDuration.count() << " nsec" << std::endl;
    std::cout << "gpu - bfs part: " << std::setw(7) << 0.001 * bfsDuration.count() << " nsec" << std::endl;
    std::cout << "gpu - dfs part: " << std::setw(7) << 0.001 * dfsDuration.count() << " nsec" << std::endl;

    cudaFree(dev_boards);
    cudaFree(dev_solvedBoardStatus);
    cudaFree(dev_forkedBoardsCount);
    cudaFree(dev_nextBoardId);
}

__device__ int updateConstraints(board_t& board, int tid, constraints_t& constraints)
{
    int isBlockProgressing = 0;
    uint old_value;

    old_value = constraints.rows[tid / 9];
    atomicOr(&(constraints.rows[tid / 9]), (1u << board[tid] & ~1u));
    __syncthreads();
    if (old_value < constraints.rows[tid / 9])
    {
        isBlockProgressing = 1;
    }

    old_value = constraints.columns[tid % 9];
    atomicOr(&(constraints.columns[tid % 9]), (1u << board[tid] & ~1u));
    __syncthreads();
    if (old_value < constraints.columns[tid % 9])
    {
        isBlockProgressing = 1;
    }

    old_value = constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3];
    atomicOr(&(constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]), (1u << board[tid] & ~1u));
    __syncthreads();
    if (old_value < constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3])
    {
        isBlockProgressing = 1;
    }

    return isBlockProgressing;
}

__device__ int fillBoard(board_t& board, int tid, constraints_t& constraints, int& isThreadProgressing, int& isSquareSet, int* takenValues)
{
    int isThreadInvalid = 0;
    uint availableDigits = 0;
    uchar digit;

    isSquareSet = 0;

    availableDigits =
        0x000003FE &
        (~(constraints.rows[tid / 9]
            | constraints.columns[tid % 9]
            | constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]));
    digit = onePosToDigit(availableDigits);

    if (tid < 10) takenValues[tid] = INT_MAX;
    __syncthreads();

    if (board[tid] > 0)
    {
        isSquareSet = 1;
        digit = 12;
    }

    if (digit == 0) isThreadInvalid = 1;

    if (digit > 0 && digit < 10) atomicMin(&takenValues[digit], tid);
    __syncthreads();

    if (digit > 0 && digit < 10 && takenValues[digit] == tid)
    {
        board[tid] = digit;

        atomicOr(&(constraints.rows[tid / 9]), (1u << board[tid] & ~1u));
        atomicOr(&(constraints.columns[tid % 9]), (1u << board[tid] & ~1u));
        atomicOr(&(constraints.blocks[tid / 9 / 3 * 3 + tid % 9 / 3]), (1u << board[tid] & ~1u));

        isThreadProgressing = 1;
        isSquareSet = 1;
    }
    __syncthreads();

    return isThreadInvalid;
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
