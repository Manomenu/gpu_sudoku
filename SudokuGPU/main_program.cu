#include "device_launch_parameters.h"
#include "cuda_utils.cuh"
#include "config.cuh"
#include "boards_loader.cuh"

#pragma region Global_variables

cudaError_t cudaStatus;

#pragma endregion

#pragma region Function_definitions

__device__ uchar zeroPosToDigit(ushort a);

__device__ bool hasOneTrueBit(ushort a);

__device__ bool differsAtOneBit(ushort a, ushort b);

__device__ bool updateConstraints(board_t& board, int tid, constraints_t& constraints);

__global__ void solveBoardKernel(board_t* boards, uchar* boardsStatuses);

void solveBoardsWithCuda(board_t* boards, board_t* solvedBoards);

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

#pragma endregion

int main()
{
    board_t boards[BOARDS_NUM];
    board_t solvedBoards[BOARDS_NUM];

    #pragma region Cuda_setup

    cudaStatus = cudaDeviceReset();
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaSetDevice(0);
    validateCudaStatus(cudaStatus);

    #pragma endregion

    BoardsLoader::loadBoardsFromBinary(boards);

    solveBoardsWithCuda(boards, solvedBoards);

    #ifdef DEBUG 
        printBoard(solvedBoards[0]);
    #endif

    return 0;
}

#pragma region Function_implementations


__global__ void solveBoardKernel(board_t* boards, uchar* boardsStatuses)
{
    /*
        I have BLOCKS_NUM blocks available
    */
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ bool isBlockProgressing;
    __shared__ bool isBlockInvalid;
    __shared__ constraints_t constraints;
    __shared__ board_t board;

    memcpy(&board, boards + bid, sizeof(board_t));


    //board_t* board = boards + bid; // TODO to consider if there would be a lack of shared memory

    while (boardsStatuses[BOARDS_NUM] != BoardStatus::Solved) // only first 32 threads can check it REFACTOR
    {
        if (isBlockInvalid) continue;

        isBlockProgressing = 0; // if no update was made it will stay as  0 and stop working

        isBlockProgressing |= updateConstraints(board, tid, constraints);
        __syncthreads();
        
        // TODO check if I can put value to an unambigous field








        __syncthreads();
    }

    memcpy(boards + bid, &board, sizeof(board_t));
}

__device__ uchar zeroPosToDigit(ushort a) // O(1)
{
    switch (~a)
    {
    case 1 << 0: return 0; 
    case 1 << 1: return 1;
    case 1 << 2: return 2;
    case 1 << 3: return 3;
    case 1 << 4: return 4;
    case 1 << 5: return 5;
    case 1 << 6: return 6;
    case 1 << 7: return 7;
    case 1 << 8: return 8;
    case 1 << 9: return 9;
    default: return -1;
    }
}

__device__ bool updateConstraints(board_t& board, int tid, constraints_t& constraints)
{
    bool isBlockProgressing = false;
    ushort old_value;

    old_value = constraints.rows[tid / 9];
    constraints.rows[tid / 9] |= 1 << board[tid] & ~(ushort)1;
    if (differsAtOneBit(old_value, constraints.rows[tid / 9]))
    {
        isBlockProgressing = true;
    }

    old_value = constraints.columns[tid % 9];
    constraints.columns[tid % 9] |= 1 << board[tid] & ~(ushort)1;
    if (differsAtOneBit(old_value, constraints.columns[tid % 9]))
    {
        isBlockProgressing = true;
    }

    old_value = constraints.columns[tid / 9 / 3 * 3 + tid % 9 / 3];
    constraints.columns[tid / 9 / 3 * 3 + tid % 9 / 3] |= 1 << board[tid] & ~(ushort)1;
    if (differsAtOneBit(old_value, constraints.columns[tid / 9 / 3 * 3 + tid % 9 / 3]))
    {
        isBlockProgressing = true;
    }

    return isBlockProgressing;
}

__device__ bool hasOneTrueBit(ushort a)
{
    return a && !(a & (a - 1));
}

__device__ bool differsAtOneBit(ushort a, ushort b)
{
    return hasOneTrueBit(a ^ b);
}

void solveBoardsWithCuda(board_t* boards, board_t* solvedBoards)
{
    board_t* dev_boards;
    uchar* dev_boardsStatuses;

    #pragma region Device_memory_allocation
    // Allocating one additional board memory and boardStatus for a solved board.

    cudaStatus = cudaMalloc((void**)&dev_boards, sizeof(board_t) * (BLOCKS_NUM + 1));
    validateCudaStatus(cudaStatus);

    cudaStatus = cudaMalloc((void**)&dev_boardsStatuses, sizeof(uchar) * (BLOCKS_NUM + 1));
    validateCudaStatus(cudaStatus);

    #pragma endregion

    for (int i = 0; i < BOARDS_NUM; ++i)
    {
        #pragma region Data_for_solving_sudoku_preparation

        cudaStatus = cudaMemcpy(dev_boards, boards + i, sizeof(board_t), cudaMemcpyHostToDevice);
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_boardsStatuses, BoardStatus::Available, (BLOCKS_NUM + 1) * sizeof(uchar));
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemset(dev_boardsStatuses, BoardStatus::InProgress, 1);
        validateCudaStatus(cudaStatus);

        #pragma endregion

        solveBoardKernel<<<BLOCKS_NUM, THREADS_IN_BLOCK_NUM>>>(dev_boards, dev_boardsStatuses);

        cudaStatus = cudaGetLastError();
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaDeviceSynchronize();
        validateCudaStatus(cudaStatus);

        cudaStatus = cudaMemcpy(solvedBoards + i, dev_boards + BLOCKS_NUM, sizeof(board_t), cudaMemcpyDeviceToHost);
        validateCudaStatus(cudaStatus);
    }

    cudaFree(dev_boards);
    cudaFree(dev_boardsStatuses);
}

#pragma endregion
