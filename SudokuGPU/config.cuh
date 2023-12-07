#pragma once

#pragma region General_includes

#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <device_atomic_functions.h>

#pragma endregion

#pragma region Configuration_constants

#define BOARDS_NUM 200
#define BOARD_SIZE 9
#define GROUP_SIZE 3
#define BOARDS_BATCH_SIZE 40
#define THREADS_IN_BLOCK_NUM 81
#define MAX_KERNEL_ITERATIONS 82 // shall not be exceeded, because it will never have more than 81 unambigous situations
#define BLOCKS_NUM 20000
#define DEBUG
const std::string BOARDS_PATH = std::filesystem::current_path().string() + "\\boards_input.bin";
const std::string OUTPUT_PATH = std::filesystem::current_path().string() + "\\boards_output.txt";

#pragma endregion

#pragma region Data_types_definitions

typedef unsigned char uchar;
typedef unsigned int uint;

typedef struct
{
	uint rows[9];
	uint columns[9];
	uint blocks[9];
} constraints_t;

typedef struct
{
	 uchar cells[BOARD_SIZE * BOARD_SIZE];

	 __host__ __device__ uchar& operator[](int id) { return cells[id]; }
} board_t;

typedef enum {
	Invalid = 3,
	Solved = 2,
	InProgress = 1,
	Available = 0,
} BoardStatus;

#pragma endregion







