#pragma once

#pragma region General_includes

#include "cuda_runtime.h"
#include <stdio.h>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "dev_array.cuh"
#include <queue>

#pragma endregion

#pragma region Configuration_constants

#define BOARDS_NUM 200
#define BOARD_SIZE 9
#define GROUP_SIZE 3
#define BOARDS_BATCH_SIZE 20
#define THREADS_IN_BLOCK_NUM 81
#define MAX_KERNEL_ITERATIONS 82 // shall not be exceeded, because it will never have more than 81 unambigous situations
#define BLOCKS_NUM 20000
#define DEBUG
const std::string BOARDS_PATH = std::filesystem::current_path().string() + "\\boards_input.bin";

#pragma endregion

#pragma region Data_types_definitions

typedef unsigned short ushort;
typedef unsigned char uchar;

typedef struct
{
	ushort rows[9];
	ushort columns[9];
	ushort blocks[9];
} constraints_t;

typedef struct
{
	 uchar cells[81];

	 uchar& operator[](int id) { return cells[id]; }
} board_t;

typedef enum {
	Solved = 2,
	InProgress = 1,
	Available = 0,
} BoardStatus;

#pragma endregion







