#pragma once
#include "config.cuh"

class BoardsLoader
{
public:
    static void loadBoardsFromBinary(board_t* boards)
    {
        std::ifstream file(BOARDS_PATH, std::ios::binary);

        if (!file.is_open()) {
            std::cerr << "Unable to open file " << BOARDS_PATH << std::endl;
            return;
        }

        for (int i = 0; i < BOARDS_NUM; ++i) {
            file.read(reinterpret_cast<char*>(&boards[i].cells), sizeof(board_t));

            if (file.eof()) {
                break;
            }
        }

        file.close();
    }

    static void saveBoardsToTxt(board_t* boards) {
        std::ofstream file(OUTPUT_PATH);

        if (!file.is_open()) {
            std::cerr << "Unable to open file " << OUTPUT_PATH << std::endl;
            return;
        }

        for (int boardIndex = 0; boardIndex < BOARDS_NUM; ++boardIndex) {
            for (int i = 0; i < BOARD_SIZE * BOARD_SIZE; ++i) {
                file << char(boards[boardIndex].cells[i] + '0') << " ";
                if ((i + 1) % BOARD_SIZE == 0) {
                    file << std::endl;
                }
            }
            file << std::endl;
        }

        file.close();
    }
};

