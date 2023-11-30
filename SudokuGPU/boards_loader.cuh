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
};

