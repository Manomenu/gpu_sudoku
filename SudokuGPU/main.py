import random
import pickle
import struct

def generate_sudoku():
    base = 3
    side = base * base

    numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    board = [[numbers[random.randint(0, 20)] for _ in range(9)] for _ in range(9)]
    return board

def write_sudoku_to_file(filename, num_boards):
    with open(filename, 'wb') as file:
        for _ in range(num_boards):
            sudoku_board = generate_sudoku()
            for row in sudoku_board:
                for num in row:
                    byte_value = struct.pack('B', num)  # Convert number to a single byte (8-bit value)
                    file.write(byte_value)

if __name__ == "__main__":
    file_name = "boards_input.bin"
    num_boards_to_generate = 200  # Change this value to generate more boards

    write_sudoku_to_file(file_name, num_boards_to_generate)
    print(f"{num_boards_to_generate} Sudoku boards of size 9x9 have been generated and saved in {file_name}.")
