import random
import pickle
import struct

# def generate_sudoku():
#     board = [x for x in range(1, 10) for _ in range(9)]
#     random.shuffle(board)

#     ids = list(range(81))

#     numbers_to_remove = random.randint(81 - 40, 81 - 4)

#     random_elements = random.sample(ids, numbers_to_remove)

#     for x in random_elements:
#         board[x] = 0

#     return board
    
def generate_sudoku():
    # Initialize empty board
    board = [[0 for _ in range(9)] for _ in range(9)]
    fill_board(board)

    

    res =  [cell for row in board for cell in row]

    ids = list(range(81))

    numbers_to_remove = random.randint(81 - 40, 81 - 4)

    random_elements = random.sample(ids, numbers_to_remove)

    for x in random_elements:
        res[x] = 0

    return res

def fill_board(board):
    # Start filling the board using backtracking
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if fill_board(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def is_valid(board, row, col, num):
    # Check if the current placement of the number is valid in the board
    return (
        is_row_valid(board, row, num)
        and is_col_valid(board, col, num)
        and is_box_valid(board, row - row % 3, col - col % 3, num)
    )

def is_row_valid(board, row, num):
    return num not in board[row]

def is_col_valid(board, col, num):
    return num not in [board[row][col] for row in range(9)]

def is_box_valid(board, start_row, start_col, num):
    for row in range(3):
        for col in range(3):
            if board[row + start_row][col + start_col] == num:
                return False
    return True


def write_sudoku_to_file(filename, num_boards):
    
    with open(filename, 'wb') as file:
        file.write(struct.pack('B', 5))
        file.write(struct.pack('B',3 ))
        file.write(struct.pack('B',0))
        file.write(struct.pack('B',0 ))
        file.write(struct.pack('B', 7))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 6))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 1))
        file.write(struct.pack('B', 9))
        file.write(struct.pack('B', 5))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 9))
        file.write(struct.pack('B', 8))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 6))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 8))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 6))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 3))
        file.write(struct.pack('B', 4))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 8))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 3))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 1))
        file.write(struct.pack('B', 7))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 2))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 6))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 6))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 2))
        file.write(struct.pack('B', 8))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 4))
        file.write(struct.pack('B', 1))
        file.write(struct.pack('B', 9))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 5))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 8))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 0))
        file.write(struct.pack('B', 7))
        file.write(struct.pack('B', 9))
        for _ in range(num_boards):
            sudoku_board = generate_sudoku()
            for num in sudoku_board:
                byte_value = struct.pack('B', num)  # Convert number to a single byte (8-bit value)
                file.write(byte_value)

if __name__ == "__main__":
    file_name = "boards_input.bin"
    num_boards_to_generate = 200 - 1  # Change this value to generate more boards

    write_sudoku_to_file(file_name, num_boards_to_generate)
    print(f"{num_boards_to_generate} Sudoku boards of size 9x9 have been generated and saved in {file_name}.")



# # Generate Sudoku board
# if __name__ == "__main__":
#     sudoku_board = generate_sudoku()
#     for i in range(0, len(sudoku_board), 9):
#         print(sudoku_board[i:i+9])