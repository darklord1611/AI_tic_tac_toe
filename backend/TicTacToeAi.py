
import random
from heuristic import _is_first_move
from heuristic_v2 import evaluate as evaluate_v2
import numpy as np

def random_move(board):
    size = len(board)
    possible_moves = get_possible_moves(board)
    return possible_moves[random.randint(0, len(possible_moves) - 1)]

def is_winner(board, player):
    size = len(board)
    for i in range(size):
        for j in range(size - 4):
            if all(board[i][j+k] == player for k in range(5)):
                return True
            if all(board[j+k][i] == player for k in range(5)):
                return True
    for i in range(size - 4):
        for j in range(size - 4):
            if all(board[i+k][j+k] == player for k in range(5)):
                return True
            if all(board[i+k][j+4-k] == player for k in range(5)):
                return True
    return False


#Hàm get_possible_moves được thiết kế để lấy danh sách các nước đi khả dụng,
# đồng thời đánh giá và sắp xếp các nước đi dựa trên tiềm năng chiến thắng của chúng.
def get_possible_moves(board):
    size = len(board)
    possible_moves = []
    move_scores = {}

    for i in range(size):
        for j in range(size):
            if board[i][j] == ' ':
                score = 0
                # Kiểm tra các ô xung quanh để đánh giá tiềm năng
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < size and 0 <= nj < size and board[ni][nj] != ' ':
                            score += 1
                            # Đánh giá tiềm năng tạo thành chuỗi
                            # Điểm bổ sung cho chuỗi có tiềm năng lớn hơn
                            length, open_ends = check_line_length(board, ni, nj, di, dj, board[ni][nj])
                            if length >= 4:
                                score += 100
                            elif length == 3 and open_ends == 2:
                                score += 50
                if score > 0:
                    possible_moves.append((i, j))
                    move_scores[(i, j)] = score

    # Sắp xếp các nước đi theo điểm số giảm dần
    sorted_moves = sorted(possible_moves, key=lambda move: move_scores[move], reverse=True)
    return sorted_moves

def check_line_length(board, x, y, dx, dy, player):
    length = 1
    open_ends = 0

    # Kiểm tra một hướng
    nx, ny = x + dx, y + dy
    while 0 <= nx < len(board) and 0 <= ny < len(board) and board[nx][ny] == player:
        length += 1
        nx += dx
        ny += dy
    if 0 <= nx < len(board) and 0 <= ny < len(board) and board[nx][ny] == ' ':
        open_ends += 1

    # Kiểm tra hướng ngược lại
    nx, ny = x - dx, y - dy
    while 0 <= nx < len(board) and 0 <= ny < len(board) and board[nx][ny] == player:
        length += 1
        nx -= dx
        ny -= dy
    if 0 <= nx < len(board) and 0 <= ny < len(board) and board[nx][ny] == ' ':
        open_ends += 1

    return length, open_ends

# Hàm evaluate_board để đánh giá trạng thái hiện tại của bàn cờ dựa trên lợi thế của người chơi so với đối thủ
def evaluate_board(board, player):
    score = 0
    opponent = 'o' if player == 'x' else 'x'
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    n = len(board)

    def count_threats(x, y, dx, dy, player):
        length = 1
        open_ends = 0

        # Kiểm tra hướng đi xuôi
        nx, ny = x + dx, y + dy
        while 0 <= nx < n and 0 <= ny < n and board[nx][ny] == player:
            length += 1
            nx += dx
            ny += dy
        if 0 <= nx < n and 0 <= ny < n and board[nx][ny] == ' ':
            open_ends += 1

        # Kiểm tra hướng đi ngược
        nx, ny = x - dx, y - dy
        while 0 <= nx < n and 0 <= ny < n and board[nx][ny] == player:
            length += 1
            nx -= dx
            ny -= dy
        if 0 <= nx < n and 0 <= ny < n and board[nx][ny] == ' ':
            open_ends += 1

        # Điểm số dựa trên độ dài và số đầu mở
        if length >= 5:
            return float('inf')
        elif length == 4:
            if open_ends == 2:
                return 10000
            elif open_ends == 1:
                return 5000
        elif length == 3:
            if open_ends == 2:
                return 1000
            elif open_ends == 1:
                return 500
        elif length == 2:
            if open_ends == 2:
                return 100
            elif open_ends == 1:
                return 50
        return 0

    # Tính điểm cho tất cả các ô
    for x in range(n):
        for y in range(n):
            if board[x][y] == player or board[x][y] == opponent:
                current_player = board[x][y]
                for dx, dy in directions:
                    score += count_threats(x, y, dx, dy, current_player) * (1 if current_player == player else -1)

    return score

def minimax(board, depth, max_depth, alpha, beta, maximizingPlayer, player):
    size = len(board)
    opponent = 'o' if player == 'x' else 'x'
    if depth == max_depth:
        return None, evaluate_board(board, player)
    if is_winner(board, player):
        return None, float('inf')  # Hoặc một giá trị rất lớn phù hợp
    if is_winner(board, opponent):
        return None, float('-inf')  # Hoặc một giá trị rất nhỏ phù hợp

    possible_moves = get_possible_moves(board)
    # ?
    if not possible_moves:  # Không còn nước đi
        return None, evaluate_board(board, player)

    if maximizingPlayer:
        maxEval = float('-inf')
        best_move = None
        for i, j in possible_moves:
            board[i][j] = player
            _, eval = minimax(board, depth + 1, max_depth, alpha, beta, False, player)
            board[i][j] = ' '
            if eval > maxEval:
                maxEval = eval
                best_move = (i, j)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        if best_move == None:
            best_move = random_move(board)
        return best_move, maxEval
    else:
        minEval = float('inf')
        best_move = None
        for i, j in possible_moves:
            board[i][j] = opponent
            _, eval = minimax(board, depth + 1, max_depth, alpha, beta, True, player)
            board[i][j] = ' '
            if eval < minEval:
                minEval = eval
                best_move = (i, j)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        if best_move == None:
            best_move = random_move(board)
        return best_move, minEval


def get_move(board, player, max_depth=3, policy="minimax"):
    # new_board = np.array(board, dtype=int)
    print(f"Using policy: {policy}")
    if _is_first_move(board):
        return (len(board) // 2, len(board) // 2)
    match(policy):
        case "minimax":
            best_move, _ = minimax(board, 0, max_depth, float('-inf'), float('inf'), True if player == 'x' else False, player)
        case "heuristic":
            best_move = evaluate_v2(board, True if player == 'x' else False, False)
    return best_move

