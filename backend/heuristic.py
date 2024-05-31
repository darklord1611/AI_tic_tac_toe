
opponent_scores = {
    "FOUR": float('inf'),
    "OPEN_THREE": 10000,
    "CLOSED_THREE": 5000,
    "OPEN_TWO": 1000,
    "CLOSED_TWO": 500,
    "ONE": 100,
}

player_scores = {
    "FOUR": float('inf'),
    "OPEN_THREE": 8000,
    "CLOSED_THREE": 4000,
    "OPEN_TWO": 800,
    "CLOSED_TWO": 400,
    "ONE": 80,
}

def get_possible_moves(board):
    def get_boundaries(board):
        n = len(board)
        min_x, min_y = 0, 0
        max_x, max_y = n, n
        for i in range(n):
            for j in range(n):
                if board[i][j] != " ":
                    min_x = max(min_x, i - 5)
                    min_y = max(min_y, j - 5)
                    max_x = min(max_x, i + 5)
                    max_y = min(max_y, j + 5)
        return min_x, min_y, max_x, max_y
    
    possible_moves = []
    min_x, min_y, max_x, max_y = get_boundaries(board)
    for i in range(min_x, max_x):
        for j in range(min_y, max_y):
            if board[i][j] == " ":
                possible_moves.append((i, j))
    return possible_moves


def evaluate_single_step(board, cur_player):
    opponent = 'o' if cur_player == 'x' else 'x'
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    n = len(board)
    max_point = 0

    def count_threats(x, y, dx, dy, player):
        length = 0
        open_ends = 1
        cur_scores = player_scores if player == cur_player else opponent_scores
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
        if length == 4:
            if open_ends > 0:
                return cur_scores["FOUR"]
        elif length == 3:
            if open_ends == 2:
                return cur_scores["OPEN_THREE"]
            elif open_ends == 1:
                return cur_scores["CLOSED_THREE"]
        elif length == 2:
            if open_ends == 2:
                return cur_scores["OPEN_TWO"]
            elif open_ends == 1:
                return cur_scores["CLOSED_TWO"]
        elif length == 1:
            return cur_scores["ONE"]
        return 0

    # Tính điểm cho tất cả các ô
    moves = get_possible_moves(board)
    for x, y in moves:
        attack_point = 0
        defense_point = 0
        for dx, dy in directions:
            attack_point += count_threats(x, y, dx, dy, cur_player)
            defense_point += count_threats(x, y, dx, dy, opponent)
        point = max(defense_point, attack_point)
        if max_point < point:
            max_point = point
            best_move = (x, y)
    print(max_point)
    return best_move