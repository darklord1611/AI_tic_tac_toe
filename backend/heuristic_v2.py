def evaluate(board, player, get_point=True):
    max_point = 0
    directions = [(0, 1), (1, 0), (1, 1), (-1, -1)]
    n = len(board)
    best_move = None
    def count_threats(x, y, dx, dy, player):
        length = 0
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
        if length >= 4:
            return float('inf')
        elif length == 3:
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
        elif length == 1:
            return 10
        return 0
    
    for i in range(n):
        for j in range(n):
            if board[i][j] == " ":
                attack_point = 0
                defense_point = 0
                for dx, dy in directions:
                    attack_point += count_threats(i, j, dx, dy, player)
                    defense_point += count_threats(i, j, dx, dy, "o" if player == "x" else "x")
                point = max(defense_point, attack_point)
                if max_point < point:
                    max_point = point
                    best_move = (i, j)
    print("Max point: ", max_point)
    return best_move
