
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
    center = size // 2
    moves = []
    for i in range(size):
        for j in range(size):
            if board[i][j] == ' ':
                score = 0
                directions = [
                    [(i, j + k) for k in range(-4, 5) if 0 <= j + k < size],
                    [(i + k, j) for k in range(-4, 5) if 0 <= i + k < size],
                    [(i + k, j + k) for k in range(-4, 5) if 0 <= i + k < size and 0 <= j + k < size],
                    [(i + k, j - k) for k in range(-4, 5) if 0 <= i + k < size and 0 <= j - k < size]
                ]
                for direction in directions:
                    line_potential = 0
                    for x, y in direction:
                        if board[x][y] == ' ':
                            line_potential += 1
                        elif board[x][y] != board[i][j]:
                            line_potential = 0
                            break
                    score += line_potential
                #Thêm điểm ưu tiên cho các ô gần tâm bàn cờ hơn
                score += (1 - (abs(center - i) + abs(center - j)) / center) * 2
                moves.append((score, i, j))
    moves.sort(reverse=True, key=lambda x: x[0]) #soort theo score
    return [(i, j) for _, i, j in moves]

#Hàm evaluate_board để đánh giá trạng thái hiện tại của bàn cờ dựa trên lợi thế củangười chơi so với đối thủ
def evaluate_board(board, player):
    score = 0
    size = len(board)
    opponent = 'o' if player == 'x' else 'x'

    # Directions arrays for sequences check
    directions = [
        [(0, 1)],  # Horizontal
        [(1, 0)],  # Vertical
        [(1, 1)],  # Diagonal right
        [(1, -1)]  # Diagonal left
    ]

    for i in range(size):
        for j in range(size):
            if board[i][j] == player:
                # Check all directions
                for direction in directions:
                    count = 0
                    for k in range(1, 5):  # Check up to 4 cells from the current position
                        ni, nj = i + direction[0][0] * k, j + direction[0][1] * k
                        if 0 <= ni < size and 0 <= nj < size and board[ni][nj] == player:
                            count += 1
                        else:
                            break
                    score += count ** 2  # Square the count to prioritize longer sequences
            elif board[i][j] == opponent:
                for direction in directions:
                    count = 0
                    for k in range(1, 5):
                        ni, nj = i + direction[0][0] * k, j + direction[0][1] * k
                        if 0 <= ni < size and 0 <= nj < size and board[ni][nj] == opponent:
                            count += 1
                        else:
                            break
                    score -= count ** 2  # Penalize opponent sequences similarly

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
        return best_move, minEval


def get_move(board, player, max_depth=5):
    best_move, _ = minimax(board, 0, max_depth, float('-inf'), float('inf'), True if player == 'x' else False, player)
    return best_move


