import random
# defense_points = [0, 1, 9, 81, 729, 6561, 59049]

attack_points = [0, 3, 24, 192, 1536, 12288, 98304]
defense_points = [0, 1, 11, 400, 1600, 6561, 59049]

# ([0, 1, 26, 63, 1264, 3249, 131913], [0, 1, 8, 11, 109, 236, 1953])

# 21 wins against original heuristic
# attack_points = [0, 1, 26, 63, 1264, 3249, 131913]
# defense_points = [0, 1, 8, 80, 109, 236, 1953]

def _is_first_move(state):
    for row in state:
        for col in row:
            if col != " ":
                return False
    return True

def evaluate(state, cur_player, get_point=False, maximizingPlayer=True):
    cur_player = 1 if cur_player == "x" else 2
    max_point = float("-inf")
    n = len(state)
    move = (-1, -1)
    for i in range(n):
        for j in range(n):
            if state[i][j] == " ":
                attack_p = calc_row_AP(state, i, j, cur_player) + calc_col_AP(state, i, j, cur_player) + calc_main_diagonal_AP(state, i, j, cur_player) + calc_sub_diagonal_AP(state, i, j, cur_player)

                defense_p = calc_row_DP(state, i, j, cur_player) + calc_col_DP(state, i, j, cur_player) + calc_main_diagonal_DP(state, i, j, cur_player) + calc_sub_diagonal_DP(state, i, j, cur_player)

                point = max(attack_p, defense_p)
                if max_point < point:
                    max_point = point
                    move = (i, j)
    if get_point == True:
        return max_point
    return move                 
    

def calc_row_AP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n:
            break
        if state[row + i][col] == player:
            player_marks += 1
        elif state[row + i][col] == opponent:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0:
            break
        if state[row - i][col] == player:
            player_marks += 1
        elif state[row - i][col] == opponent:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[min(opponent_marks + 1, len(defense_points) - 1)]
    return total_points

def calc_col_AP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if col + i >= n:
            break
        if state[row][col + i] == player:
            player_marks += 1
        elif state[row][col + i] == opponent:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row < 0:
            break
        if state[row][col - i] == player:
            player_marks += 1
        elif state[row][col - i] == opponent:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[min(opponent_marks + 1, len(defense_points) - 1)]
    
    return total_points

def calc_main_diagonal_AP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n or col + i >= n:
            break
        if state[row + i][col + i] == player:
            player_marks += 1
        elif state[row + i][col + i] == opponent:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0 or col - i < 0:
            break
        if state[row - i][col - i] == player:
            player_marks += 1
        elif state[row - i][col - i] == opponent:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[min(opponent_marks + 1, len(defense_points) - 1)]
    
    return total_points

def calc_sub_diagonal_AP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0

    # forward check
    for i in range(1, 6):
        if row - i < 0 or col + i >= n:
            break
        if state[row - i][col + i] == player:
            player_marks += 1
        elif state[row - i][col + i] == opponent:
            opponent_marks += 1
            break
        else:
            break
    
    # backward check
    for i in range(1, 6):
        if row + i >= n or col - i < 0:
            break
        if state[row + i][col - i] == player:
            player_marks += 1
        elif state[row + i][col - i] == opponent:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[min(opponent_marks + 1, len(defense_points) - 1)]
    
    return total_points



def calc_row_DP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n:
            break
        if state[row + i][col] == player:
            break
        elif state[row + i][col] == opponent:
            opponent_marks += 1
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0:
            break
        if state[row - i][col] == player:
            player_marks += 1
            break
        elif state[row - i][col] == opponent:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[min(opponent_marks, len(defense_points) - 1)]
    return total_points


def calc_col_DP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if col + i >= n:
            break
        if state[row][col + i] == player:
            player_marks += 1
            break
        elif state[row][col + i] == opponent:
            opponent_marks += 1
        else:
            break
    # backward check
    for i in range(1, 6):
        if row < 0:
            break
        if state[row][col - i] == player:
            player_marks += 1
            break
        elif state[row][col - i] == opponent:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[min(opponent_marks, len(defense_points) - 1)]
    
    return total_points


def calc_main_diagonal_DP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n or col + i >= n:
            break
        if state[row + i][col + i] == player:
            player_marks += 1
            break
        elif state[row + i][col + i] == opponent:
            opponent_marks += 1
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0 or col - i < 0:
            break
        if state[row - i][col - i] == player:
            player_marks += 1
            break
        elif state[row - i][col - i] == opponent:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[min(opponent_marks, len(defense_points) - 1)]
    
    return total_points

def calc_sub_diagonal_DP(state, row, col, cur_player = 1):
    n = len(state)
    player, opponent = ("x", "o") if cur_player == 1 else ("o", "x")
    player_marks = 0
    opponent_marks = 0
    total_points = 0

    # forward check
    for i in range(1, 6):
        if row - i < 0 or col + i >= n:
            break
        if state[row - i][col + i] == player:
            player_marks += 1
            break
        elif state[row - i][col + i] == opponent:
            opponent_marks += 1
        else:
            break
    
    # backward check
    for i in range(1, 6):
        if row + i >= n or col - i < 0:
            break
        if state[row + i][col - i] == player:
            player_marks += 1
            break
        elif state[row + i][col - i] == opponent:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[min(opponent_marks, len(defense_points) - 1)]
    
    return total_points