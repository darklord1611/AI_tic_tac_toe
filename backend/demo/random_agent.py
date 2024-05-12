import random


def random_action(state):
    temp = state.flatten()
    valid_actions = [i for i in range(len(temp)) if temp[i] == 0]
    return random.choice(valid_actions)



attack_points = [0, 3, 24, 192, 1536, 12288, 98304]
defense_points = [0, 1, 9, 81, 729, 6561, 59049]

def evaluate(state):
    max_point = 0
    n = len(state)
    move = [-1, -1]
    for i in range(n):
        for j in range(n):
            if state[i][j] == 0:
                attack_p = calc_row_AP(state, i, j, 1) + calc_col_AP(state, i, j, 1) + calc_main_diagonal_AP(state, i, j, 1) + calc_sub_diagonal_AP(state, i, j, 1)
                defense_p = calc_row_DP(state, i, j, 1) + calc_col_DP(state, i, j, 1) + calc_main_diagonal_DP(state, i, j, 1) + calc_sub_diagonal_DP(state, i, j, 1)
                point = max(attack_p, defense_p)
                max_point = max(max_point, point)
                move = [i, j]

    return move                 
    

def calc_row_AP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n:
            break
        if state[row + i][col] == cur_player:
            player_marks += 1
        elif state[row + i][col] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0:
            break
        if state[row - i][col] == cur_player:
            player_marks += 1
        elif state[row - i][col] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[opponent_marks + 1]
    return total_points

def calc_col_AP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if col + i >= n:
            break
        if state[row][col + i] == cur_player:
            player_marks += 1
        elif state[row][col + i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row < 0:
            break
        if state[row][col - i] == cur_player:
            player_marks += 1
        elif state[row][col - i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[opponent_marks + 1]
    
    return total_points

def calc_main_diagonal_AP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n or col + i >= n:
            break
        if state[row + i][col + i] == cur_player:
            player_marks += 1
        elif state[row + i][col + i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0 or col - i < 0:
            break
        if state[row - i][col - i] == cur_player:
            player_marks += 1
        elif state[row - i][col - i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[opponent_marks + 1]
    
    return total_points

def calc_sub_diagonal_AP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0

    # forward check
    for i in range(1, 6):
        if row - i < 0 or col + i >= n:
            break
        if state[row - i][col + i] == cur_player:
            player_marks += 1
        elif state[row - i][col + i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    
    # backward check
    for i in range(1, 6):
        if row + i >= n or col - i < 0:
            break
        if state[row + i][col - i] == cur_player:
            player_marks += 1
        elif state[row + i][col - i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    
    total_points = attack_points[player_marks] - defense_points[opponent_marks + 1]
    
    return total_points



def calc_row_DP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n:
            break
        if state[row + i][col] == cur_player:
            break
        elif state[row + i][col] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0:
            break
        if state[row - i][col] == cur_player:
            player_marks += 1
            break
        elif state[row - i][col] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[opponent_marks]
    return total_points


def calc_col_DP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if col + i >= n:
            break
        if state[row][col + i] == cur_player:
            player_marks += 1
            break
        elif state[row][col + i] == 3 - cur_player:
            opponent_marks += 1
            break
        else:
            break
    # backward check
    for i in range(1, 6):
        if row < 0:
            break
        if state[row][col - i] == cur_player:
            player_marks += 1
            break
        elif state[row][col - i] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[opponent_marks]
    
    return total_points


def calc_main_diagonal_DP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0
    # forward check
    for i in range(1, 6):
        if row + i >= n or col + i >= n:
            break
        if state[row + i][col + i] == cur_player:
            player_marks += 1
            break
        elif state[row + i][col + i] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    # backward check
    for i in range(1, 6):
        if row - i < 0 or col - i < 0:
            break
        if state[row - i][col - i] == cur_player:
            player_marks += 1
            break
        elif state[row - i][col - i] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[opponent_marks]
    
    return total_points

def calc_sub_diagonal_DP(state, row, col, cur_player = 1):
    n = len(state)
    player_marks = 0
    opponent_marks = 0
    total_points = 0

    # forward check
    for i in range(1, 6):
        if row - i < 0 or col + i >= n:
            break
        if state[row - i][col + i] == cur_player:
            player_marks += 1
            break
        elif state[row - i][col + i] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    
    # backward check
    for i in range(1, 6):
        if row + i >= n or col - i < 0:
            break
        if state[row + i][col - i] == cur_player:
            player_marks += 1
            break
        elif state[row + i][col - i] == 3 - cur_player:
            opponent_marks += 1
        else:
            break
    
    total_points = defense_points[opponent_marks]
    
    return total_points