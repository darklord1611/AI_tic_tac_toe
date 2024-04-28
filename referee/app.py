from flask_cors import CORS, cross_origin
from flask import Flask, request, jsonify, make_response
# from flask_socketio import SocketIO
import json
import time
from Board import BoardGame
import copy
import utils

from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {'default': {
        'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
    }},
    'handlers': {'wsgi': {
        'class': 'logging.StreamHandler',
        'stream': 'ext://flask.logging.wsgi_errors_stream',
        'formatter': 'default'
    }},
    'root': {
        'level': 'INFO',
        'handlers': ['wsgi']
    }
})

import logging

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# socketio = SocketIO(app, cors_allowed_origins="*")

app.logger.setLevel(logging.ERROR)


def log(*args):
    params = []
    for x in args:
        params.append(f"{x}")

    app.logger.error(" ".join(params))


# Global variance
PORT = 80
team1_role = "x"
team2_role = "o"
size = 5
#################

rooms = {}
room_by_teams = {}

BOARD = []
for i in range(size):
    BOARD.append([])
    for j in range(size):
        BOARD[i].append(' ')


@app.route('/init', methods=['POST'])
@cross_origin()
def get_data():
    log("/init")
    data = request.data
    info = json.loads(data.decode('utf-8'))
    log(info)
    global rooms
    global room_by_teams
    room_id = info["room_id"]
    is_init = False
    if room_id not in rooms:
        match_id = 1
        team1_id = info["team1_id"]
        team2_id = info["team2_id"]
        team1_id_full = team1_id + "+" + team1_role
        team2_id_full = team2_id + "+" + team2_role
        room_by_teams[team1_id] = room_id
        room_by_teams[team2_id] = room_id
        board_game = BoardGame(size, BOARD, room_id, match_id, team1_id_full, team2_id_full)
        rooms[room_id] = board_game
        is_init = True

    board_game = rooms[room_id]
    return {
        "room_id": board_game.game_info["room_id"],
        "match_id": board_game.game_info["match_id"],
        "team1_id": board_game.game_info["team1_id"],
        "team2_id": board_game.game_info["team2_id"],
        "size": board_game.game_info["size"],
        "init": is_init,
    }


@app.route('/', methods=['POST'])
@cross_origin()
def render_board():
    data = request.data
    info = json.loads(data.decode('utf-8'))
    log(info['team_id'])
    global rooms
    room_id = info["room_id"]
    board_game = rooms[room_id]
    team1_id_full = board_game.game_info["team1_id"]
    team2_id_full = board_game.game_info["team2_id"]
    time_list = board_game.timestamps

    if (info["team_id"] == team1_id_full and not board_game.start_game):
        time_list[0] = time.time()
        board_game.start_game = True
    # log(f'Board: {board_game.game_info["board"]}')
    response = make_response(jsonify(board_game.game_info))
    return board_game.game_info


@app.route('/')
@cross_origin()
def fe_render_board():
    global rooms
    if "room_id" not in request.args:
        return {
            "code": 1,
            "error": "missing room_id"
        }
    room_id = request.args.get('room_id')
    if room_id not in rooms:
        return {
            "code": 1,
            "error": f"not found room: {room_id}"
        }
    board_game = rooms[room_id]
    # log(board_game.game_info)
    response = make_response(jsonify(board_game.game_info))
    # log(board_game.game_info)
    return response


@app.route('/move', methods=['POST'])
@cross_origin()
def handle_move():
    log("handle_move")
    data = request.data

    data = json.loads(data.decode('utf-8'))
    global rooms
    room_id = data["room_id"]
    if room_id not in rooms:
        return {
            "code": 1,
            "error": "Room not found"
        }
    board_game = rooms[room_id]
    team1_id_full = board_game.game_info["team1_id"]
    team2_id_full = board_game.game_info["team2_id"]
    time_list = board_game.timestamps

    log(f"game info: {board_game.game_info}")
    if data["turn"] == board_game.game_info["turn"] and data["status"] == None:
        board_game.game_info.update(data)
        if data["turn"] == team1_id_full:
            board_game.game_info["time1"] += time.time() - time_list[0]
            board_game.game_info["turn"] = team2_id_full
            time_list[1] = time.time()
        else:
            board_game.game_info["time2"] += time.time() - time_list[1]
            board_game.game_info["turn"] = team1_id_full
            time_list[0] = time.time()
    log("Team 1 time: ", time_list[0])
    log("Team 2 time: ", time_list[1])
    if data["status"] == None:
        log("Checking status...")
        board_game.check_status(data["board"])
    # log("After check status: ",board_game.game_info)

    # board_game.convert_board(board_game.game_info["board"])

    return {
        "code": 0,
        "error": "",
        "status": board_game.game_info["status"],
        "size": board_game.game_info["size"],
        "turn": board_game.game_info["turn"],
        "time1": board_game.game_info["time1"],
        "time2": board_game.game_info["time2"],
        "score1": board_game.game_info["score1"],
        "score2": board_game.game_info["score2"],
        "board": board_game.game_info["board"],
        "room_id": board_game.game_info["room_id"],
        "match_id": board_game.game_info["match_id"]
    }


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=PORT)
