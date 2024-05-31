import unittest

from Board import BoardGame


class BoardTest(unittest.TestCase):
    def test_make_empty_board(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])

        self.assertTrue( board.is_empty(board.board), "test empty board")

    def test_x_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[['x', 'x', 'x', 'x', 'x'], [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])

        status = board.is_win(board.board)
        print(status)
        self.assertTrue("X won" == status, "test win")

    def test_x_win_diagonal(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[['x', ' ', ' ', ' ', ' '], [' ', 'x', ' ', ' ', ' '], [' ', ' ', 'x', ' ', ' '],
                                 [' ', ' ', ' ', 'x', ' '], [' ', ' ', ' ', ' ', 'x']])

        status = board.is_win(board.board)
        print(status)
        self.assertTrue("X won" == status, "test win")

    def test_x_win_diagonal_2(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', 'x'], [' ', ' ', ' ', 'x', ' '], [' ', ' ', 'x', ' ', ' '],
                                 [' ', 'x', ' ', ' ', ' '], ['x', ' ', ' ', ' ', ' ']])

        status = board.is_win(board.board)
        print(status)
        self.assertTrue("X won" == status, "test win")

    def test_x_not_win_line_1(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[['x', 'x', 'x', 'x', ' '], [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    def test_x_not_win_line_2(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '], ['x', 'x', 'x', 'x', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    ### create more test cases for 'o' win, draw, continue playing
    def test_o_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[['o', 'o', 'o', 'o', 'o'], [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])

        status = board.is_win(board.board)
        print(status)
        self.assertTrue("O won" == status, "test win")

    def test_o_win_diagonal(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[['o', ' ', ' ', ' ', ' '], [' ', 'o', ' ', ' ', ' '], [' ', ' ', 'o', ' ', ' '],
                                 [' ', ' ', ' ', 'o', ' '], [' ', ' ', ' ', ' ', 'o']])

        status = board.is_win(board.board)
        print(status)
        self.assertTrue("O won" == status, "test win")

    def test_o_win_diagonal_2(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', 'o'], [' ', ' ', ' ', 'o', ' '], [' ', ' ', 'o', ' ', ' '],
                                 [' ', 'o', ' ', ' ', ' '], ['o', ' ', ' ', ' ', ' ']])

        status = board.is_win(board.board)
        print(status)
        self.assertTrue("O won" == status, "test win")


    ### create sequence of 4x
    def test_4_x_not_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[['x', 'x', 'x', 'x', ' '], [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    def test_4_x_line2_not_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '], ['x', 'x', 'x', 'x', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    def test_4_x_line2_right_not_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '], [' ', 'x', 'x', 'x', 'x'], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    def test_4_x_line_3_not_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '], [' ', 'x', 'x', 'x', 'x'], [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    def test_4_x_line_4_not_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '],  [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', 'x', 'x', 'x', 'x'], [' ', ' ', ' ', ' ', ' ']])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")

    def test_4_x_line_5_not_win(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', ' ', ' ', ' ', ' '],  [' ', ' ', ' ', ' ', ' '],
                                 [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', 'x', 'x', 'x', 'x'], ])
        self.assertNotEqual("X won", board.is_win(board.board), "test win")
    
    def test_diff_same(self):
        board = BoardGame(size=5,
                          room_id="room-1",
                          match_id=1,
                          board=[[' ', 'x', ' ', ' ', ' '],  [' ', ' ', ' ', ' ', ' '],
                                 [' ', 'o', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                 [' ', 'o', ' ', ' ', ' '], ])
        self.assertEqual([], BoardGame.diff(board.board, board.board), "test diff")
    
    def test_diff_1(self):
        board1 = BoardGame(size=5,
                           room_id="room-1",
                           match_id=1,
                           board=[[' ', 'x', ' ', ' ', ' '],  [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], ])
        board2 = BoardGame(size=5,
                           room_id="room-1",
                           match_id=1,
                           board=[[' ', ' ', ' ', ' ', ' '],  [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], ])
        self.assertEqual([(0, 1)], BoardGame.diff(board1.board, board2.board), "test diff")
        self.assertEqual([(0, 1)], BoardGame.diff(board2.board, board1.board), "test diff")
    
    def test_diff_multiple(self):
        board1 = BoardGame(size=5,
                           room_id="room-1",
                           match_id=1,
                           board=[[' ', 'x', 'o', ' ', ' '],  [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], ])
        board2 = BoardGame(size=5,
                           room_id="room-1",
                           match_id=1,
                           board=[[' ', ' ', ' ', ' ', ' '],  ['o', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' '],
                                  [' ', ' ', ' ', ' ', ' '], ])
        self.assertEqual([(0, 1), (0, 2), (1, 0)], BoardGame.diff(board1.board, board2.board), "test diff")
        self.assertEqual([(0, 1), (0, 2), (1, 0)], BoardGame.diff(board2.board, board1.board), "test diff")

if __name__ == '__main__':
    unittest.main()
