import json
import sys
from abc import abstractmethod
from typing import List, Optional, Union


def initialize_board() -> List[List[Optional['Piece']]]:
    """ Инициализация доски стандартной расстановкой фигур """
    return [
        [Rook(False), Knight(False), Bishop(False), Queen(False), King(False), Bishop(False), Knight(False),
         Rook(False)],
        [Pawn(False) for _ in range(8)],
        [None] * 8,
        [None] * 8,
        [None] * 8,
        [None] * 8,
        [Pawn(True) for _ in range(8)],
        [Rook(True), Knight(True), Bishop(True), Queen(True), King(True), Bishop(True), Knight(True), Rook(True)]
    ]


class Board:
    def __init__(self, board=None):
        """
        Инициализация доски
        :param board: Опционально принимает JSON-представление доски для восстановления состояния
        """
        self.is_white_turn = True
        self.is_looped = False
        self.pieces = {
            "P": Pawn(True), "p": Pawn(False),
            "R": Rook(True), "r": Rook(False),
            "B": Bishop(True), "b": Bishop(False),
            "N": Knight(True), "n": Knight(False),
            "K": King(True), "k": King(False),
            "Q": Queen(True), "q": Queen(False)}

        if not board:
            self.board = initialize_board()
        else:
            self.board = [[None for _ in range(8)] for _ in range(8)]
            for pos, piece in self.from_json(board).items():
                col, row = parse_position(pos)
                self.board[row][col] = self.pieces[piece]

    def __str__(self) -> str:
        """ Форматированный вывод доски на экран """
        display = []
        for row in self.board:
            display.append(' '.join(str(piece) if piece else '.' for piece in row))

        output = "\n   A B C D E F G H \n\n"
        for i, row_str in enumerate(display):
            output += f"{8 - i}  {row_str}  {8 - i}\n"

        output += "\n   A B C D E F G H\n"

        return output

    def __getitem__(self, pos: List[int]) -> Optional[List[None]]:
        """ Позволяет получить фигуру по позиции """
        col, row = pos
        return self.board[row][col]

    def __setitem__(self, pos: List[int], piece: Union['Piece', List[None], None]) -> None:
        """ Позволяет поменять фигуру на доске по позиции """
        col, row = pos
        self.board[row][col] = piece

    def __contains__(self, piece: 'Piece') -> bool:
        """ Проверка наличия фигуры на доске """
        return any(piece in row for row in self.board)

    @property
    def num_white_pieces(self) -> int:
        """ Возвращает количество белых фигур на доске """
        return sum(1 for row in self.board for piece in row if piece and piece.is_white)

    @property
    def num_black_pieces(self) -> int:
        """ Возвращает количество черных фигур на доске """
        return sum(1 for row in self.board for piece in row if piece and not piece.is_white)

    def balance(self) -> int:
        """
        Вычисляет баланс фигур на доске, где белые фигуры положительно влияют на результат, а черные — отрицательно.

        :return: Разница между суммой значений белых и черных фигур.
        """
        return sum(piece.value for row in self.board for piece in row if piece and piece.is_white) - \
               sum(piece.value for row in self.board for piece in row if piece and not piece.is_white)

    def move_piece(self, from_pos: List[int], to_pos: List[int]) -> None:
        """ Метод для перемещения фигуры """
        piece = self[from_pos]
        if piece is None:
            raise ValueError("The piece cannot make the specified move.")

        if self.is_white_turn and not piece.is_white or not self.is_white_turn and piece.is_white:
            raise ValueError("The piece cannot make the specified move.")

        if not piece.can_move(self, from_pos, to_pos):
            raise ValueError("The piece cannot make the specified move.")

        self[to_pos] = piece
        self[from_pos] = None
        piece.position = to_pos

    def make_looped(self) -> None:
        """ Переводит доску в зацикленный режим """
        self.is_looped = True

    def make_default(self) -> None:
        """ Возвращает доску в обычный режим """
        self.is_looped = False

    def to_json(self) -> str:
        """ Преобразует доску в JSON """
        data = {}
        for i in range(8):
            for j in range(8):
                if self.board[7 - j][i]:
                    data[f"{chr(i + ord('a'))}{j + 1}"] = str(self.board[7 - j][i])

        return json.dumps(data, indent=4, sort_keys=True)

    @staticmethod
    def from_json(json_data: str):
        """ Восстанавливает состояние доски из JSON """
        return json.loads(json_data)


class Piece:
    def __init__(self, is_white: bool, symbol: str, value: int):
        """ Инициализация фигуры """
        self.is_white = is_white
        self.symbol = symbol
        self.value = value

    def __str__(self) -> str:
        """ Возвращает символ фигуры в зависимости от цвета """
        return self.symbol.upper() if self.is_white else self.symbol.lower()

    @abstractmethod
    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        """ Абстрактный метод, который определяет возможность хода для фигуры """
        pass


class King(Piece):
    def __init__(self, is_white: bool):
        super().__init__(is_white, "k", 0)

    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        """ Проверка возможности хода короля """
        if board.is_looped:
            delta_col = (to_pos[0] - from_pos[0]) % 8
            delta_row = (to_pos[1] - from_pos[1]) % 8
        else:
            delta_col = abs(to_pos[0] - from_pos[0])
            delta_row = abs(to_pos[1] - from_pos[1])

        if max(delta_col, delta_row) == 1:
            target_piece = board[to_pos]
            if target_piece is None or target_piece.is_white != self.is_white:
                return True
        return False


class Queen(Piece):
    def __init__(self, is_white: bool):
        super().__init__(is_white, "q", 9)

    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        # Королева может двигаться как ладья или как слон
        return Rook(self.is_white).can_move(board, from_pos, to_pos) or \
               Bishop(self.is_white).can_move(board, from_pos, to_pos)


class Rook(Piece):
    def __init__(self, is_white: bool):
        super().__init__(is_white, "r", 5)

    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        """ Проверка возможного хода ладьи """
        delta_col = to_pos[0] - from_pos[0]
        delta_row = to_pos[1] - from_pos[1]

        if delta_row != 0 and delta_col != 0:
            return False  # Ладья двигается только по прямым линиям

        # Проверяем стандартный путь
        if self.path_checker(board, from_pos, to_pos, False):
            return True

        # Если доска зациклена, проверяем путь через края доски
        if board.is_looped:
            return self.path_checker(board, from_pos, to_pos, True)

    def path_checker(self, board: 'Board', from_pos: List[int], to_pos: List[int], direction: bool) -> bool:
        """ Проверяет, свободен ли путь для движения ладьи """
        delta_col = to_pos[0] - from_pos[0]
        delta_row = to_pos[1] - from_pos[1]

        step_row = 0 if delta_row == 0 else (1 if delta_row > 0 else -1)
        step_col = 0 if delta_col == 0 else (1 if delta_col > 0 else -1)

        if direction:  # Обратная дорога, если looped
            step_row *= -1
            step_col *= -1

        cur_row, cur_col = (from_pos[1] + step_row) % 8, (from_pos[0] + step_col) % 8

        while [cur_col, cur_row] != to_pos:
            if board[[cur_col, cur_row]] is not None:
                return False
            cur_row = (cur_row + step_row) % 8
            cur_col = (cur_col + step_col) % 8

        target_piece = board[to_pos]
        if target_piece is None or target_piece.is_white != self.is_white:
            return True

        return False


class Bishop(Piece):
    def __init__(self, is_white: bool):
        super().__init__(is_white, "b", 3)

    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        """ Проверка возможного хода слона """
        if board.is_looped:
            # Если доска зациклена, то делаем перебор всех диагоналей, куда ладья может в теории пойти
            for step_row in [-1, 1]:
                for step_col in [-1, 1]:
                    if self.path_check(board, from_pos, to_pos, step_row, step_col):
                        return True
        else:
            # Если доска не зациклена идем по направлению к намеченной позиции
            delta_col = to_pos[0] - from_pos[0]
            delta_row = to_pos[1] - from_pos[1]

            if abs(delta_row) != abs(delta_col):
                return False  # Слон двигается только по диагонали

            step_row = 1 if delta_row > 0 else -1
            step_col = 1 if delta_col > 0 else -1
            return self.path_check(board, from_pos, to_pos, step_row, step_col)

    def path_check(self, board: 'Board', from_pos: List[int], to_pos: List[int], step_row: int, step_col: int) -> bool:
        """ Проверяет, свободен ли путь для движения ладьи """
        cur_row, cur_col = (from_pos[1] + step_row) % 8, (from_pos[0] + step_col) % 8

        while [cur_col, cur_row] != to_pos:
            if board[[cur_col, cur_row]] is not None:
                return False
            cur_row = (cur_row + step_row) % 8
            cur_col = (cur_col + step_col) % 8

        target_piece = board[to_pos]
        if target_piece is None or target_piece.is_white != self.is_white:
            return True

        return False


class Knight(Piece):
    def __init__(self, is_white: bool):
        super().__init__(is_white, "n", 3)

    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        """ Проверка возможного хода коня """
        if board.is_looped:
            delta_col = (to_pos[0] - from_pos[0]) % 8
            delta_row = (to_pos[1] - from_pos[1]) % 8

            # Обработка случаев, когда конь идет через край
            delta_col = 8 - delta_col if delta_col > 2 else delta_col
            delta_row = 8 - delta_row if delta_row > 2 else delta_row
        else:
            delta_col = abs(to_pos[0] - from_pos[0])
            delta_row = abs(to_pos[1] - from_pos[1])

        target_piece = board[to_pos]
        if target_piece is None or target_piece.is_white != self.is_white:
            return (delta_col, delta_row) in [(2, 1), (1, 2)]

        return False


class Pawn(Piece):
    def __init__(self, is_white: bool):
        super().__init__(is_white, "p", 1)

    def can_move(self, board: 'Board', from_pos: List[int], to_pos: List[int]) -> bool:
        """ Проверка возможного хода пешки """
        direction = -1 if self.is_white else 1
        start_row = 6 if self.is_white else 1

        delta_col = to_pos[0] - from_pos[0]
        delta_row = to_pos[1] - from_pos[1]
        if delta_col == 0:  # Пешка движется вперед
            # Пешка двигается на одну клетку
            if delta_row == direction and board[to_pos] is None:
                return True
            # Пешка двигается на две клетки с начальной позиции
            if delta_row == 2 * direction and from_pos[1] == start_row and board[to_pos] is None:
                return True
            # Пешка двигается через край
            if from_pos[1] in (7, 0) and board.is_looped and board[to_pos] is None and abs(delta_row) == 7:
                return True
        elif (abs(delta_col) == 1 or abs(delta_col) == 7 and board.is_looped) and delta_row == direction:
            # Захват фигуры
            return isinstance(board[to_pos], Piece) and board[to_pos].is_white != self.is_white

        return False


def parse_position(pos: str) -> Optional[List[int]]:
    """ Преобразование шахматной нотации в индексы массива """
    if len(pos) != 2 or pos[0] not in "abcdefgh" or pos[1] not in "12345678":
        return None
    col = ord(pos[0]) - ord('a')
    row = 8 - int(pos[1])
    return [col, row]


def main():
    board = Board()
    move_count = 0
    while True:
        sys_output = f"white {move_count // 2 + 1}:\n" if board.is_white_turn else f"black {move_count // 2 + 1}:\n"
        user_input = input(sys_output)
        if user_input == "exit":
            sys.exit()
        elif user_input == "draw":
            print(board)
            continue
        elif user_input == "balance white":
            print(board.balance())
            continue
        elif user_input == "balance black":
            print(-board.balance())
            continue
        elif user_input == "dump":
            print(board.to_json())
            continue
        elif user_input == "make_looped":
            board.make_looped()
            continue
        elif user_input == "make_default":
            board.make_default()
            continue

        move = user_input.split('-')
        if len(move) != 2:
            print("Error. Type: Wrong input format.")
            continue

        from_pos = parse_position(move[0])
        to_pos = parse_position(move[1])
        if not from_pos or not to_pos:
            print("Error. Type: Wrong input format.")
            continue

        try:
            board.move_piece(from_pos, to_pos)
        except ValueError as e:
            print(f"Error. Type: {e}")
            continue

        move_count += 1
        board.is_white_turn = not board.is_white_turn


if __name__ == "__main__":
    main()
