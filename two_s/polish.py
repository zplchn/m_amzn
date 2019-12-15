from abc import abstractmethod, ABC
from typing import List


class BaseExpr(ABC):

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def __init__(self):
        pass


class Operand(BaseExpr):
    def __init__(self, val):
        super().__init__()
        self._val = float(val)

    def eval(self):
        return self._val


class BinaryOperator(BaseExpr):
    def __init__(self, l_expr, r_expr):
        super().__init__()
        self._l = l_expr
        self._r = r_expr


class AddExpr(BinaryOperator):
    def eval(self):
        return self._l.eval() + self._r.eval()


class SubExpr(BinaryOperator):
    def eval(self):
        return self._l.eval() - self._r.eval()


class MulExpr(BinaryOperator):
    def eval(self):
        return self._l.eval() * self._r.eval()


class DivExpr(BinaryOperator):
    def eval(self):
        return self._l.eval() / self._r.eval()


class ExprFactory:
    @staticmethod
    def parse(tokens: List[str]) -> BaseExpr:
        st = []
        op = {
            '+': lambda l, r: AddExpr(l, r),
            '-': lambda l, r: SubExpr(l, r),
            '*': lambda l, r: MulExpr(l, r),
            '/': lambda l, r: DivExpr(l, r),
        }

        for t in tokens:
            if t in op:
                r, l = st.pop(), st.pop()
                st.append(op[t](l, r))
            else:
                st.append(Operand(t))
        return st.pop()


input = ['1', '2', '*', '9', '/']
root = ExprFactory.parse(input)
res = root.eval()
print(res)


