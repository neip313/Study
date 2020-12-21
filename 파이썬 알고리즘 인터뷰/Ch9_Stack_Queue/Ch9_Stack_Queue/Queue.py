import collections

class MyStack:
    def __init__(self):
        self.q = collections.deque()

    def push(self, x):
        self.q.append(x)
        # 요소 삽입 후 맨 앞에 두는 상태로 재정렬
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self):
        return self.q.popleft()

    def top(self):
        return self.q[0]

    def empty(self):
        return len(self.q) == 0


class MyQueue:
    def __init__(self):
        self.input = []
        self.output = []

    def push(self, x):
        self.input.append(x)

    def pop(self):
        self.peek()
        return self.output.pop()

    def peek(self):
        # output이 없으면 모두 재입력
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())
        return self.output[-1]

    def empty(self):
        return self.input == [] and self.output == []


class MyCircularQueue:
    def __init__(self, k: int):
        self.q = [None] * k
        self.maxlen = k
        self.p1 = 0
        self.p2 = 0

    # enQueue(): 리어 포인터 이동
    def enQueue(self, value: int) -> bool:
        if self.q[self.p2] is None:
            self.q[self.p2] = value
            self.p2 = (self.p2 + 1) % self.maxlen
            return True
        else:
            return False

    # deQueue(): 프론트 포인터 이동
    def deQueue(self) -> bool:
        if self.q[self.p1] is None:
            return False
        else:
            self.q[self.p1] = None
            self.p1 = (self.p1 + 1) % self.maxlen
            return True

    def Front(self) -> int:
        return -1 if self.q[self.p1] is None else self.q[self.p1]

    def Rear(self) -> int:
        return -1 if self.q[self.p2 - 1] is None else self.q[self.p2 - 1]

    def isEmpty(self) -> bool:
        return self.p1 == self.p2 and self.q[self.p1] is None

    def isFull(self) -> bool:
        return self.p1 == self.p2 and self.q[self.p1] is not None