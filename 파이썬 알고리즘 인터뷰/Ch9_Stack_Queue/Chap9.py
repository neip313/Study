from Ch9_Stack_Queue import Stack
from Ch9_Stack_Queue import Queue

sta = Stack.Solution()
qms = Queue.MyStack()
qmq = Queue.MyQueue()
qmc = Queue.MyCircularQueue(5)  # 크기를 5로 지정

# Question 20
if __name__ == '__main__':
  print(sta.isValid("()[]{}"))
  print(sta.isValid(()))
  print(sta.isValid({}))
  print(sta.isValid([]))

# Question 21
if __name__ == '__main__':
  print(sta.removeDuplicateLetters1("bcabc"))
  print(sta.removeDuplicateLetters1("cbacdcbc"))

# Question 22
if __name__ == '__main__':
  print(sta.dailyTemperatures([73, 74, 75, 71, 69, 72, 76, 73]))

# Question 23
if __name__ == '__main__':
  print(qms.push(1))
  print(qms.push(2))
  print(qms.top())
  print(qms.pop())
  print(qms.empty())

# Question 24
if __name__ == '__main__':
  print(qmq.push(1))
  print(qmq.push(2))
  print(qmq.peek())
  print(qmq.pop())
  print(qmq.empty())

# Question 25
if __name__ == '__main__':
  print(qmc.enQueue(10))
  print(qmc.enQueue(20))
  print(qmc.enQueue(30))
  print(qmc.enQueue(40))
  print(qmc.Rear())
  print(qmc.isFull())
  print(qmc.deQueue())
  print(qmc.deQueue())
  print(qmc.enQueue(50))
  print(qmc.enQueue(60))
  print(qmc.Rear())
  print(qmc.Front())