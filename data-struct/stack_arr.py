class StackArr:
    def __init__(self, size):
        self.size = size
        self.stack_arr = [0] * size
        self.top = -1

    def is_full(self):
        return self.top == (self.size - 1)

    def is_empty(self):
        return self.top == -1

    def push(self, new_item):
        if self.is_full():
            print("Stack is Full")
            return None
        self.top += 1
        self.stack_arr[self.top] = new_item

    def pop(self):
        if self.is_empty():
            print("Stack is Empty")
            return None
        item = self.stack_arr[self.top]
        self.top -= 1
        return item

if __name__ == '__main__':
    stack = StackArr(5)
    stack.push(11)
    stack.push(12)
    stack.push(13)
    stack.push(14)
    stack.push(15)
    stack.push(16)
    while not stack.is_empty():
        print(stack.pop())
