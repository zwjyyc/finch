class Node:
    def __init__(self, val, next, previous):
        self.val = val
        self.next = next
        self.previous = previous

class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, val):
        new_node = Node(val, None, None)
        if self.head is None:
            self.head = new_node
        else:
            new_node.next = self.head
            self.head.previous = new_node
            self.head = new_node
    
    def delete(self):
        self.head = self.head.next
        self.head.previous = None

    def display(self):
        n = self.head
        while n is not None:
            print(n.val)
            n = n.next

if __name__ == '__main__':
    ls = LinkedList()
    ls.add('a')
    ls.add('b')
    ls.add('c')
    ls.add('d')
    ls.display()
    print()
    ls.delete()
    ls.display()
