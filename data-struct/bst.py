class Node:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def add(self, new_node, start_node):
        if self.root is None:
            self.root = new_node
            return None
        if new_node.val > start_node.val:
            if start_node.right is None:
                start_node.right = new_node
            self.add(new_node, start_node.right)
        if new_node.val < start_node.val:
            if start_node.left is None:
                start_node.left = new_node
            self.add(new_node, start_node.left)
    
    def search(self, value, start_node):
        if start_node is None:
            print("Node is Not Found")
            return None
        if start_node.val == value:
            print("Node is Found")
            return None
        if value > start_node.val:
            self.search(value, start_node.right)
        if value < start_node.val:
            self.search(value, start_node.left)

if __name__ == '__main__':
    bst = BST()

    bst.add(Node(10, None, None), bst.root)
    bst.add(Node(12, None, None), bst.root)
    bst.add(Node(11, None, None), bst.root)
    bst.add(Node(13, None, None), bst.root)
    bst.add(Node(6, None, None), bst.root)
    
    bst.search(11, bst.root)
    bst.search(21, bst.root)
