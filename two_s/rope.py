class Rope:

    def __init__(self, data=''):
        self.left = self.right = None
        self.data = data
        self.length = len(data)

        self.current = self

    def __eq__(self, other):
        # bool if self and other are
        pass

    def __len__(self):



