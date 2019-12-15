from types import *

class Rope(object):

    #Accepts a string or list of strings and forms it into a rope.
    #This is the basic implementation which forms a rope with the input data.
    #Create an object and pass the string or list as the argument to the constructor to form a rope.
    #Ex - s= Rope('abcxyz') or s = Rope(['abc','def','ghi'])
    #Simply use print on the instance of this class in order to print the entire string. Ex - print(s)

    def __init__(self, data=''):
	#check if the input data is string or not
        if isinstance(data, list):
            # if len(data) == 0:
            #     self.__init__()
            # elif len(data) == 1:
            #     self.__init__(data[0])
            # else:
            #     # Round-up division (to match rope arithmetic associativity)
            #     idiv = len(data) // 2 + (len(data) % 2 > 0)
            #     self.left = Rope(data[:idiv])
            #     self.right = Rope(data[idiv:])
            #     self.data = ''
            #     self.length = self.left.length + self.right.length
        elif isinstance(data, str):
            self.left = None
            self.right = None
            self.data = data
            self.length = len(data)
        else:
            raise TypeError('Kindly use strings for the purpose')
        # Word iteration
        self.current = self

    #checks if the tree is balanced
    def __eq__(self, other):
        if (self.left and self.right) and (other.left and other.right):
            return self.left == other.left and self.right == other.right
        elif (self.left and self.right) or (other.left and other.right):
            return False
        else:
            return self.data == other.data


    #concatenation of strings
    #forms a new rope with other string and joins the 2 ropes
    #into a new rope
    def __add__(self, other):
        #ensure that the string being concatenated is of type string
        if isinstance(other, str):
            other = Rope(other)
        r = Rope()
        r.left = self
        r.right = other
        r.length = self.length + other.length
        r.current = self
        return r

    #Fetch the length of string in the specified rope
    def __len__(self):
        if self.left and self.right:
            return len(self.left.data) + len(self.right.data)
        else:
            return(len(self.data))

    #fetch the word present at the specified index
    def __getitem__(self, index):
        #ensure the index specified is an integer
        if isinstance(index, int):
            if self.left and self.right:
                if index < -self.right.length:
                    subindex = index + self.right.length
                elif index >= self.left.length:
                    subindex = index - self.left.length
                else:
                    subindex = index

                if index < -self.right.length or 0 <= index < self.left.length:
                    return self.left[subindex]
                else:
                    return self.right[subindex]
            else:
                return Rope(self.data[index])

            elif isinstance(index, slice):
                if self.left and self.right:
                    start = index.start
                    if index.start is None:
                        if index.step is None or index.step > 0:
                            head = self.left
                        else:
                            head = self.right
                    elif (index.start < -self.right.length or
                            0 <= index.start < self.left.length):
                        head = self.left
                        if index.start and index.start < -self.right.length:
                            start += self.right.length
                    else:
                        head = self.right
                        if index.start and index.start >= self.left.length:
                            start -= self.left.length
                    stop = index.stop
                    if index.step is None or index.step > 0:
                        if (index.stop is None or
                            -self.right.length <= index.stop < 0 or
                            index.stop > self.left.length):
                            tail = self.right
                            if index.stop and index.stop > self.left.length:
                                stop -= self.left.length
                        else:
                            if head == self.right:
                                tail = self.right
                                stop = 0
                            else:
                                tail = self.left
                                if index.stop < -self.right.length:
                                    stop += self.right.length
                    else:
                        if (index.stop is None or
                                index.stop < (-self.right.length - 1) or
                                0 <= index.stop < self.left.length):
                            tail = self.left
                            if index.stop and index.stop < (-self.right.length - 1):
                                stop += self.right.length
                        else:
                            if head == self.left:
                                tail = self.left
                                stop = -1   # Or self.left.length - 1 ?
                            else:
                                tail = self.right
                                if index.stop >= self.left.length:
                                    stop -= self.left.length

                    # Construct the rope as a binary tree
                    if head == tail:
                        return head[start:stop:index.step]
                    else:
                        if not index.step:
                            offset = None
                        elif index.step > 0:
                            if start is None:
                                delta = -head.length
                            elif start >= 0:
                                delta = start - head.length
                            else:
                                delta = max(index.start, -self.length) + tail.length

                            offset = delta % index.step
                            if offset == 0:
                                offset = None
                        else:
                            if start is None:
                                offset = index.step + (head.length - 1) % (-index.step)
                            elif start >= 0:
                                offset = index.step + min(start, head.length - 1) % (-index.step)
                            else:
                                offset = index.step + (start + head.length) % (-index.step)

                        if not tail[offset:stop:index.step]:
                            return head[start::index.step]
                        else:
                            return head[start::index.step] + tail[offset:stop:index.step]
                else:
                    return Rope(self.data[index])

        #Report the characters present inside the rope
        def __repr__(self):
            if self.left and self.right:
                return '{}{} + {}{}'.format('(' if self.left else '',
                                        self.left.__repr__(),
                                        self.right.__repr__(),
                                        ')' if self.right else '')
            else:
                return "Rope('{}')".format(self.data)

        #Check if the current node is a leaf node and print its value
        #otherwise iterate down a step on both sides
        def __str__(self):
            if self.left and self.right:
                return self.left.__str__() + self.right.__str__()
            else:
                return self.data

        #The iterator
        def __iter__(self):
            return self

        #Iteration down the tree
        def __next__(self):
            if self.current:
                if self.left and self.right:
                    try:
                        return next(self.left)
                    except StopIteration:
                        self.current = self.right
                    return next(self.right)
                else:
                    self.current = None
                    return self.data
            else:
                raise StopIteration

        def next(self):
            return self.__next__()

firstRope = Rope('hello_my_name_is')
secondRope = Rope('_rope_data_structure')

#concatenate these strings
fullRope = firstRope.__add__(secondRope)
print(fullRope)
#Gives result hello_my_name_is_rope_data_structure

#Show the character at specified index
print(fullRope.__getitem__(14))
#Displays character "i" which is at index 14
