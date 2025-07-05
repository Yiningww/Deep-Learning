class Person:
    def __init__(self, name):
        self.name = name

    def __call__(self, gender):
        print("__call__" + " Hello " + self.name + gender)

    def hello(self, gender):
        print("hello " + self.name + gender)


person = Person("wyn")
person("female")
person.hello("female")



class Parent:
    def __init__(self):
        print("Parent init")
    def p(self, name):
        self.name = name
        print("my name is wyn")
        return 0

class Child(Parent):
    def __init__(self, name):
        # super(Child, self).__init__()  # 调用父类的初始化方法
        super().__init__()  # 调用父类的初始化方法
        self.name = name
        print(f"Child init: {self.p(self.name)}")

child = Child("wyn")
