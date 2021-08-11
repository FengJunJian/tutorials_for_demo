class Cat(object):
    def __init__(self):
        self.name = "jn"

    def __getattr__(self, item):
        return "tm"


cat = Cat()
print(cat.name)
print(getattr(cat, 'name'))
print("*" * 20)
print(cat.age)
print(getattr(cat, 'age'))