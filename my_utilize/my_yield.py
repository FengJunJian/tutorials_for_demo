
def generator():
    for i in range(5):
        print('next',i)
        yield i


fun=generator()
fun.__next__()