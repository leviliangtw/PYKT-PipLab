x = 1
for i in range(0, 1000000):
    x += 0.000001
x -= 1
print(x)

x = 10000000
for i in range(0, 1000000):
    x += 10.000001
x -= 10000000
print(x)
