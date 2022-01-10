import math

def getLength(point1, point2):
    length = 0
    for i in range(len(point1)):
        length += (point1[i] - point2[i]) ** 2

    return math.sqrt(length)

def norm(vector):
    value = 0
    for v in vector:
        value += math.pow(v, 2)

    return math.sqrt(value)

def dot(vector1, vector2):
    value = 0
    for i in range(len(vector1)):
        value += vector1[i]*vector2[i]

    return value

def getAngle(point1, point2, point3):
    v1 = [point2[0] - point1[0], point2[1] - point1[1]]
    v2 = [point2[0] - point3[0], point2[1] - point3[1]]
    return math.acos( dot(v1, v2) / (norm(v1)*norm(v2)) )

line = [[0,1], [4,3]]
point1 = [0,-2]
point2 = [3,-1]

a = getLength(line[0], line[1])
b = getLength(line[0], point1)
c = getLength(line[1], point1)

A = getAngle(line[0], point1, line[1])
B = getAngle(point1, line[0], line[1])
C = getAngle(point1, line[1], line[0])

_a = math.cos(B)*b + math.cos(C)*c

base = a - math.cos(B)*b
hypot = c

print(base, hypot)
length = math.sqrt(math.pow(hypot, 2) - math.pow(base, 2))
print(length)