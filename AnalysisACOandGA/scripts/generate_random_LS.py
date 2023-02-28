import random

dim = [
    ["AKT", "REF"],
    ["INT", "SNS"],
    ["VIS", "VRB"],
    ["GLO", "SEQ"]
]

value = [1, 3, 5, 7, 9, 11]

ls_list = []

for _ in range(25):
    ls = {}
    for ele in dim:
        rand_dim = random.choice(ele)
        rand_val = random.choice(value)
        ls[rand_dim] = rand_val
    print(ls)
    ls_list.append(ls)

print(ls_list)