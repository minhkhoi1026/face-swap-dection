import os

fake = os.listdir("dataset_v1/videos/roop/fake")

val = ["mnhduong", "nqktuyen", "ntthau", "sedanh"]
test = ["nhlong", "ndtri", "tnhnhu", "mekhoi"]

print(len(fake))

d = dict()

valid = []
testid = []

for file in fake:
    name = file.split(".")[0]
    id, char = name.split("_")

    if char in val:
        valid.append(id)
    if char in test:
        testid.append(id)
    
#     if char in d.keys():
#         d[char] += 1
#     else:
#         d[char] = 1
    
# for key in d.keys():
#     print(key, d[key])


print(valid)
print(testid)

