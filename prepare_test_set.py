import os

with open("./sc5-2013-Mar-Apr-Test-20130412/ground_truth.txt") as f:
    data = map(lambda x: x.strip().split(";"), f.readlines())

classes = os.listdir("./sc5")
classes_map = {x.replace(":","").replace(" ","").lower(): x for x in classes}
classes_included = {x: False for x in classes}

os.system("mkdir -p ./sc5_test")

skipped = []

for x,y in data:
    z = y.replace(":","").replace(" ","").lower()
    if z == "snapshotacqua":
        z = "water"
    if z not in classes_map.keys():
        if y not in skipped:
            skipped.append(y)
        continue
    classes_included[classes_map[z]] = True
    os.system("mkdir -p './sc5_test/%s'" % classes_map[z])
    os.system("mv './sc5-2013-Mar-Apr-Test-20130412/" + x + "' './sc5_test/" + classes_map[z] + "/'")

    
for x in skipped:
    print("skipped " + x)

for x in classes_included:
    if not classes_included[x]:
        print("including empty " + x)
        os.system("mkdir -p './sc5_test/%s'" % x)