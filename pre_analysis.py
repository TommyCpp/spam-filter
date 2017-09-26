import os

DATA_SOURCE = "./data/lingspam_public/bare/part%s/"

target = "./data/source/"


def process(filepath, filename: str, target):
    if filename.find("spms") == -1:
        # if it is not a spam
        target_file = target + "/msg.txt"
    else:
        target_file = target + "/spam.txt"

    with open(filepath + filename) as f:
        with open(target_file, "a") as target_f:
            lines = f.readlines()
            subject = lines[0].replace("Subject:", "")
            raw_str = lines[2]
            target_f.write(subject + raw_str)


for i in range(1, 10):
    data_source = DATA_SOURCE % i
    for filename in os.listdir(data_source):
        process(data_source, filename, target)
