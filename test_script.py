import argparse


parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--id', type=int, default=0, help='ID of the job')

args = parser.parse_args()
id = args.id


with open(f"job_{id}.txt", "w") as file:
    file.write(f"This is text for job {id}")


while True:
    pass