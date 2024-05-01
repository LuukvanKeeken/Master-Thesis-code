import argparse


parser = argparse.ArgumentParser(description=' ')
parser.add_argument('--id', type=int, default=0, help='ID of the job')
parser.add_argument('--group', type=int, default=0, help='ID of the group')

args = parser.parse_args()
id = args.id
group = args.group


with open(f"Master_Thesis_Code/LTC_A2C/all_jobs_{group}/job_{id}.txt", "w") as file:
    file.write(f"This is text for job {id} in group {group}")


while True:
    pass