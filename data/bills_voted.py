# which bills has a person voted?

person_aim = 750

voteOfPersonT = [] 
voteOfPersonF = []

fin = open('Person_Bill_test', 'r')
lines = fin.readlines()
for line in lines:
	person, bill, vote = line.split('\t')
	person = int(person)
	if person == person_aim:
		bill = int(bill)
		vote = int(vote.split('\n')[0])
		if vote == 1:
			voteOfPersonT.append(bill)
		if vote == -1:
			voteOfPersonF.append(bill)
fin.close()

print len(voteOfPersonT)
print voteOfPersonT

print len(voteOfPersonF)
print voteOfPersonF
