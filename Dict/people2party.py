p2p = dict()

fin = open('Dict_Person_House')
lines = fin.readlines()
for line in lines:
	uid, name, party, x, y = line.split('\t')
	p2p[uid] = party
fin.close()

fin = open('Dict_Person_Senate')
lines = fin.readlines()
for line in lines:
	uid, fn, ln, party, state = line.split('\t')
	p2p[uid] = party
fin.close()

fout = open('Person_party', 'w')
for u in p2p:
	newline = u + '\t' + p2p[u] + '\n'
	fout.write(newline)
fout.close()
