def printQuery(No1,No2,Query):
    No1 = list(No1)
    No2 = list(No2)
    Val =  []
    for i in range(len(Query)):
        QuerySet = [str(i) for i in Query[i].strip().split()]
        if QuerySet[0] == 'set_a':
            No1[len(No1) - int(QuerySet[1]) -  1] = QuerySet[2]
        elif QuerySet[0] == 'set_b':
            No2[len(No2) - int(QuerySet[1]) - 1] = QuerySet[2]
        else:
            added = binAdd(No1,No2)
            Q = bin(added)[2:]
            Val.append(Q[len(Q) - 1 - int(QuerySet[1])])

    print ''.join(Val)

def binAdd(No1,No2):
    return int(''.join(No1),2) + int(''.join(No2),2)
    
    
m = [int(i) for i in raw_input().strip().split()]
No1 = raw_input().strip()
No2 = raw_input().strip()
Query = []
for i in range(m[1]):
    Query.append((raw_input()))
                 
printQuery(No1,No2,Query)
