from ast import literal_eval
import matplotlib.pyplot as plt

with open("successes.txt","r") as file:
    ll=list(map(literal_eval,file.readlines()))

t=[0]*len(ll[0])
for i in range(len(ll[0])):
    for l in ll:
        t[i]+=l[i]
print("total:",t)
print(len(ll),"contributions")

exponents=list(range(1,len(t)+1))
for i in range(len(exponents)):
    exponents[i]/=len(t)

plt.switch_backend('agg')
fig=plt.figure()
plt.plot(exponents,list(map(lambda x:x/(1000*len(ll)),t)))
fig.savefig('damascusvsexpon.pdf')
