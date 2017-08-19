import time
import lehar

# Try to hide the cursor 

data = [5,4,3,2,1,3,5,1,2]


for i in range(len(data)):
    print("     ",end="")
    print(lehar.draw(data,color='cyan'),end="")
    maxi = data[0]
    max_index = 0
    for j in range(len(data)-i):
        if data[j] > maxi:
            maxi = data[j]
            max_index = j
    
    data[max_index] = data[j]
    data[j] = maxi
    time.sleep(1)
    print("\r",end="")

print(data)