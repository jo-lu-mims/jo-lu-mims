# simple data visualization
import matplotlib.pyplot as plt
plt.plot([1,2,3,4])
plt.show()


# simple data visualization
import matplotlib.pyplot as plt
plt.plot([1,2,3,4], [1,4,9,16], 'ro-')
plt.axis([0, 5, 0, 17])
plt.show()


# muliple plots per figure
import matplotlib.pyplot as plt
plt.subplot(221) 
plt.plot([1, 2, 3, 4], [1, 2, 3, 4], 'bo-')
plt.subplot(222) 
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'bo-')
plt.subplot(223)
plt.plot([1, 2, 3, 4], [1, 8, 27, 64], 'bo-')
plt.subplot(224)
plt.plot([1, 2, 3, 4], [1, 2, 3, 4], 'bo-', 
         [1, 2, 3, 4], [1, 8, 27, 64], 'ro-' )
plt.show()


# numpy for data processing
import numpy as np
t = np.arange(0.0, 1.01, 0.01)
t


# numpy for data processing
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 1.01, 0.01)
plt.plot(2*np.pi*t, np.sin(2*np.pi*t) )
plt.show()


# numpy for data processing
import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0.0, 1.01, 0.01)
plt.plot(2*np.pi*t, np.sin(2*np.pi*t) )
plt.annotate('zero crossing', xy=(np.pi, 0), xytext=(4, 0.5),
            arrowprops=dict(facecolor='black'))
plt.grid(True)
plt.show()


# numpy for data processing
import matplotlib.pyplot as plt
import numpy as np

def decay(t):
    return np.exp(-t) * np.cos(2*np.pi*t) # damped sinusoid  

t = np.arange(0.0, 4.01, 0.01)
plt.plot(t, decay(t))
plt.xlabel('time')
plt.ylabel('energy')
plt.title('damped sinusoid')
plt.show()


# data histogram
import numpy as np
import matplotlib.pyplot as plt

x = np.random.randn(100000)
plt.hist(x, 20)
plt.show()


# ----- DRILL -----
# The csv file temperatures.csv consists of two columns [year, temperature] 
# corresponding to the change in global temperatures (C) relative to a baseline 
# from 1880 to 2016. Read this csv file creating two lists consisting of the 
# year and temperature. Plot the change in temperature as a function of year 
# and label the axis accordingly. [NOTE: data will be read in as a string]

import csv
import matplotlib.pyplot as plt

with open( 'temperatures.csv' ) as csvfile:
    csvdata = csv.reader(csvfile)
    next(csvdata, None)  # skip the header
    T = []
    Y = []
    for row in csvdata:
        Y.append( int(row[0]) )
        T.append( float(row[1]) )

plt.plot(Y,T)
plt.xlabel('year')
plt.ylabel('temperature difference (C)')
plt.show()


# extrapolate changes in temperature
import csv
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

# ----- load year/temperature data -----
with open( 'temperatures.csv' ) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)  # skip the header
    T = []
    Y = []
    for row in reader:
        Y.append( int(row[0]) )
        T.append( float(row[1]) )

        
# ----- fit second-order polynomial -----

# build matrix of years [Y^2 Y 1]
a1 = np.matrix(Y)
a0 = np.matrix( np.ones(len(Y)) )
a2 = np.power(a1,2)
A  = np.concatenate( (a2,a1,a0), axis=0 )
A  = A.transpose()

# build vector of temperatures [T]
b  = np.transpose( np.matrix(T) )

# least-squares estimate of temperature as a function of year
m = inv(A.transpose()*A) * A.transpose() * b

# evaluate and plot model estimates
for y in range(1880,2100,10):
    t = m[0]*y*y + m[1]*y + m[2] # evaluate second-order model
    plt.plot(y,t,'r.')

# plot original data
plt.plot(Y,T)
plt.xlabel('year')
plt.ylabel('temperature difference (C)')
plt.show()




