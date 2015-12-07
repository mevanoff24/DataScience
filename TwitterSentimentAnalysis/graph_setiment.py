import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
style.use('fivethirtyeight')


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


# live updating graph
def animate(i):
    # read in pos / neg 
    graph_data = open('twitter-out.txt','r').read()
    lines = graph_data.split('\n')

    xs = []
    ys = []

    x = 0
    y = 0

    # plot only last 200 tweets
    # if 'pos' + 1 
    # if 'neg' - 1
    for l in lines[-200:]:
        x += 1
        if 'pos' in l:
            y += 1
        elif 'neg' in l:
            y -= 1

        xs.append(x)
        ys.append(y)


    ax1.clear()
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval = 1000)
plt.show()


