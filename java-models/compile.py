import os
import time
import sys


def quote(in_path):
    return '"' + in_path + '"'


jar_path = './lib/opencsv-3.9.jar'
command1 = 'javac ' + '-cp ' + quote(jar_path) + ' *.java'


if __name__ == '__main__':
    t1 = time.time()
    os.system(command1)
    t2 = time.time()
    print("compile time: %.2f" % (t2-t1))
    if len(sys.argv) > 1:
        command2 = 'java ' + '-cp ' + quote(jar_path+';.') + ' ' + sys.argv[1]
        os.system(command2)
        print("run time: %.2f" % (time.time()-t2))
