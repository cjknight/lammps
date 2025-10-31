#!/usr/bin/env python3
import sys
import math

def main():

    printHelp = False

    if (len(sys.argv) < 3 or sys.argv[1] == '-h' or sys.argv[1] == '-?'):
        printHelp = True

    # Help message
    if (printHelp):
        print("")
        print("Provide 3-D decomposition factors")
        print("Use as")
        print("    ./decompose_global_problem_xyz.py <global_size> [optional bias direction and multiplicative factor]")
        print("    global_size: Integer denoting global parameter that needs to be factored along three directions")
        print("    optional bias direction: Integer denoting direction along which to prioritize largest factor")
        print("                             0 for X, 1 for Y, 2 for Z")
        print("    multiplicative factor: scale factors")
        print("    Example: ./decompose_global_problem_xyz.py 216 0 1")
        print("")
        sys.exit(0)

    global_size = int(sys.argv[1])

    bias = 0

    if ( len(sys.argv) > 2 ):
        bias = int(sys.argv[2])

    if (bias > 2):
        print("ERROR: bias direction must be <=2");
        sys.exit(1)

    mult = 1

    if ( len(sys.argv) > 3 ):
        mult = int(sys.argv[3])

    v = [0, 0, 0];

    v[0] = int(math.floor(global_size**(1/3)))

    while ( global_size%v[0] != 0 ):
        v[0]=v[0]+1

    v[1] = int(math.floor((global_size/v[0])**(1/2)))

    while ( (global_size/v[0])%v[1] != 0 ):
        v[1]=v[1]+1

    v[2] = int(global_size / v[0] / v[1])

    v.sort(reverse=False)

    # Give precendence to bias direction
    temp = v[2]
    v[2] = v[bias]
    v[bias] = temp

    #sizestr = str(int(v[0])) + "x" + str(int(v[1])) + "x" + str(int(v[2]))
    sizestr = "-var x " + str(int(v[0] * mult)) + " -var y " + str(int(v[1] * mult)) + " -var z " + str(int(v[2] * mult))
    print(sizestr)

if __name__ == "__main__":
    main()
