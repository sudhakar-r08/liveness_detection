from pip._vendor.distlib.compat import raw_input


def main():
    # Write code here
    count = int(input())
    alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
                 "U", "V", "W", "X", "Y", "Z"]
    times = 1
    for i in range(0, count):
        for j in range(count - (i + 1)):
            print(" ", end=" ")

        for k in range(0, times):
            if (k != (times-1)):
                print(alphabets[times - (k+1)], end=" ")
            else:
                print(alphabets[times - (k+1)], end="")

        times += 2
        if i != count - 1:
            print()
main()


