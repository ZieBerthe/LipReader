import os
def test():
    print("test")
    print(os.listdir("."))
    #print what is inside the path of the data/grid_corpus directory
    print(os.listdir("../../../../../../Data/grid_corpus/"))
if __name__ == "__main__":
    test()