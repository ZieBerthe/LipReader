import os
def test():
    print("test")
    print(os.listdir("."))
    #print what is inside the data directory
    from paths import DATA_ROOT
    print(os.listdir(DATA_ROOT))
if __name__ == "__main__":
    test()