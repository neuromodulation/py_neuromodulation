import multiprocessing
import keyboard

def getData(queue_raw):
    #while True:
    for num in range(1000):
        queue_raw.put(num)
        print("getData: put "+ str(num)+" in queue_raw")

def calcFeatures(queue_raw, queue_features):
    while not queue_raw.empty():
        data = queue_raw.get()
        queue_features.put(data**2)
        print("calcFeatures: put "+ str(data**2)+" in queue_features")

def sendFeatures(queue_features):
    while True:
        while not queue_features.empty():
            feature = queue_features.get()
            print("sendFeatures: put "+ str(feature)+" out")
        if keyboard.read_key() == "p":
            #print("You pressed p")
            break

if __name__ == "__main__":

    queue_raw = multiprocessing.Queue()
    queue_features = multiprocessing.Queue()

    processes = [

        multiprocessing.Process(target=getData, args=(queue_raw,)),
        multiprocessing.Process(target=calcFeatures, args=(queue_raw, queue_features,)),
        multiprocessing.Process(target=sendFeatures, args=(queue_features,))
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()
    