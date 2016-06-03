import cv2

class Visualizer:
    def __init__(self):
        var = 1
        # do something

    def visualizeProbabilityMap(self, map):
        #vis2 = cv2.cvtColor(map, cv2.COLOR_GRAY2BGR)

        cv2.imshow('dst_rt', map)
        cv2.waitKey(0)
        cv2.destroyAllWindows()