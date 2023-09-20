import cv2
import numpy as np


class ProductDetection:

    def videoDetector(self, filename, EDT):
        tracker = EDT
        response = ""
        net = cv2.dnn.readNet("C:\\Users\\HASSAN MOIN\\PycharmProjects\\SCSystemAPIv2\\SCSystemAPI\\api\\APIManagement\\Assets\\yolov3_custom_last20.weights", "C:\\Users\\HASSAN MOIN\\PycharmProjects\\SCSystemAPIv2\\SCSystemAPI\\api\\APIManagement\\Assets\\yolov3_customlatest.cfg")
        classes = []

        with open("C:\\Users\\HASSAN MOIN\\PycharmProjects\\SCSystemAPIv2\\SCSystemAPI\\api\\APIManagement\\Assets\\classeslatest.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        #
        img = cv2.imread(filename)
        try:
            height, width, channel = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)


            net.setInput(blob)
            outs = net.forward(outputlayers)

            boxes = []
            confidences = []
            class_ids = []
            detections = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # cv2.circle(img,(center_x,center_y), 10 , (0,255,0), 2)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    detections.append(([x, y, w, h, class_ids[i]]))

            boxes_ids = tracker.update(detections)
            for box_id in boxes_ids:
                x, y, w, h, ob_id, cid = box_id
                label = str(classes[cid])
                response += "/" + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "/" + label + "-" + str(
                    cid) + "_" + str(ob_id)
            print(response)
            return response
        except:
            return "None"