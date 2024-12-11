function imageborder = cnn(image)
    addpath model/pretrained-yolo-v4/
    addpath model/pretrained-yolo-v4/src/
    addpath model/pretrained-yolo-v4/models/

    model = helper.downloadPretrainedYOLOv4('YOLOv4-coco');
    net = model.net;
    classname = helper.getCOCOClassNames;
    anchor = helper.getAnchors('YOLOv4-coco');
    

    [boundingbox,score,label] = detectYOLOv4(net, image, anchor, classname, 'auto');

    imageborder = insertObjectAnnotation(image,"rectangle",boundingbox,label);
end