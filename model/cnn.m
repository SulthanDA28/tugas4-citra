function [imageborder,labelresult] = cnn(image)
    addpath model/pretrained-yolo-v4/
    addpath model/pretrained-yolo-v4/src/
    addpath model/pretrained-yolo-v4/models/

    model = helper.downloadPretrainedYOLOv4('YOLOv4-coco');
    net = model.net;
    classname = helper.getCOCOClassNames;
    anchor = helper.getAnchors('YOLOv4-coco');
    

    [boundingbox,score,label] = detectYOLOv4(net, image, anchor, classname, 'auto');
    class = {'bus', 'car', 'truck'};
    for j=1:length(label)
        if(ismember(label(j),class))
            continue
        else
            label(j) = 'other';
        end
    end
    
    imageborder = insertObjectAnnotation(image,"rectangle",boundingbox,label);
    
    maxscore = score(1);
    indexmax = 1;
    for i=1:length(score)
        if(maxscore < score(i))
            maxscore = score(i);
            indexmax = i;
        end
    end
    labelresult = label(indexmax);
end