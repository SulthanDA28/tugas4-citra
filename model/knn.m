function label = knn(img)
% extract features (vector representation) dari image
    function features = getFeatures(img)
        if size(img, 3) ~= 3
            img = repmat(im2uint8(img), [1, 1, 3]); 
        end
        % convert to grayscale
        img_gray = rgb2gray(img);
        % resize to 64x64
        img_gray_64 = imresize(img_gray, [64, 64]);
        % convert to feature vector
        g = graycoprops(graycomatrix(img_gray_64));
        features = [g.Contrast, g.Correlation, g.Energy, g.Homogeneity];
        % add other features
        % firstly, a histogram of the image (colored)
        img_128 = imresize(img, [128, 128]);
        img_128 = double(img_128);
        % divide into 4x4 blocks and split channels
        img_128 = mat2cell(img_128, [64 64], [64 64], 3);
        % calculate histogram for each block
        hist = [];
        for i = 1:2 
            for j = 1:2
                % reduce histogram to 4 bins
                binned_hist = histcounts(img_128{i, j}, 4);
                hist = [hist, binned_hist'];
            end
        end
        % flatten
        hist = hist(:)';
        % add to features
        features = [features, hist];
        % next feature are hough lines
        img_edge = edge(img_gray, 'canny');
        [H, T, R] = hough(img_edge);
        P = houghpeaks(H, 5);
        lines = houghlines(img_edge, T, R, P);
        % get the angles
        angles = [];
        rhos = [];
        for i = 1:length(lines)
            angles = [angles, lines(i).theta];
            rhos = [rhos, lines(i).rho];
        end
        % get the mean and std
        mean_angle = mean(angles);
        std_angle = std(angles);
        mean_rho = mean(rhos);
        std_rho = std(rhos);
        % add to features
        features = [features, mean_angle, std_angle, mean_rho, std_rho, length(lines)];
        % normalize features
        features = features ./ max(features);
    end
% Train KNN if trained matrix files don't exist
% read from the train folder, each image will be extracted to a feature vector
% then store all feature vectors in a matrix
% store the labels in a vector
dataset_path = 'training_image/';
if ~exist('train_matrix.mat', 'file') || ~exist('train_labels.mat', 'file')
    train_matrix = [];
    train_labels = {};
    % read all files
    file_names = dir(dataset_path);
    % iterate over all files
    for i = 1:length(file_names)
        disp('Training KNN... (' + string(i) + '/' + string(length(file_names)) + ')')
        % get the file name
        file_name = file_names(i).name;
        % check if it is a valid file
        if length(file_name) > 4 && (strcmp(file_name(end-3:end), '.jpg') || strcmp(file_name(end-3:end), '.png') || strcmp(file_name(end-4:end), '.jpeg')) && ~contains(file_name, 'Ambulance')
            % read the image
            img = imread(strcat(dataset_path, file_name));
            % get the features
            feats = getFeatures(img);
            % append to train matrix
            train_matrix = [train_matrix; feats];
            % read the label
            % label can be [car, bus, truck, ambulance]
            % its taken from file name, but can be in various formats
            if contains(file_name, 'car')
                label = 'car';
            elseif contains(file_name, 'BUS')
                label = 'bus';
            elseif contains(file_name, 'Truck')
                label = 'truck';
            else
                label = '';
                disp(file_name);
            end
            % append to train labels
            train_labels = [train_labels; {label}];
            % close files
            close all;
        end
        clc; % clear console
    end
    
    save('train_matrix.mat', 'train_matrix');
    save('train_labels.mat', 'train_labels');
end

% load the train matrix and labels
train_matrix = load('train_matrix.mat');
train_labels = load('train_labels.mat');
model = fitcknn(train_matrix.train_matrix, train_labels.train_labels);
% get features from image
feats = getFeatures(img);
% predict
label = predict(model, feats);
end