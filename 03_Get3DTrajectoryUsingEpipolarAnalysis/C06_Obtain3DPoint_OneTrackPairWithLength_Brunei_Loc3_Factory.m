clc; clear all;
close all;

Track_foler = 'G:\My Drive\Work\_GIT (2023~)\01 Mudskipper\20241216 3DTrackVideoCases\01_Videos\Brunei_Loc3';

matFiles = dir(fullfile(Track_foler,'Cropped_*.tsv'));

% Remove entries containing 'Lx'
hasLx = contains({matFiles.name}, 'Lx');
matFiles(hasLx) = [];

% Load 3D field data
load("D02_3DFieldData_Brunei_Loc3.mat");
load("D03_WaterEdge_Incheon_Loc3.mat");
Caminfo1=load('D01_camera_parameters_Cam1_Brunei_Loc3.mat');
Caminfo2=load('D01_camera_parameters_Cam2_Brunei_Loc3.mat');

for i = 1:length(matFiles)
if mod(i, round(length(matFiles) * 0.1)) == 0 || i == length(matFiles)
    fprintf('Progress: %d%%\n', round(100 * i / length(matFiles)));
end

Track_read = matFiles(i).name;

TrackLength_read = [Track_read(1:end-9),'_Lxypts.tsv'];
Track_save = ['D04_Track3DPoints_BRN_',Track_read(9:end-9),'.mat'];

TimeStep = 5;
Cam_FPS = 240/TimeStep; %Hz

% Specify the file name
Track_filename = fullfile(Track_foler , Track_read);
Track_data_raw = readtable(Track_filename, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);
% Convert table to array
Track_data_raw = table2array(Track_data_raw);
% Extract columns
Track_time_index = Track_data_raw(:, 1); % 1st column: Time index
Track_data_type = Track_data_raw(:, 2);  % 2nd column: Data type (1 for x-px, 2 for y-px)
Track_px_value = Track_data_raw(:, 3);   % 3rd column: Pixel value

% Sort data by time index
[time_index_sorted, sort_idx] = sort(Track_time_index);
data_sorted = Track_data_raw(sort_idx, :);

% Separate x-px and y-px data for left and right tracks
Track_x_data_L = data_sorted(data_sorted(:, 2) == 1, :); % Rows where data type is 1 (x-px)
Track_y_data_L = data_sorted(data_sorted(:, 2) == 2, :); % Rows where data type is 2 (y-px)
Track_x_data_L(:,2) = [];
Track_y_data_L(:,2) = [];

Track_x_data_R = data_sorted(data_sorted(:, 2) == 3, :); % Rows where data type is 3 (x-px)
Track_y_data_R = data_sorted(data_sorted(:, 2) == 4, :); % Rows where data type is 4 (y-px)
Track_x_data_R(:,2) = [];
Track_y_data_R(:,2) = [];

Track_x_data_R(:,2) = Track_x_data_R(:,2) - Caminfo1.imageWidth;

% Create a complete array of consecutive integers from min to max for each track
Track_x_data_L_full = (min(Track_x_data_L(:,1)):TimeStep:max(Track_x_data_L(:,1)))';
Track_y_data_L_full = (min(Track_y_data_L(:,1)):TimeStep:max(Track_y_data_L(:,1)))';
Track_x_data_R_full = (min(Track_x_data_R(:,1)):TimeStep:max(Track_x_data_R(:,1)))';
Track_y_data_R_full = (min(Track_y_data_R(:,1)):TimeStep:max(Track_y_data_R(:,1)))';

% Create a new full dataset with NaN in the second column for missing values for each track
Track_x_data_L_full_data = zeros(length(Track_x_data_L_full), 2);
Track_x_data_L_full_data(:, 1) = Track_x_data_L_full;
[~, ia, ib] = intersect(Track_x_data_L_full, Track_x_data_L(:, 1));
Track_x_data_L_full_data(ia, 2) = Track_x_data_L(ib, 2);

Track_y_data_L_full_data = zeros(length(Track_y_data_L_full), 2);
Track_y_data_L_full_data(:, 1) = Track_y_data_L_full;
[~, ia, ib] = intersect(Track_y_data_L_full, Track_y_data_L(:, 1));
Track_y_data_L_full_data(ia, 2) = Track_y_data_L(ib, 2);

Track_x_data_R_full_data = zeros(length(Track_x_data_R_full), 2);
Track_x_data_R_full_data(:, 1) = Track_x_data_R_full;
[~, ia, ib] = intersect(Track_x_data_R_full, Track_x_data_R(:, 1));
Track_x_data_R_full_data(ia, 2) = Track_x_data_R(ib, 2);

Track_y_data_R_full_data = zeros(length(Track_y_data_R_full), 2);
Track_y_data_R_full_data(:, 1) = Track_y_data_R_full;
[~, ia, ib] = intersect(Track_y_data_R_full, Track_y_data_R(:, 1));
Track_y_data_R_full_data(ia, 2) = Track_y_data_R(ib, 2);

% Find the common time range across all tracks
Track_minTime = max([min(Track_x_data_L(:,1)) min(Track_y_data_L(:,1)) min(Track_x_data_R(:,1)) min(Track_y_data_R(:,1))]);
Track_maxTime = min([max(Track_x_data_L(:,1)) max(Track_y_data_L(:,1)) max(Track_x_data_R(:,1)) max(Track_y_data_R(:,1))]);

% Extract data within the common time range
Track_x_data_L_range = Track_x_data_L_full_data(Track_x_data_L_full_data(:,1)>=Track_minTime & Track_x_data_L_full_data(:,1)<=Track_maxTime, :);
Track_y_data_L_range = Track_y_data_L_full_data(Track_y_data_L_full_data(:,1)>=Track_minTime & Track_y_data_L_full_data(:,1)<=Track_maxTime, :);
Track_x_data_R_range = Track_x_data_R_full_data(Track_x_data_R_full_data(:,1)>=Track_minTime & Track_x_data_R_full_data(:,1)<=Track_maxTime, :);
Track_y_data_R_range = Track_y_data_R_full_data(Track_y_data_R_full_data(:,1)>=Track_minTime & Track_y_data_R_full_data(:,1)<=Track_maxTime, :);

% Create the final TrackData array
TrackData = zeros(length(Track_x_data_L_range), 5);
TrackData(:, 1) = Track_x_data_L_range(:, 1); % Time index
TrackData(:, 2) = Track_x_data_L_range(:, 2); % X at left
TrackData(:, 3) = Track_y_data_L_range(:, 2); % Y at left
TrackData(:, 4) = Track_x_data_R_range(:, 2); % X at right
TrackData(:, 5) = Track_y_data_R_range(:, 2); % Y at right

[Tracklength, ~] = size(TrackData);

TrackData(:,1) = TrackData(:,1)/Cam_FPS;

patch('Vertices', plotVertices, 'Faces', plotFaces, ...
      'FaceVertexCData', plotColors, ...
      'FaceColor', 'interp', 'EdgeColor', 'none'); % Interpolate face colors
hold on;

%% %%%%%%%%% BACK-PROJECT IMAGE POINT (500, 500) INTO 3D SPACE %%%%%%%%%

track_save_point = zeros(Tracklength,3);
h = waitbar(0, 'Processing...'); % Initialize progress bar

for i=1:Tracklength
    % Display message
    % disp('3D curve plotted for image point projected into the 3D field.');
        % Update progress bar at 10% intervals
    if mod(i, round(Tracklength * 0.1)) == 0 || i == Tracklength
        waitbar(i / Tracklength, h, sprintf('Progress: %d%%', round(100 * i / Tracklength)));
    end

    imagePoint2D_1(1) = TrackData(i,2);
    imagePoint2D_1(2) = TrackData(i,3);
    imagePoint2D_2(1) = TrackData(i,4);
    imagePoint2D_2(2) = TrackData(i,5);

    undistortedPoint2D_1 = undistortPoints([imagePoint2D_1(1) imagePoint2D_1(2)], Caminfo1.intrinsics);
    undistortedPoint2D_2 = undistortPoints([imagePoint2D_2(1) imagePoint2D_2(2)], Caminfo1.intrinsics);

    % Convert to normalized image coordinates
    normalizedImagePoint_1 = [-(undistortedPoint2D_1(1) - Caminfo1.principalPoint(1)) / Caminfo1.focalLengthX;
        -(undistortedPoint2D_1(2) - Caminfo1.principalPoint(2)) / Caminfo1.focalLengthY;
        1]; % Assume z = 1 for normalization
    normalizedImagePoint_2 = [-(undistortedPoint2D_2(1) - Caminfo2.principalPoint(1)) / Caminfo2.focalLengthX;
        -(undistortedPoint2D_2(2) - Caminfo2.principalPoint(2)) / Caminfo2.focalLengthY;
        1]; % Assume z = 1 for normalization

    % Use camera intrinsic parameters to back-project the 2D point to 3D ray
    backProjectionDirection_1 = Caminfo1.rotationMatrix * normalizedImagePoint_1;
    backProjectionDirection_2 = Caminfo2.rotationMatrix * normalizedImagePoint_2;

    % Scale the back-projected ray to pass through the 3D field
    t = linspace(0, 1, 100); % Parameter t to scale the line (0 to 2 units)
    ray3DPoints_1 = Caminfo1.cameraPosition + t' .* backProjectionDirection_1';
    ray3DPoints_2 = Caminfo2.cameraPosition + t' .* backProjectionDirection_2';





    %% %%%%%%%% ADDITIONAL CODE TO CAPTURE THE IMAGE %%%%%%%%%%%%

    % Get starting point and direction vector of each ray
    p1 = ray3DPoints_1(1, :); % First point of ray1
    d1 = ray3DPoints_1(end, :) - ray3DPoints_1(1, :); % Direction of ray1 from first to last point
    d1 = d1 / norm(d1); % Normalize direction

    p2 = ray3DPoints_2(1, :); % First point of ray2
    d2 = ray3DPoints_2(end, :) - ray3DPoints_2(1, :); % Direction of ray2 from first to last point
    d2 = d2 / norm(d2); % Normalize direction

    % Calculate the cross product of direction vectors
    cross_d1d2 = cross(d1, d2);
    cross_d1d2_norm = norm(cross_d1d2);

    % Calculate the shortest distance using linear algebra
    A = [dot(d1, d1), -dot(d1, d2);
        dot(d1, d2), -dot(d2, d2)];
    B = [dot(d1, p2 - p1);
        dot(d2, p2 - p1)];

    % Solve for t1 and t2 (positions on the rays where closest points occur)
    t = A \ B;
    t1 = t(1);
    t2 = t(2);

    % Calculate closest points
    closestPoint1 = p1 + t1 * d1;
    closestPoint2 = p2 + t2 * d2;

    midpoint = (closestPoint1 + closestPoint2)/2;

    % Normalize TrackData(i,1) to a range (0 to 1) for color mapping
    time_index = TrackData(i, 1); % Get the time index
    normalized_time = (time_index - Track_minTime/Cam_FPS) / (Track_maxTime/Cam_FPS - Track_minTime/Cam_FPS); % Normalize to [0, 1]
    
    % Convert normalized time to a color using a colormap (e.g., jet)
    colormap_jet = jet; % Get the jet colormap
    color_index = round(normalized_time * (size(colormap_jet, 1) - 1)) + 1; % Map to index of colormap
    color = colormap_jet(color_index, :); % Extract the RGB color from the colormap
    
    % Plot the midpoint with the color corresponding to the time index
    plot3(midpoint(1), midpoint(2), midpoint(3), 'Marker', '^', ...
          'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'MarkerSize', 2); 
    hold on;
    track_save_point(i,:) = [midpoint(1), midpoint(2), midpoint(3)];

end
close(h); % Close progress bar when done

colormap(jet); % Set the colormap to jet
colorbar_handle = colorbar; % Create the colorbar
clim([0, Track_maxTime/Cam_FPS-Track_minTime/Cam_FPS]); % Set the colorbar limits to match the time index range

fill3(projected_points(:, 1), projected_points(:, 2), projected_points(:, 3), 'cyan', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;

set(colorbar_handle, 'Position', [0.9, 0.15, 0.02, 0.5]); % [x, y, width, height] of colorbar
set(colorbar_handle, 'FontSize', 10); % Set the font size of the colorbar labels
ylabel(colorbar_handle, 'Time [s]'); % Add a label to the colorbar


axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');
xlim([-0 1.5])
ylim([-1 0.5])
% zlim([0.1 0.9])

% Use pbaspect to control the aspect ratio of the plot
pbaspect([1, 1, 1]); % Set aspect ratio (X:Y:Z) to 1:1:1 
% camzoom(1.5); % Increase magnification of the view (1.5x zoom) 

title('3D Trajectory');
grid on;
% view([-0.0 -0.7 0.2])  % Set a 3D view angle








%%





% %% Get Length Info
% TrackLength_filename = fullfile(Track_foler , TrackLength_read);
% TrackLength_data_raw = readtable(TrackLength_filename, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);
% TrackLength_data_raw = table2array(TrackLength_data_raw);
% 
% TrackLength_time_index = TrackLength_data_raw(:, 1); % 1st column: Time index
% TrackLength_data_type = TrackLength_data_raw(:, 2);  % 2nd column: Data type (1 for x-px, 2 for y-px)
% TrackLength_px_value = TrackLength_data_raw(:, 3);   % 3rd column: Pixel value
% 
% % Sort data by time index
% [time_index_sorted, sortLength_idx] = sort(TrackLength_time_index);
% dataLength_sorted = TrackLength_data_raw(sortLength_idx, :);
% 
% % Separate x-px and y-px data for left and right tracks
% TrackLength_x_data_L_H = dataLength_sorted(dataLength_sorted(:, 2) == 1, :); % Rows where data type is 1 (x-px)
% TrackLength_y_data_L_H = dataLength_sorted(dataLength_sorted(:, 2) == 2, :); % Rows where data type is 2 (y-px)
% TrackLength_x_data_L_H(:,2) = [];
% TrackLength_y_data_L_H(:,2) = [];
% 
% TrackLength_x_data_L_T = dataLength_sorted(dataLength_sorted(:, 2) == 3, :); % Rows where data type is 1 (x-px)
% TrackLength_y_data_L_T = dataLength_sorted(dataLength_sorted(:, 2) == 4, :); % Rows where data type is 2 (y-px)
% TrackLength_x_data_L_T(:,2) = [];
% TrackLength_y_data_L_T(:,2) = [];
% 
% TrackLength_x_data_R_H = dataLength_sorted(dataLength_sorted(:, 2) == 5, :); % Rows where data type is 1 (x-px)
% TrackLength_y_data_R_H = dataLength_sorted(dataLength_sorted(:, 2) == 6, :); % Rows where data type is 2 (y-px)
% TrackLength_x_data_R_H(:,2) = [];
% TrackLength_y_data_R_H(:,2) = [];
% 
% TrackLength_x_data_R_T = dataLength_sorted(dataLength_sorted(:, 2) == 7, :); % Rows where data type is 1 (x-px)
% TrackLength_y_data_R_T = dataLength_sorted(dataLength_sorted(:, 2) == 8, :); % Rows where data type is 2 (y-px)
% TrackLength_x_data_R_T(:,2) = [];
% TrackLength_y_data_R_T(:,2) = [];


track_save_point = zeros(Tracklength,3);


[Tracklength, ~] = size(TrackData);

for i=1:Tracklength

    imagePoint2D_1(1) = TrackData(i,2);
    imagePoint2D_1(2) = TrackData(i,3);
    imagePoint2D_2(1) = TrackData(i,4);
    imagePoint2D_2(2) = TrackData(i,5);

    undistortedPoint2D_1 = undistortPoints([imagePoint2D_1(1) imagePoint2D_1(2)], Caminfo1.intrinsics);
    undistortedPoint2D_2 = undistortPoints([imagePoint2D_2(1) imagePoint2D_2(2)], Caminfo1.intrinsics);

    % Convert to normalized image coordinates
    normalizedImagePoint_1 = [-(undistortedPoint2D_1(1) - Caminfo1.principalPoint(1)) / Caminfo1.focalLengthX;
        -(undistortedPoint2D_1(2) - Caminfo1.principalPoint(2)) / Caminfo1.focalLengthY;
        1]; % Assume z = 1 for normalization
    normalizedImagePoint_2 = [-(undistortedPoint2D_2(1) - Caminfo2.principalPoint(1)) / Caminfo2.focalLengthX;
        -(undistortedPoint2D_2(2) - Caminfo2.principalPoint(2)) / Caminfo2.focalLengthY;
        1]; % Assume z = 1 for normalization

    % Use camera intrinsic parameters to back-project the 2D point to 3D ray
    backProjectionDirection_1 = Caminfo1.rotationMatrix * normalizedImagePoint_1;
    backProjectionDirection_2 = Caminfo2.rotationMatrix * normalizedImagePoint_2;

    % Scale the back-projected ray to pass through the 3D field
    t = linspace(0, 1, 100); % Parameter t to scale the line (0 to 2 units)
    ray3DPoints_1 = Caminfo1.cameraPosition + t' .* backProjectionDirection_1';
    ray3DPoints_2 = Caminfo2.cameraPosition + t' .* backProjectionDirection_2';

    % Display message
    % disp('3D curve plotted for image point projected into the 3D field.');



    %% %%%%%%%% ADDITIONAL CODE TO CAPTURE THE IMAGE %%%%%%%%%%%%

    % Get starting point and direction vector of each ray
    p1 = ray3DPoints_1(1, :); % First point of ray1
    d1 = ray3DPoints_1(end, :) - ray3DPoints_1(1, :); % Direction of ray1 from first to last point
    d1 = d1 / norm(d1); % Normalize direction

    p2 = ray3DPoints_2(1, :); % First point of ray2
    d2 = ray3DPoints_2(end, :) - ray3DPoints_2(1, :); % Direction of ray2 from first to last point
    d2 = d2 / norm(d2); % Normalize direction

    % Calculate the cross product of direction vectors
    cross_d1d2 = cross(d1, d2);
    cross_d1d2_norm = norm(cross_d1d2);

    % Calculate the shortest distance using linear algebra
    A = [dot(d1, d1), -dot(d1, d2);
        dot(d1, d2), -dot(d2, d2)];
    B = [dot(d1, p2 - p1);
        dot(d2, p2 - p1)];

    % Solve for t1 and t2 (positions on the rays where closest points occur)
    t = A \ B;
    t1 = t(1);
    t2 = t(2);

    % Calculate closest points
    closestPoint1 = p1 + t1 * d1;
    closestPoint2 = p2 + t2 * d2;

    midpoint = (closestPoint1 + closestPoint2)/2;

    % Normalize TrackData(i,1) to a range (0 to 1) for color mapping
    time_index = TrackData(i, 1); % Get the time index
    normalized_time = (time_index - Track_minTime/Cam_FPS) / (Track_maxTime/Cam_FPS - Track_minTime/Cam_FPS); % Normalize to [0, 1]
    
    % Convert normalized time to a color using a colormap (e.g., jet)
    colormap_jet = jet; % Get the jet colormap
    color_index = round(normalized_time * (size(colormap_jet, 1) - 1)) + 1; % Map to index of colormap
    color = colormap_jet(color_index, :); % Extract the RGB color from the colormap
    
    % Plot the midpoint with the color corresponding to the time index
    plot3(midpoint(1), midpoint(2), midpoint(3), 'Marker', '^', ...
          'MarkerFaceColor', color, 'MarkerEdgeColor', color, 'MarkerSize', 2); 
    hold on;
    track_save_point(i,:) = [midpoint(1), midpoint(2), midpoint(3)];

end


%% Obtain length

for HToption = [0 1]

% Specify the file name
Track_filename = fullfile(Track_foler , TrackLength_read);
% Track_filename_R = fullfile(Track_foler , '02_TrackData/DLTdv8_Case01_L.tsv');
Track_L_data_raw = readtable(Track_filename, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);
% data_R = readtable(Track_filename_R, 'FileType', 'text', 'Delimiter', '\t', 'ReadVariableNames', false);

% Convert table to array
Track_L_data_raw = table2array(Track_L_data_raw);
% data_R = table2array(data_R);

% Extract columns
Track_L_time_index = Track_L_data_raw(:, 1); % 1st column: Time index
Track_L_data_type = Track_L_data_raw(:, 2);  % 2nd column: Data type (1 for x-px, 2 for y-px)
Track_L_px_value = Track_L_data_raw(:, 3);   % 3rd column: Pixel value

% Sort data by time index
[time_index_sorted, sort_idx_L] = sort(Track_L_time_index);
data_L_sorted = Track_L_data_raw(sort_idx_L, :);

% Separate x-px and y-px data for left and right tracks
Track_L_x_data_L = data_L_sorted(data_L_sorted(:, 2) == 1+2*HToption, :); % Rows where data type is 1 (x-px)
Track_L_y_data_L = data_L_sorted(data_L_sorted(:, 2) == 2+2*HToption, :); % Rows where data type is 2 (y-px)
Track_L_x_data_L(:,2) = [];
Track_L_y_data_L(:,2) = [];

Track_L_x_data_R = data_L_sorted(data_L_sorted(:, 2) == 5+2*HToption, :); % Rows where data type is 3 (x-px)
Track_L_y_data_R = data_L_sorted(data_L_sorted(:, 2) == 6+2*HToption, :); % Rows where data type is 4 (y-px)
Track_L_x_data_R(:,2) = [];
Track_L_y_data_R(:,2) = [];

Track_L_x_data_R(:,2) = Track_L_x_data_R(:,2) - Caminfo1.imageWidth;

% Create a complete array of consecutive integers from min to max for each track
Track_L_x_data_L_full = (min(Track_L_x_data_L(:,1)):max(Track_L_x_data_L(:,1)))';
Track_L_y_data_L_full = (min(Track_L_y_data_L(:,1)):max(Track_L_y_data_L(:,1)))';
Track_L_x_data_R_full = (min(Track_L_x_data_R(:,1)):max(Track_L_x_data_R(:,1)))';
Track_L_y_data_R_full = (min(Track_L_y_data_R(:,1)):max(Track_L_y_data_R(:,1)))';

% Create a new full dataset with NaN in the second column for missing values for each track
Track_L_x_data_L_full_data = zeros(length(Track_L_x_data_L_full), 2);
Track_L_x_data_L_full_data(:, 1) = Track_L_x_data_L_full;
[~, ia, ib] = intersect(Track_L_x_data_L_full, Track_L_x_data_L(:, 1));
Track_L_x_data_L_full_data(ia, 2) = Track_L_x_data_L(ib, 2);

Track_L_y_data_L_full_data = zeros(length(Track_L_y_data_L_full), 2);
Track_L_y_data_L_full_data(:, 1) = Track_L_y_data_L_full;
[~, ia, ib] = intersect(Track_L_y_data_L_full, Track_L_y_data_L(:, 1));
Track_L_y_data_L_full_data(ia, 2) = Track_L_y_data_L(ib, 2);

Track_L_x_data_R_full_data = zeros(length(Track_L_x_data_R_full), 2);
Track_L_x_data_R_full_data(:, 1) = Track_L_x_data_R_full;
[~, ia, ib] = intersect(Track_L_x_data_R_full, Track_L_x_data_R(:, 1));
Track_L_x_data_R_full_data(ia, 2) = Track_L_x_data_R(ib, 2);

Track_L_y_data_R_full_data = zeros(length(Track_L_y_data_R_full), 2);
Track_L_y_data_R_full_data(:, 1) = Track_L_y_data_R_full;
[~, ia, ib] = intersect(Track_L_y_data_R_full, Track_L_y_data_R(:, 1));
Track_L_y_data_R_full_data(ia, 2) = Track_L_y_data_R(ib, 2);

% Find the common time range across all tracks
Track_L_minTime = max([min(Track_L_x_data_L(:,1)) min(Track_L_y_data_L(:,1)) min(Track_L_x_data_R(:,1)) min(Track_L_y_data_R(:,1))]);
Track_L_maxTime = min([max(Track_L_x_data_L(:,1)) max(Track_L_y_data_L(:,1)) max(Track_L_x_data_R(:,1)) max(Track_L_y_data_R(:,1))]);

% Extract data within the common time range
Track_L_x_data_L_range = Track_L_x_data_L_full_data(Track_L_x_data_L_full_data(:,1)>=Track_L_minTime & Track_L_x_data_L_full_data(:,1)<=Track_L_maxTime, :);
Track_L_y_data_L_range = Track_L_y_data_L_full_data(Track_L_y_data_L_full_data(:,1)>=Track_L_minTime & Track_L_y_data_L_full_data(:,1)<=Track_L_maxTime, :);
Track_L_x_data_R_range = Track_L_x_data_R_full_data(Track_L_x_data_R_full_data(:,1)>=Track_L_minTime & Track_L_x_data_R_full_data(:,1)<=Track_L_maxTime, :);
Track_L_y_data_R_range = Track_L_y_data_R_full_data(Track_L_y_data_R_full_data(:,1)>=Track_L_minTime & Track_L_y_data_R_full_data(:,1)<=Track_L_maxTime, :);

% Create the final TrackData array
% TrackData = zeros(length(Track_x_data_L_range), 5);
TrackData_L(HToption + 1, 1) = Track_L_x_data_L_range(1, 1); % Time index
TrackData_L(HToption + 1, 2) = Track_L_x_data_L_range(1, 2); % X at left
TrackData_L(HToption + 1, 3) = Track_L_y_data_L_range(1, 2); % Y at left
TrackData_L(HToption + 1, 4) = Track_L_x_data_R_range(1, 2); % X at right
TrackData_L(HToption + 1, 5) = Track_L_y_data_R_range(1, 2); % Y at right

[Tracklength, ~] = size(TrackData_L);

TrackData_L(:,1) = TrackData_L(:,1)/Cam_FPS;
end

%% %%%%%%%%% BACK-PROJECT IMAGE POINT (500, 500) INTO 3D SPACE %%%%%%%%%

track_save_point_L = zeros(Tracklength,3);

for i=1:Tracklength

    imagePoint2D_1(1) = TrackData_L(i,2);
    imagePoint2D_1(2) = TrackData_L(i,3);
    imagePoint2D_2(1) = TrackData_L(i,4);
    imagePoint2D_2(2) = TrackData_L(i,5);

    undistortedPoint2D_1 = undistortPoints([imagePoint2D_1(1) imagePoint2D_1(2)], Caminfo1.intrinsics);
    undistortedPoint2D_2 = undistortPoints([imagePoint2D_2(1) imagePoint2D_2(2)], Caminfo1.intrinsics);

    % Convert to normalized image coordinates
    normalizedImagePoint_1 = [-(undistortedPoint2D_1(1) - Caminfo1.principalPoint(1)) / Caminfo1.focalLengthX;
        -(undistortedPoint2D_1(2) - Caminfo1.principalPoint(2)) / Caminfo1.focalLengthY;
        1]; % Assume z = 1 for normalization
    normalizedImagePoint_2 = [-(undistortedPoint2D_2(1) - Caminfo2.principalPoint(1)) / Caminfo2.focalLengthX;
        -(undistortedPoint2D_2(2) - Caminfo2.principalPoint(2)) / Caminfo2.focalLengthY;
        1]; % Assume z = 1 for normalization

    % Use camera intrinsic parameters to back-project the 2D point to 3D ray
    backProjectionDirection_1 = Caminfo1.rotationMatrix * normalizedImagePoint_1;
    backProjectionDirection_2 = Caminfo2.rotationMatrix * normalizedImagePoint_2;

    % Scale the back-projected ray to pass through the 3D field
    t = linspace(0, 1, 100); % Parameter t to scale the line (0 to 2 units)
    ray3DPoints_1 = Caminfo1.cameraPosition + t' .* backProjectionDirection_1';
    ray3DPoints_2 = Caminfo2.cameraPosition + t' .* backProjectionDirection_2';

    % Plot the 3D curve from the camera through the 3D space
    % plot3(ray3DPoints_1(:, 1), ray3DPoints_1(:, 2), ray3DPoints_1(:, 3), 'm', 'LineWidth', 2);
    % plot3(ray3DPoints_2(:, 1), ray3DPoints_2(:, 2), ray3DPoints_2(:, 3), 'm', 'LineWidth', 2);
    % 
    % % Highlight the start and end of the curve
    % scatter3(ray3DPoints_1(1, 1), ray3DPoints_1(1, 2), ray3DPoints_1(1, 3), 50, 'g', 'filled');
    % scatter3(ray3DPoints_2(1, 1), ray3DPoints_2(1, 2), ray3DPoints_2(1, 3), 50, 'g', 'filled');
    % scatter3(ray3DPoints_1(end, 1), ray3DPoints_1(end, 2), ray3DPoints_1(end, 3), 50, 'b', 'filled');
    % scatter3(ray3DPoints_2(end, 1), ray3DPoints_2(end, 2), ray3DPoints_2(end, 3), 50, 'b', 'filled');
    % 
    % % Annotate the start and end points
    % text(ray3DPoints_1(1, 1), ray3DPoints_1(1, 2), ray3DPoints_1(1, 3), 'Start', 'Color', 'g');
    % text(ray3DPoints_2(1, 1), ray3DPoints_2(1, 2), ray3DPoints_2(1, 3), 'Start', 'Color', 'g');
    % text(ray3DPoints_1(end, 1), ray3DPoints_1(end, 2), ray3DPoints_1(end, 3), 'End', 'Color', 'b');
    % text(ray3DPoints_2(end, 1), ray3DPoints_2(end, 2), ray3DPoints_2(end, 3), 'End', 'Color', 'b');

    % Display message
    % disp('3D curve plotted for image point projected into the 3D field.');



    %% %%%%%%%% ADDITIONAL CODE TO CAPTURE THE IMAGE %%%%%%%%%%%%

    % Get starting point and direction vector of each ray
    p1 = ray3DPoints_1(1, :); % First point of ray1
    d1 = ray3DPoints_1(end, :) - ray3DPoints_1(1, :); % Direction of ray1 from first to last point
    d1 = d1 / norm(d1); % Normalize direction

    p2 = ray3DPoints_2(1, :); % First point of ray2
    d2 = ray3DPoints_2(end, :) - ray3DPoints_2(1, :); % Direction of ray2 from first to last point
    d2 = d2 / norm(d2); % Normalize direction

    % Calculate the cross product of direction vectors
    cross_d1d2 = cross(d1, d2);
    cross_d1d2_norm = norm(cross_d1d2);

    % Calculate the shortest distance using linear algebra
    A = [dot(d1, d1), -dot(d1, d2);
        dot(d1, d2), -dot(d2, d2)];
    B = [dot(d1, p2 - p1);
        dot(d2, p2 - p1)];

    % Solve for t1 and t2 (positions on the rays where closest points occur)
    t = A \ B;
    t1 = t(1);
    t2 = t(2);

    % Calculate closest points
    closestPoint1 = p1 + t1 * d1;
    closestPoint2 = p2 + t2 * d2;

    % Calculate distance between closest points
    % distance = norm(closestPoint1 - closestPoint2);

    midpoint = (closestPoint1 + closestPoint2)/2;
    % plot3(p1(1),p1(2),p1(3),'ro','MarkerSize',100);
    % plot3(p1(1)+d1(1),p1(2)+d1(2),p1(3)+d1(3),'ro','MarkerSize',100);

    % plot3(closestPoint1(1),closestPoint1(2),closestPoint1(3),'bo');
    % plot3(closestPoint2(1),closestPoint2(2),closestPoint2(3),'bo');

    % Normalize TrackData(i,1) to a range (0 to 1) for color mapping
    time_index = TrackData_L(i, 1); % Get the time index
    normalized_time = (time_index - Track_L_minTime/Cam_FPS) / (Track_L_maxTime/Cam_FPS - Track_L_minTime/Cam_FPS); % Normalize to [0, 1]
    
    % Convert normalized time to a color using a colormap (e.g., jet)
    % colormap_jet = jet; % Get the jet colormap
    % color_index = round(normalized_time * (size(colormap_jet, 1) - 1)) + 1; % Map to index of colormap
    % color = colormap_jet(color_index, :); % Extract the RGB color from the colormap
    
    % Plot the midpoint with the color corresponding to the time index
    plot3(midpoint(1), midpoint(2), midpoint(3), 'Marker', '^','MarkerSize', 2); 
    hold on;
    track_save_point_L(i,:) = [midpoint(1), midpoint(2), midpoint(3)];

end

% colormap(jet); % Set the colormap to jet
% colorbar_handle = colorbar; % Create the colorbar
% clim([0, Track_maxTime/Cam_FPS-Track_minTime/Cam_FPS]); % Set the colorbar limits to match the time index range

% fill3(projected_points(:, 1), projected_points(:, 2), projected_points(:, 3), 'red', 'EdgeColor', 'none'); hold on;

X = [track_save_point_L(1,1), track_save_point_L(2,1)];
Y = [track_save_point_L(1,2), track_save_point_L(2,2)];
Z = [track_save_point_L(1,3), track_save_point_L(2,3)];

% Plot the line
% figure;
plot3(X, Y, Z, 'o-', 'LineWidth', 2, 'MarkerSize', 2, 'MarkerFaceColor', 'r','Color','r');
% text(track_save_point_L(1,1), track_save_point_L(1,2), track_save_point_L(1,3), ' Fish Length', 'FontSize', 12, 'Color', 'b', 'VerticalAlignment', 'bottom');


% set(colorbar_handle, 'Position', [0.9, 0.15, 0.02, 0.5]); % [x, y, width, height] of colorbar
% set(colorbar_handle, 'FontSize', 10); % Set the font size of the colorbar labels
% ylabel(colorbar_handle, 'Time [s]'); % Add a label to the colorbar

Fish_Length = norm(track_save_point_L(1,:)-track_save_point_L(2,:));


%%
save(Track_save, "track_save_point","Track_L_minTime","Track_L_maxTime","Cam_FPS","Fish_Length");
clear Track_save track_save_point_L track_save_point
end

