clc;  clear all;
close all;
% Load 3D field data
load("D02_3DFieldData_Brunei_Loc3.mat");

% Read image
parentFolder = 'G:\My Drive\Work\_GIT (2023~)\01 Mudskipper\20241216 3DTrackVideoCases\01_Videos\Brunei_Loc3\Cropped_Brunei_Loc3_0212_0222'; % Change this to your parent folder path
n = 5; % The index of the image you want to read (e.g., 5th image)
fileExtension = 'bmp'; % File extension to search for

Data_cam = 'D05_camera_parameters_Cam1_Brunei_Loc3.mat';

% Given 2D image point (500, 500) 
imagePoint2D = [500; 842; 1]; % Homogeneous 2D point

downsampleRatio = 0.1; % Keep 20% of points

% Define near and far clipping planes
near = 0.1; % Near clipping plane distance
far = 0.2; % Far clipping plane distance


%%%%%%%%%%%%%%%%%%%%%%% CAM1_GoPro Hero 12 specifications %%%%%%%%%%%%%%%%%%%%%%%

% Extrinsics: Camera position and orientation
cameraPosition = [0.691 -1.585 -0.936];  % Camera location in 3D space
cameraTarget = [0.66 -0.758 -0.95]; % Point the camera is looking at y down
cameraUpVector = [0.1, 0, 1]; % Up direction for the camera
% [0.2, 0, 1];
% Intrinsics: Camera position and orientation
imageWidth = 2704;    % Image width in pixels
imageHeight = 1520;   % Image height in pixels

% Measured Intrinsics from Calibration
focalLengthX = 1250; %1351.4236; 
focalLengthY = focalLengthX; 
principalPoint = [imageWidth/2, imageHeight/2]; 

% Measured distortion coefficients (can be updated if necessary)
radialDistortion = [0,0]%[-0.07, -0.05];  % No radial distortion (from the provided data) %[-0.05, -0.05]
tangentialDistortion = [0, 0]; % No tangential distortion (from the provided data)

% Create the camera intrinsics object
intrinsics = cameraIntrinsics([focalLengthX, focalLengthY], principalPoint, [imageHeight, imageWidth], ...
    'RadialDistortion', radialDistortion, ...
    'TangentialDistortion', tangentialDistortion);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Create a figure to visualize the 3D field
figure(2);
patch('Vertices', plotVertices, 'Faces', plotFaces, ...
      'FaceVertexCData', plotColors, ...
      'FaceColor', 'interp', 'EdgeColor', 'none'); % Interpolate face colors
hold on;

% Plot the camera location
scatter3(cameraPosition(1), cameraPosition(2), cameraPosition(3), 100, 'r', 'filled');
text(cameraPosition(1), cameraPosition(2), cameraPosition(3), 'Camera', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'right');

% Draw the camera direction as a line (axis)
cameraDirection = cameraTarget - cameraPosition;
quiver3(cameraPosition(1), cameraPosition(2), cameraPosition(3), ...
        cameraDirection(1), cameraDirection(2), cameraDirection(3), ...
        1, 'r', 'LineWidth', 2, 'MaxHeadSize', 2);
    
% Calculate the rotation matrix for the camera orientation
zAxis = cameraDirection / norm(cameraDirection);
xAxis = cross(cameraUpVector, zAxis);
xAxis = xAxis / norm(xAxis);
yAxis = cross(zAxis, xAxis);

rotationMatrix = [xAxis; yAxis; zAxis]';

% Plot the camera's X, Y, Z axes
axisLength = 0.2; % Length of the axes
xAxisWorld = axisLength * rotationMatrix(:, 1); % X-axis in world frame
yAxisWorld = axisLength * rotationMatrix(:, 2); % Y-axis in world frame
zAxisWorld = axisLength * rotationMatrix(:, 3); % Z-axis in world frame

% Plot the X-axis (Red)
quiver3(cameraPosition(1), cameraPosition(2), cameraPosition(3), ...
        xAxisWorld(1), xAxisWorld(2), xAxisWorld(3), ...
        0, 'r', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(cameraPosition(1) + xAxisWorld(1), ...
     cameraPosition(2) + xAxisWorld(2), ...
     cameraPosition(3) + xAxisWorld(3), ...
     'X', 'Color', 'r', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

% Plot the Y-axis (Green)
quiver3(cameraPosition(1), cameraPosition(2), cameraPosition(3), ...
        yAxisWorld(1), yAxisWorld(2), yAxisWorld(3), ...
        0, 'g', 'LineWidth', 2, 'MaxHeadSize', 0.5);
text(cameraPosition(1) + yAxisWorld(1), ...
     cameraPosition(2) + yAxisWorld(2), ...
     cameraPosition(3) + yAxisWorld(3), ...
     'Y', 'Color', 'g', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');

% Plot the Z-axis (Blue)
quiver3(cameraPosition(1), cameraPosition(2), cameraPosition(3), ...
        zAxisWorld(1), zAxisWorld(2), zAxisWorld(3), ...
        0, 'b', 'LineWidth', 2, 'MaxHeadSize', 0.5);
% Add text label for Z-axis
text(cameraPosition(1) + zAxisWorld(1), ...
     cameraPosition(2) + zAxisWorld(2), ...
     cameraPosition(3) + zAxisWorld(3), ...
     'Z', 'Color', 'b', 'FontSize', 12, 'FontWeight', 'bold', ...
     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');


% Calculate frustum dimensions at near and far planes
nearWidth = 2 * near * principalPoint(1) / focalLengthX;
nearHeight = 2 * near * principalPoint(2) / focalLengthY;
farWidth = 2 * far * principalPoint(1) / focalLengthX;
farHeight = 2 * far * principalPoint(2) / focalLengthY;

% Frustum corner points in camera coordinate system (origin at camera center)
frustumPoints = [
    -nearWidth / 2, -nearHeight / 2, near;
     nearWidth / 2, -nearHeight / 2, near;
     nearWidth / 2,  nearHeight / 2, near;
    -nearWidth / 2,  nearHeight / 2, near;
    -farWidth / 2,  -farHeight / 2, far;
     farWidth / 2,  -farHeight / 2, far;
     farWidth / 2,   farHeight / 2, far;
    -farWidth / 2,   farHeight / 2, far;
];

% Rotate and translate the frustum points to world coordinates
frustumPointsWorld = (rotationMatrix * frustumPoints')';
frustumPointsWorld = frustumPointsWorld + cameraPosition;

% Draw the frustum edges
frustumEdges = [
    1, 2; 2, 3; 3, 4; 4, 1; % Near plane edges
    5, 6; 6, 7; 7, 8; 8, 5; % Far plane edges
    1, 5; 2, 6; 3, 7; 4, 8  % Connect near and far planes
];

for i = 1:size(frustumEdges, 1)
    p1 = frustumPointsWorld(frustumEdges(i, 1), :);
    p2 = frustumPointsWorld(frustumEdges(i, 2), :);
    plot3([p1(1), p2(1)], [p1(2), p2(2)], [p1(3), p2(3)], 'g', 'LineWidth', 2);
end

% Draw the frustum faces
faces = [
    1, 2, 3, 4; % Near plane
    5, 6, 7, 8; % Far plane
    1, 2, 6, 5; % Side plane 1
    2, 3, 7, 6; % Side plane 2
    3, 4, 8, 7; % Side plane 3
    4, 1, 5, 8; % Side plane 4
];

% Plot frustum as a transparent surface
frustumColor = [0, 0.5, 1];
for i = 1:size(faces, 1)
    vertices = frustumPointsWorld(faces(i, :), :);
    fill3(vertices(:, 1), vertices(:, 2), vertices(:, 3), frustumColor, 'FaceAlpha', 0.1, 'EdgeColor', 'none');
end

% Set axis properties
axis equal;
xlabel('X');
ylabel('Y');
zlabel('Z');

% zlim([0.3 0.7])

title('3D Field with Camera Location, Direction, and Frustum');
grid on;
view([-0.0 -0.7 0.2])  % Set a 3D view angle


%% %%%%%%%% ADDITIONAL CODE TO CAPTURE THE IMAGE %%%%%%%%%%%%

rotationMatrix = [xAxis; yAxis; zAxis]';

% Transform 3D points into the camera's coordinate system
worldPoints = plotVertices;
relativePoints = rotationMatrix' * (worldPoints - cameraPosition)'; % Transform to camera frame
relativePoints = relativePoints';


%% %%%%%%%%% BACK-PROJECT IMAGE POINT (500, 500) INTO 3D SPACE %%%%%%%%%

undistortedPoint2D = undistortPoints([imagePoint2D(1) imagePoint2D(2)], intrinsics);

% Convert to normalized image coordinates
normalizedImagePoint = [-(undistortedPoint2D(1) - principalPoint(1)) / focalLengthX; 
                        -(undistortedPoint2D(2) - principalPoint(2)) / focalLengthY; 
                        1]; % Assume z = 1 for normalization

% Use camera intrinsic parameters to back-project the 2D point to 3D ray
backProjectionDirection = rotationMatrix * normalizedImagePoint;

% Scale the back-projected ray to pass through the 3D field
t = linspace(0, 1, 100); % Parameter t to scale the line (0 to 2 units)
ray3DPoints = cameraPosition + t' .* backProjectionDirection';

% Plot the 3D curve from the camera through the 3D space
plot3(ray3DPoints(:, 1), ray3DPoints(:, 2), ray3DPoints(:, 3), 'm', 'LineWidth', 2);

% Highlight the start and end of the curve
scatter3(ray3DPoints(1, 1), ray3DPoints(1, 2), ray3DPoints(1, 3), 50, 'g', 'filled');
scatter3(ray3DPoints(end, 1), ray3DPoints(end, 2), ray3DPoints(end, 3), 50, 'b', 'filled');

% Annotate the start and end points
text(ray3DPoints(1, 1), ray3DPoints(1, 2), ray3DPoints(1, 3), 'Start', 'Color', 'g');
text(ray3DPoints(end, 1), ray3DPoints(end, 2), ray3DPoints(end, 3), 'End', 'Color', 'b');

% Display message
disp('3D curve plotted for image point (500, 500) projected into the 3D field.');


%%%%% HERE %%%%%

% **Calculate Z-depths for sorting**
% zDepths = relativePoints(:, 3).^2 ; % Z-coordinate of each vertex relative to the camera
zDepths = relativePoints(:, 3).^2 + relativePoints(:, 2).^2 + relativePoints(:, 1).^2; % Z-coordinate of each vertex relative to the camera

% **Sort vertices (relativePoints), plotColors, and plotFaces by Z-depth**
[~, vertexSortIdx] = sort(zDepths, 'descend'); % Sort farthest to nearest
sortedRelativePoints = relativePoints(vertexSortIdx, :); % Sort vertices
sortedPlotColors = plotColors(vertexSortIdx, :); % Sort vertex colors

% **Update plotFaces to match the sorted vertices**
% Remap the vertex indices in plotFaces to match the sorted vertices
reverseIndexMap = zeros(size(vertexSortIdx)); 
reverseIndexMap(vertexSortIdx) = 1:length(vertexSortIdx); % Map sorted indices back to original positions
sortedPlotFaces = reverseIndexMap(plotFaces); % Reindex the face vertices

% **Remove invalid faces (faces that reference non-existing vertices)**
% Ensure faces have valid vertex indices
validFaces = all(sortedPlotFaces > 0 & sortedPlotFaces <= size(sortedRelativePoints, 1), 2);
sortedPlotFaces = sortedPlotFaces(validFaces, :); 

% % **Remove duplicate faces (if necessary)**
% [sortedPlotFaces, uniqueFaceIdx] = unique(sortedPlotFaces, 'rows', 'stable'); 
% sortedPlotColors = sortedPlotColors(uniqueFaceIdx, :); % Sort face colors to match sortedPlotFaces

%%%%% END OF SORTING %%%%%


% Determine which points are in front of the camera
isInFront = sortedRelativePoints(:, 3) > 0.1;
visiblePoints = sortedRelativePoints(isInFront, :);
visibleIndices = find(isInFront); % Keep track of which points remain

% Update plotColors to correspond to visible points
plotColors2 = sortedPlotColors(isInFront, :);

% Project the 3D points to the 2D image plane using intrinsic parameters
imagePoints = [focalLengthX, 0, principalPoint(1);
               0, focalLengthY, principalPoint(2);
               0, 0, 1] * visiblePoints';

% Normalize the points
imagePoints(1, :) = imagePoints(1, :) ./ imagePoints(3, :);
imagePoints(2, :) = imagePoints(2, :) ./ imagePoints(3, :);

% Normalize image points for distortion application
normalizedPoints = [(imagePoints(1, :) - principalPoint(1)) / focalLengthX; 
                    (imagePoints(2, :) - principalPoint(2)) / focalLengthY];

% Apply radial distortion
r2 = normalizedPoints(1, :).^2 + normalizedPoints(2, :).^2;
radialDistortion = 1 + intrinsics.RadialDistortion(1) * r2 + ...
                       intrinsics.RadialDistortion(2) * r2.^2;

% Apply tangential distortion
tangentialX = 2 * intrinsics.TangentialDistortion(1) * normalizedPoints(1, :) .* normalizedPoints(2, :) + ...
              intrinsics.TangentialDistortion(2) * (r2 + 2 * normalizedPoints(1, :).^2);
tangentialY = 2 * intrinsics.TangentialDistortion(2) * normalizedPoints(1, :) .* normalizedPoints(2, :) + ...
              intrinsics.TangentialDistortion(1) * (r2 + 2 * normalizedPoints(2, :).^2);

% Apply distortions to the normalized points
distortedX = normalizedPoints(1, :) .* radialDistortion + tangentialX;
distortedY = normalizedPoints(2, :) .* radialDistortion + tangentialY;

% Convert back to pixel coordinates
imagePoints(1, :) = -distortedX * focalLengthX + principalPoint(1);
imagePoints(2, :) = -distortedY * focalLengthY + principalPoint(2);

% Update plotFaces to refer to the new indices of visible points
updatedIndexMap = zeros(size(sortedRelativePoints, 1), 1);
updatedIndexMap(visibleIndices) = 1:length(visibleIndices); % Map original indices to new indices
validFaces = all(ismember(sortedPlotFaces, visibleIndices), 2); % Check if all points in a face are still visible
updatedFaces = sortedPlotFaces(validFaces, :); % Remove invalid faces
updatedFaces = updatedIndexMap(updatedFaces); % Update the vertex indices of the faces


% ** Step 1: Filter only points visible within image bounds **
marginpx = 00;
validX = imagePoints(1, :) >= -marginpx & imagePoints(1, :) <= imageWidth+marginpx;
validY = imagePoints(2, :) >= -marginpx & imagePoints(2, :) <= imageHeight+marginpx;
visiblePointsMask = validX & validY;
visibleImagePoints = imagePoints(:, visiblePointsMask);
visiblePlotColors = plotColors2(visiblePointsMask, :);

% ** Step 2: Downsample points to reduce density **
numPoints = size(visibleImagePoints, 2);
% Calculate the total number of points to keep
numToKeep = round(downsampleRatio * numPoints);

% Generate indices using linear spacing
indicesToKeep = round(linspace(1, numPoints, numToKeep));

visibleImagePoints = visibleImagePoints(:, indicesToKeep);
visiblePlotColors = visiblePlotColors(indicesToKeep, :);

%% Plot

% Create an image from the projected points


% Get a list of all BMP images in the parent folder and its subfolders
imageFiles = dir(fullfile(parentFolder, '**', ['*.', fileExtension]));
imagePath = fullfile(imageFiles(n).folder, imageFiles(n).name);
imageData = imread(imagePath);
figure(1); 
imshow(imageData); hold on;

figure(1);
scatter(visibleImagePoints(1, :)', visibleImagePoints(2, :)', 5, visiblePlotColors, 'filled', 'MarkerFaceAlpha', 0.2);
hold on;
plot(imagePoint2D(1),imagePoint2D(2),'ro');
axis equal;
xlim([0, imageWidth]);
ylim([0, imageHeight]);
set(gca, 'YDir', 'reverse');
% axes.YDir = 'reverse';

%%
figure(4);
subplot(1,2,1);
imshow(imageData); 

% Add grid on top of the image
[rows, columns, numberOfColorChannels] = size(imageData);
hold on;
lineSpacing = 100; % Whatever you want.
for row = lineSpacing : lineSpacing : rows
    line([1, columns], [row, row], 'Color', 'r', 'LineWidth', 0.5);
end
for col = lineSpacing*2 : lineSpacing*2 : columns
    line([col, col], [1, rows], 'Color', 'r', 'LineWidth', 0.5);
end

subplot(1,2,2);
scatter(visibleImagePoints(1, :)', visibleImagePoints(2, :)', 3, visiblePlotColors, 'filled'); 
hold on;
plot(imagePoint2D(1),imagePoint2D(2),'ro');
axis equal;
xlim([0, imageWidth]);
ylim([0, imageHeight]);
set(gca, 'YDir', 'reverse');
title('Image Captured from Virtual Camera with Distortion');
xlabel('Pixels (X-axis)');
ylabel('Pixels (Y-axis)');
% axes.YDir = 'reverse';
box on;
% 
% Get the current axes handle
ax = gca; 
ax.Layer = 'top'; % Bring the grid on top
ax.GridColor = [1, 0, 0]; % Set grid color to red
ax.GridAlpha = 0.5; % Set transparency of the grid (1 = opaque, 0 = transparent)
grid on; % Turn on the grid

% Control label intervals for x-axis and y-axis
xTickInterval = 200; % Set the x-axis interval (adjust as needed)
yTickInterval = 100; % Set the y-axis interval (adjust as needed)
ax.XTick = 0:xTickInterval:imageWidth; % Set x-axis tick positions
ax.YTick = 0:yTickInterval:imageHeight; % Set y-axis tick positions

set(gcf,'Position', get(0,'ScreenSize'))



% % Save the image as a file
% frame = getframe(gca);
% imwrite(frame.cdata, 'captured_image_with_distortion.png');

% Save camera extrinsics and intrinsics to a .mat file
save(Data_cam, 'cameraPosition', 'cameraTarget', 'cameraUpVector', 'imageWidth', 'imageHeight', ...
     'focalLengthX', 'focalLengthY', 'principalPoint', ...
     'radialDistortion', 'tangentialDistortion', 'rotationMatrix', 'intrinsics','rotationMatrix','cameraDirection', ...
     'frustumPointsWorld','xAxisWorld','yAxisWorld','zAxisWorld');



