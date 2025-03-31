clc; close all; clear all;

% Path to the OBJ file and texture image
objFilePath = 'G:\My Drive\Work\_GIT (2023~)\01 Mudskipper\20241216 3DTrackVideoCases\01_Videos\Incheon_Loc2\ObjFiles\8_24_2024.obj'; % Replace with your OBJ file path
textureImagePath = 'G:\My Drive\Work\_GIT (2023~)\01 Mudskipper\20241216 3DTrackVideoCases\01_Videos\Incheon_Loc2\ObjFiles\textures\color.jpg'; % Replace with your texture image path
polygonFilePath = 'G:\My Drive\Work\_GIT (2023~)\01 Mudskipper\20241216 3DTrackVideoCases\01_Videos\Incheon_Loc2\ObjFiles\polygon.xlsx'; % Excel file with polygon points

dataFile = 'D02_3DFieldData_Incheon_Loc1.mat';

subdivisionLevel = 3; % Increase for finer subdivisions

disp('OBJ file reading...');

% Open the OBJ file
fid = fopen(objFilePath, 'r');
if fid == -1
    error('Failed to open the file.');
end

% Initialize arrays
vertices = [];
faces = [];
textureCoords = [];
faceTextureIndices = [];

% Read the file line by line
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if startsWith(line, 'v ') % Vertex definition
        vertex = sscanf(line(2:end), '%f %f %f');
        vertices = [vertices; vertex'];
    elseif startsWith(line, 'vt ') % Texture coordinate definition
        texCoord = sscanf(line(3:end), '%f %f');
        textureCoords = [textureCoords; texCoord'];
    elseif startsWith(line, 'f ') % Face definition
        % Parse face indices in the format v/vt/vn
        faceParts = regexp(line(2:end), '\s+', 'split');
        face = [];
        texIndices = [];
        for i = 2:length(faceParts)
            indices = sscanf(faceParts{i}, '%d/%d/%d');
            face = [face, indices(1)]; % Vertex index (v)
            texIndices = [texIndices, indices(2)]; % Texture index (vt)
        end
        faces = [faces; face];
        faceTextureIndices = [faceTextureIndices; texIndices];
    end
end
disp('OBJ file read done');

% Close the file
fclose(fid);



%%

% User input: three points on the reference plane
% disp('Input three points (as [x, y, z]) on the reference plane:');
P1 = [0.195017250000000,0.664487125000000,0.0634277500000000]; %input('Point 1: ');
P3 = [0.391021000000000,0.333203000000000,0.0314905000000000]; %input('Point 2: ');
P2 = [0.302736500000000,0.347288250000000,0.148493000000000];%input('Point 3: ');

% P1 = [0,0,0]; %input('Point 1: ');
% P3 = [1,0,0]; %input('Point 2: ');
% P2 = [0,1,0];%input('Point 3: ');

% Compute the normal vector of the reference plane
v1 = P2 - P1;
v2 = P3 - P1;
normal = cross(v1, v2);
normal = normal / norm(normal); % Normalize the normal vector

% Rotation: Align the normal vector to the Z-axis
zAxis = [0, 0, 1];
rotationAxis = cross(normal, zAxis);
rotationAngle = acos(dot(normal, zAxis));
if norm(rotationAxis) > 1e-6
    rotationMatrix = axang2rotm([rotationAxis / norm(rotationAxis), rotationAngle]);
else
    rotationMatrix = eye(3); % If already aligned, use identity matrix
end

% Apply rotation to all vertices
vertices = (rotationMatrix * vertices')';

% User input: new origin point
% disp('Input the new origin point (as [x, y, z]):');
newOrigin = P3;%input('New origin: ');

% Translate vertices to set the new origin
translationVector = -newOrigin;
vertices = vertices + translationVector;


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Fixing %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

excludePolygon = readmatrix(polygonFilePath, 'Range', 'A2:C1000'); % Read x and y columns from the 2nd row onwards
if isempty(excludePolygon)
    error('Polygon file does not contain any valid data.');
end

% User-defined polygon to exclude
% excludePolygon = [-0.291, -0.4994; -0.9231, 1.3387; -0.6250, 1.3957; -0.041, -0.4391];
% excludePolygon = [0, 0; 0, 0; 0, 0; 0, 0];

% 1. Remove vertices that are inside the exclusion polygon
inPolygon = inpolygon(vertices(:, 1), vertices(:, 2), excludePolygon(:, 1), excludePolygon(:, 2));
excludedVertexIndices = find(inPolygon);
vertices(excludedVertexIndices, :) = [];

% 2. Update face vertex indices to account for excluded vertices
allVertexIndices = 1:size(vertices, 1) + numel(excludedVertexIndices);
validVertexIndices = setdiff(allVertexIndices, excludedVertexIndices);
[~, newFaceIndices] = ismember(faces, validVertexIndices);

% Update face indices
faces = reshape(newFaceIndices, size(faces));

% Remove any face with an invalid vertex reference (0 means vertex was removed)
invalidFaces = any(faces == 0, 2);
faces(invalidFaces, :) = [];
faceTextureIndices(invalidFaces, :) = []; % Also remove corresponding face texture indices

% 3. Remove texture coordinates associated with excluded vertices
% This is done by checking which texture coordinates are no longer referenced in faceTextureIndices
uniqueTexIndices = unique(faceTextureIndices(:));
allTexIndices = 1:size(textureCoords, 1);
unusedTexIndices = setdiff(allTexIndices, uniqueTexIndices);
textureCoords(unusedTexIndices, :) = [];

% 4. Update texture indices to remove unused indices and re-index faceTextureIndices
[~, newTextureIndices] = ismember(faceTextureIndices, uniqueTexIndices);

% Update face texture indices
faceTextureIndices = reshape(newTextureIndices, size(faceTextureIndices));

% Remove any face with an invalid texture reference (0 means texture index was removed)
invalidTextureFaces = any(faceTextureIndices == 0, 2);
faces(invalidTextureFaces, :) = [];
faceTextureIndices(invalidTextureFaces, :) = [];

disp('Vertex, texture, face, and texture index processing complete');


% Compute the mean (center) of the points
excludePolygon(any(isnan(excludePolygon), 2), :) = [];
center = mean(excludePolygon, 1);

% Subtract the center to find deviations
deviations = excludePolygon - center;

% Perform Singular Value Decomposition (SVD)
[~, ~, V] = svd(deviations, 'econ');

% The normal vector to the plane
normal = V(:, end);

% Project the points onto the plane
projected_points = deviations - (deviations * normal) * normal';
projected_points = projected_points + center;

% Compute the convex hull of the projected points
hull_indices = convhull(projected_points(:, 1), projected_points(:, 2));
hull_points = projected_points(hull_indices, :);

% Plot the scatter points
% fill3(projected_points(:, 1), projected_points(:, 2), projected_points(:, 3), 'cyan', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%

% Load the texture image
textureImage = imread(textureImagePath);

% Normalize texture coordinates
textureCoords(:, 1) = textureCoords(:, 1); % U-coordinates (already normalized 0-1)
textureCoords(:, 2) = 1 - textureCoords(:, 2); % Flip V-coordinates for MATLAB

% Preallocate space for subdivided data
numFaces = size(faces, 1);
estimatedVertices = numFaces * (3 * (4 ^ subdivisionLevel)); % Estimate total vertices
estimatedFaces = numFaces * (4 ^ subdivisionLevel);          % Estimate total faces

plotVertices = zeros(estimatedVertices, 3); % Preallocate for vertex positions
plotFaces = zeros(estimatedFaces, 3);       % Preallocate for face indices
plotColors = zeros(estimatedVertices, 3);  % Preallocate for vertex colors

vertexCount = 0; % To track current number of vertices
faceCount = 0;   % To track current number of faces

% Subdivide each face
for i = 1:numFaces
    disp(['Subdivision progress:',num2str(round(i/numFaces*100)),'%']);
    currentFace = faces(i, :); % Indices of the vertices for the current face
    currentTexIndices = faceTextureIndices(i, :); % Texture indices of the current face

    % Get vertices and corresponding texture coordinates
    faceVertices = vertices(currentFace, :);
    uvCoords = textureCoords(currentTexIndices, :);

    % Subdivide the face
    [subdividedVertices, subdividedUV, subdividedFaces] = subdivideTriangle(faceVertices, uvCoords, subdivisionLevel);

    % Convert UV coordinates to pixel indices for the texture image
    subdividedUV(:, 1) = subdividedUV(:, 1) * (size(textureImage, 2) - 1) + 1; % U to column
    subdividedUV(:, 2) = subdividedUV(:, 2) * (size(textureImage, 1) - 1) + 1; % V to row

    % Get the texture colors for the subdivided vertices
    subdividedColors = zeros(size(subdividedVertices, 1), 3);
    for j = 1:size(subdividedVertices, 1)
        row = round(subdividedUV(j, 2));
        col = round(subdividedUV(j, 1));
        subdividedColors(j, :) = double(reshape(textureImage(row, col, :), 1, 3)) / 255; % Normalize to [0, 1]
    end

    % Add the subdivided data to the preallocated arrays
    numSubVertices = size(subdividedVertices, 1);
    numSubFaces = size(subdividedFaces, 1);

    plotVertices(vertexCount + 1:vertexCount + numSubVertices, :) = subdividedVertices;
    plotColors(vertexCount + 1:vertexCount + numSubVertices, :) = subdividedColors;
    plotFaces(faceCount + 1:faceCount + numSubFaces, :) = subdividedFaces + vertexCount;

    % Update counters
    vertexCount = vertexCount + numSubVertices;
    faceCount = faceCount + numSubFaces;
end

% Remove unused preallocated space
plotVertices = plotVertices(1:vertexCount, :);
plotColors = plotColors(1:vertexCount, :);
plotFaces = plotFaces(1:faceCount, :);
% Plot the transformed 3D field
figure;
patch('Vertices', plotVertices, 'Faces', plotFaces, ...
      'FaceVertexCData', plotColors, ...
      'FaceColor', 'interp', 'EdgeColor', 'none'); % Interpolate face colors
hold on;
fill3(projected_points(:, 1), projected_points(:, 2), projected_points(:, 3), 'cyan', 'FaceAlpha', 0.5, 'EdgeColor', 'none'); hold on;
save('D03_WaterEdge', 'projected_points','normal','center');

axis equal;
view(3);
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Transformed 3D Field');
grid on;

% Save data
save(dataFile, 'plotVertices', 'plotFaces', 'plotColors');
disp('Transformed 3D field data saved.');

% Function to subdivide a triangle
function [subVertices, subUV, subFaces] = subdivideTriangle(vertices, uvCoords, level)
    % Initialize
    subVertices = vertices;
    subUV = uvCoords;
    subFaces = [1, 2, 3];
    
    % Subdivide
    for l = 1:level
        newVertices = [];
        newUV = [];
        newFaces = [];
        
        for f = 1:size(subFaces, 1)
            % Get the current face vertices and UVs
            v1 = subVertices(subFaces(f, 1), :);
            v2 = subVertices(subFaces(f, 2), :);
            v3 = subVertices(subFaces(f, 3), :);
            uv1 = subUV(subFaces(f, 1), :);
            uv2 = subUV(subFaces(f, 2), :);
            uv3 = subUV(subFaces(f, 3), :);
            
            % Midpoints and their UVs
            vm12 = (v1 + v2) / 2;
            vm23 = (v2 + v3) / 2;
            vm31 = (v3 + v1) / 2;
            uv12 = (uv1 + uv2) / 2;
            uv23 = (uv2 + uv3) / 2;
            uv31 = (uv3 + uv1) / 2;
            
            % Add new vertices and UVs
            newVertices = [newVertices; v1; v2; v3; vm12; vm23; vm31];
            newUV = [newUV; uv1; uv2; uv3; uv12; uv23; uv31];
            
            % New faces
            offset = size(newVertices, 1) - 6;
            newFaces = [newFaces;
                        offset + [1, 4, 6];
                        offset + [2, 5, 4];
                        offset + [3, 6, 5];
                        offset + [4, 5, 6]];
        end
        
        % Update for the next level
        subVertices = newVertices;
        subUV = newUV;
        subFaces = newFaces;
    end
end


% for i=1:size(cursor_info,2)
%     Cali_points(i,:)=cursor_info(i).Position;
% end
% save('D05_Calibration',"Cali_points");