# mudskipper-3D-tracking-ICB-2025
3D tracking for mudskippers - Choi et al. ICB 2025 (in revision)

> **Note**:  
> - The OBJ file name (e.g., `8_24_2024.obj`) reflects the date it was measured.  
> - The `_Lxypts.tsv` tracks contain additional length-measurement data for the fish’s body.  
> - The `polygon.xlsx` file defines a polygon boundary (e.g., the water edge) to exclude certain vertices from the terrain.

---

## Prerequisites

1. **MATLAB** (R2020b or later is recommended)  
2. **Computer Vision Toolbox** (used for camera calibration, undistortPoints, etc.)  
3. **Image Processing Toolbox** (for handling textures, images)  

---

## Workflow Overview

Below is the recommended order to run the scripts and how each part contributes to the 3D reconstruction.

### 1. Import the OBJ File into MATLAB

- **Folder**: `01_Matlab/01_ImportObj2Matlab`  
- **Key Script**: `C01_Draw3DField_v4_MoreDenser_Rotate_*.m`

**Purpose**  
Reads the OBJ file (`.obj`, `.mtl`, and textures) and:
- Parses and stores vertices, faces, and texture coordinates.
- Applies transformations: e.g., rotating the model, setting a new origin, or aligning the plane with Z=0.
- Excludes unwanted regions using `polygon.xlsx` (e.g., removing areas outside the water boundary).
- Subdivides faces (optional) to get a denser mesh for smooth rendering.
- Saves the processed 3D data (vertices, faces, colors) to a `.mat` file.

**Usage**  
1. Update the paths in the script (`objFilePath`, `textureImagePath`, `polygonFilePath`) to point to your OBJ and texture files in `02_Data/ObjFiles`.
2. Adjust rotation angles or reference points (e.g., `P1`, `P2`, `P3`) if you need a specific orientation.
3. Run the script in MATLAB.  
4. It will generate and save a `.mat` file with the transformed 3D mesh data (for example, `D02_3DFieldData_*.mat`).

---

### 2. Match the Camera View with the Virtual Camera

- **Folder**: `01_Matlab/02_MatchingCameraViewWithVirtualCamera`  
- **Key Scripts**: `C04_CamInfoCam1_*.m` and `C04_CamInfoCam2_*.m`

**Purpose**  
Simulates a “virtual camera” in MATLAB that matches your real camera's intrinsics (focal length, principal point, distortion) and extrinsics (camera position, target, and up-vector). This step allows you to:
- Project the imported 3D terrain onto the same viewpoint as in your real video.
- Adjust near/far clipping planes and view frustum to see what portion of the 3D scene is visible.

**Usage**  
1. Load the `.mat` file from the previous step (e.g., `D02_3DFieldData_*.mat`) that contains `plotVertices`, `plotFaces`, and `plotColors`.
2. Set camera intrinsic parameters:
   - Focal length
   - Principal point
   - Distortion coefficients, if any  
3. Set camera extrinsic parameters:
   - `cameraPosition` (3D location)
   - `cameraTarget` (the look-at point)
   - `cameraUpVector` (defines the camera’s “up” direction)
4. Run the script to visualize:
   - The 3D environment
   - The camera's position and orientation
   - A wireframe representing the camera’s viewing frustum
5. An optional section projects 3D points onto the 2D camera plane, generating a synthetic image from your virtual camera.

You can repeat for each camera, e.g., `Cam1` and `Cam2`, to save their respective camera parameters in `.mat` files (e.g., `camera_parameters_Cam1_*.mat`).

---

### 3. Reconstruct the 3D Trajectory via Epipolar Analysis

- **Folder**: `01_Matlab/03_Get3DTrajectoryUsingEpipolarAnalysis`  
- **Key Script**: `C06_Obtain3DPoint_OneTrackPairWithLength_*.m`

**Purpose**  
Uses two camera views (e.g., “left” and “right” cameras) to reconstruct a 3D trajectory. The script:
1. Loads each camera’s `.mat` parameters (from step 2).
2. Reads 2D tracking data (`*_xypts.tsv`) from DLTdv8:
   - Typically includes time vs. pixel coordinates for an animal or object.
3. For each time frame, it back-projects the 2D points into 3D rays for each camera.
4. Finds the minimal distance between these two rays (the “epipolar lines”) and uses the midpoint of those closest points as the final 3D coordinate.
5. (Optionally) applies the `_Lxypts.tsv` to measure fish length in 3D at each time step.

**Usage**  
1. Edit the script to ensure it loads the correct track files in `Video_n_TrackData` (e.g., `Cropped_Loc2_0852_0857_01xypts.tsv`).
2. Point the script to your previously saved camera parameter files (e.g., `D01_camera_parameters_Cam1_*.mat`).
3. Run the script:
   - It iterates through all time frames in your 2D track files.
   - For each frame, it calculates the 3D intersection of the two rays.
   - Plots the resulting 3D trajectory in the same coordinate system as the imported OBJ environment.
4. You can visualize fish jumps, measure trajectory distances or velocities, etc.

---

## Tips and Troubleshooting

- **Coordinate Alignment**: Ensure the 3D data from the OBJ and the camera coordinate system are aligned. Verify any rotation or translation carefully.
- **Camera Calibration**: Accurate intrinsics and extrinsics are crucial. Small errors lead to larger 3D discrepancies.
- **Distortion Coefficients**: If omitted, be sure your camera has minimal lens distortion or your image tracking is undistorted prior to running the scripts.
- **Excluding Terrain**: The `polygon.xlsx` file is used to exclude certain regions of the OBJ file (e.g., water edges). If not needed, remove or disable those code blocks.

---

## Citation & References

- **DLTdv8**: [DLTdv8 on GitHub (Boston University)](http://www.unc.edu/~thedrick/dltdv/) – Tools for 2D/3D digitizing in MATLAB.  
- **MATLAB**: [MathWorks Documentation](https://www.mathworks.com/help/).
- **PolyCam**: [Gaussian Splatting](https://poly.cam).

If you use this code in research, please cite our paper.
**To be added**

---

## License

**To be added**

---

**We hope this walkthrough clarifies each script’s purpose and usage.** Feel free to open an issue or submit a pull request if you encounter any problems or have improvements (dchoi319@gatech.edu or saadb@chbe.gatech.edu).
