# OpenCV Stereo Depth Estimation

Robot Vision Lab 02: estimate stereo disparity and relative depth using OpenCV's StereoSGBM implementation.

## Requirements

- Python 3.8 or later
- NumPy
- OpenCV Python (`opencv-python`)

## Prepare the official OpenCV stereo pair

```powershell
python download_opencv_samples.py
```

This downloads OpenCV's `aloeL.jpg` and `aloeR.jpg` sample images into `data/`.

## Run

```powershell
python stereo_depth.py
```

The program writes the left and right images, disparity maps, relative-depth visualization, and a comparison panel to `results/`.

To provide a different rectified stereo pair:

```powershell
python stereo_depth.py --left path/to/left.jpg --right path/to/right.jpg --output results_custom
```

## Notes

The output depth is relative depth because no camera calibration focal length or baseline is supplied. Larger disparity generally corresponds to objects nearer to the camera.

## Build the report with LaTeX

The report source is [`report/LAB02_Stereo_Depth_Estimation_Report.tex`](report/LAB02_Stereo_Depth_Estimation_Report.tex). Compile it from the `report` directory after installing a LaTeX distribution such as TeX Live or MiKTeX:

```powershell
cd report
xelatex LAB02_Stereo_Depth_Estimation_Report.tex
```

The source uses the result images in `../results/`.
