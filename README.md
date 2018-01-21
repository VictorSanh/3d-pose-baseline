## 3d-pose-baseline

This is the code for the project Human Pose Estimation - Extention to videos.
This project was realized in the context of the course 'Object Recognition and Computer Vision' taught in Master MVA by I. Laptev, J. Sivic, C. Schmid and J. Ponce.


This code is based on the code developped by Martinez and al. and published along with the paper

Julieta Martinez, Rayat Hossain, Javier Romero, James J. Little.
_A simple yet effective baseline for 3d human pose estimation._
In ICCV, 2017. https://arxiv.org/pdf/1705.03098.pdf.

Initial code can be found here: https://github.com/una-dinosauria/3d-pose-baseline


### Abstract

Martinez and al. introduced a surprisingly simple linear model to perform 3D human pose estimation from 2D pose estimates that compete with the best methods available in the literature. Building on the work of Martinez, I designed a simple model that try to extend the previous work to videos by taking into account the temporal continuity between consecutive frames. The model was trained and extensively tested on the Human3.6M dataset. Results show that this new model slightly improve the performance, but more importantly, it successfully capture the temporal continuity between frames as consecutive 3D predictions are
significantly smoother. The manipulation video dataset was also used to perform quantitative comparison.


### Citing of initial authors

If you use our code, please cite our work

```
@inproceedings{martinez_2017_3dbaseline,
  title={A simple yet effective baseline for 3d human pose estimation},
  author={Martinez, Julieta and Hossain, Rayat and Romero, Javier and Little, James J.},
  booktitle={ICCV},
  year={2017}
}
```

### License
MIT
