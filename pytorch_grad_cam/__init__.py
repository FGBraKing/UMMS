from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.ablation_cam import AblationCAM
from pytorch_grad_cam.xgrad_cam import XGradCAM
from pytorch_grad_cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.score_cam import ScoreCAM
from pytorch_grad_cam.guided_backprop import GuidedBackpropReLUModel
'''
copy from https://github.com/jacobgil/pytorch-grad-cam
# References

https://arxiv.org/abs/1610.02391
`Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization
Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, Dhruv Batra`

https://arxiv.org/abs/1710.11063
`Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks
Aditya Chattopadhyay, Anirban Sarkar, Prantik Howlader, Vineeth N Balasubramanian`

https://arxiv.org/abs/1910.01279
`Score-CAM: Score-Weighted Visual Explanations for Convolutional Neural Networks
Haofan Wang, Zifan Wang, Mengnan Du, Fan Yang, Zijian Zhang, Sirui Ding, Piotr Mardziel, Xia Hu`

https://ieeexplore.ieee.org/abstract/document/9093360/
`Saurabh Desai and Harish G Ramaswamy. Ablation-cam: Visual explanations for deep
convolutional network via gradient-free localization. In WACV, pages 972–980, 2020`

https://arxiv.org/abs/2008.02312
`Axiom-based Grad-CAM: Towards Accurate Visualization and Explanation of CNNs
Ruigang Fu, Qingyong Hu, Xiaohu Dong, Yulan Guo, Yinghui Gao, Biao Li`

'''