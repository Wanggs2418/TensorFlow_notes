# TensorFlow_notes
新加坡国立大学-龙良曲课程的TF学习笔记，视频教程(B站) + TF [最新开源版电子书籍](https://github.com/dragen1860/Deep-Learning-with-TensorFlow-book)

龙老师的 github:https://github.com/dragen1860



## 虚拟环境安装

```bash
#前提确保安装源更换到国内，查看命令
conda config --show-sources

conda create -p E:\python_env\tensorflowenv38 python=3.8
activate E:\python_env\tensorflowenv38
pip install jupyter
pip install matplotlib
pip install tensorflow-cpu==2.4.0 -i  https://pypi.douban.com/simple/
pip freeze >  requirements.txt
```

