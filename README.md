# Gorge-agent

本项目中通过PPO算法训练模型驱动智能体，让其在对地图不断的探索中学习移动策略，合理利用闪现技能与加速增益，在限定的时间内躲避怪物追击并尽可能多的收集宝箱。   
地图中包含英雄、怪物、英雄出生点、怪物出生点、宝箱位置、道路、障碍物。  
智能体有局部视野，可以在地图中移动，释放闪现技能，走到宝箱处可获取额外分数。


项目结构  
📦 根目录  
├── 📂 agent  
│   ├── 📂 algorithm  
│       └── 📄 __init__.py  
│       └── 📄 algorithm.py  
│   ├── 📂 conf  
│       └── 📄 __init__.py  
│       └── 📄 conf.py  
│       └── 📄 train_env_conf.toml  
│   ├── 📂 feature  
│       └── 📄 __init__.py  
│       └── 📄 definition.py  
│       └── 📄 preprocessor.py  
│   ├── 📂 model  
│       └── 📄 __init__.py  
│       └── 📄 model.py  
│   ├── 📂 workflow  
│       └── 📄 __init__.py  
│       └── 📄 train_workflow.py  
│   ├── 📄 __init__.py  
│   └── 📄 agent.py  
├── 📂 conf   
│   ├── 📄 __init__.py  
│   ├── 📄 configure_app.toml  
├── 📂 log  
└── 📄 train_test.py  
