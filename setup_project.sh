#!/bin/bash
# MMCows Project Structure Setup
# Run this once from your desired project root

mkdir -p {data/{raw,processed,splits},notebooks,src/{detection,tracking,behaviour,reid,utils},configs,outputs/{checkpoints,logs,results,visualisations},docs}

# Create placeholder README files in each module
echo "# Detection Module\nYOLOv8-L fine-tuning on MMCows visual_data" > src/detection/README.md
echo "# Tracking Module\nAdaGrad triangulation + DeepSORT in 3D world coordinates" >  src/tracking/README.md
echo "# Behaviour Classification\nBranch A: SlowFast | Branch B: Shared ResNet-50 multi-view" > src/behaviour/README.md
echo "# Re-ID Module\nContrastive pre-training (SimCLR/MoCo) for cow identity" > src/reid/README.md
echo "# Utilities\nData loading, preprocessing, metrics helpers" > src/utils/README.md

# Create a root .gitignore
cat > .gitignore << 'EOF'
# Data (too large for git)
data/raw/
data/processed/
outputs/checkpoints/
*.zip
*.pt
*.pth

# Python
__pycache__/
*.pyc
.env
*.egg-info/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
EOF

echo ""
echo "✅ Project structure created:"
find mmcows -type d | sort
