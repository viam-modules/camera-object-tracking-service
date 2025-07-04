.PHONY: setup clean pyinstaller clean-pyinstaller

MODULE_DIR=$(shell pwd)
BUILD=$(MODULE_DIR)/build

VENV_DIR=$(BUILD)/.venv
PYTHON=$(VENV_DIR)/bin/python

# PYTORCH_WHEEL=torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
# PYTORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/$(PYTORCH_WHEEL)

# TORCHVISION_REPO=https://github.com/pytorch/vision 
# TORCHVISION_WHEEL=torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
# TORCHVISION_VERSION=0.20.0

# ONNXRUNTIME_WHEEL=onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl
# ONNXRUNTIME_WHEEL_URL=https://pypi.jetson-ai-lab.dev/jp6/cu126/+f/0c4/18beb3326027d/onnxruntime_gpu-1.20.0-cp310-cp310-linux_aarch64.whl#sha256=0c418beb3326027d83acc283372ae42ebe9df12f71c3a8c2e9743a4e323443a4

REQUIREMENTS=requirements.txt

PYINSTALLER_WORKPATH=$(BUILD)/pyinstaller_build
PYINSTALLER_DISTPATH=$(BUILD)/pyinstaller_dist
	
$(VENV_DIR):
	@echo "Building python venv"
	# sudo apt install python3.10-venv
	# sudo apt install python3-pip             
	python3 -m venv $(VENV_DIR)


setup: $(VENV_DIR)
	@echo "Installing requirements"
	source $(VENV_DIR)/bin/activate &&pip install -r $(REQUIREMENTS)

pyinstaller: $(PYINSTALLER_DISTPATH)/main

$(PYINSTALLER_DISTPATH)/main: setup
	$(PYTHON) -m PyInstaller --workpath "$(PYINSTALLER_WORKPATH)" --distpath "$(PYINSTALLER_DISTPATH)" main.spec

archive.tar.gz: $(PYINSTALLER_DISTPATH)/main
	cp $(PYINSTALLER_DISTPATH)/main ./
	tar -czvf archive.tar.gz main meta.json

clean:
	rm -rf $(BUILD)
	rm -rf $(VENV_DIR)
clean-pyinstaller:
	rm -rf $(PYINSTALLER_WORKPATH)
	rm -rf $(PYINSTALLER_DISTPATH)
