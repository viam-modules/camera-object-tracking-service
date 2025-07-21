SHELL := /bin/bash
.PHONY: setup clean pyinstaller clean-pyinstaller

MODULE_DIR=$(shell pwd)
BUILD=$(MODULE_DIR)/build

VENV_DIR=$(BUILD)/.venv
PYTHON=$(VENV_DIR)/bin/python

PYTORCH_WHEEL=torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
PYTORCH_WHEEL_URL=https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/$(PYTORCH_WHEEL)

TORCHVISION_REPO=https://github.com/pytorch/vision 
TORCHVISION_WHEEL=torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
TORCHVISION_VERSION=0.20.0

JP6_REQUIREMENTS=requirements_jp6.txt
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

$(BUILD)/$(PYTORCH_WHEEL):
	@echo "Making $(BUILD)/$(PYTORCH_WHEEL)"
	wget  -P $(BUILD) $(PYTORCH_WHEEL_URL)

pytorch-wheel: $(BUILD)/$(PYTORCH_WHEEL)

$(BUILD)/$(TORCHVISION_WHEEL): $(VENV_DIR) $(BUILD)/$(PYTORCH_WHEEL)
	@echo "Installing dependencies for TorchVision"
	bin/first_run.sh
	bin/install_cusparselt.sh

	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install wheel
	$(PYTHON) -m pip install 'numpy<2' $(BUILD)/$(PYTORCH_WHEEL)

	@echo "Cloning Torchvision"
	git clone --branch v${TORCHVISION_VERSION} --recursive --depth=1 $(TORCHVISION_REPO) $(BUILD)/torchvision

	@echo "Building torchvision wheel"
	cd $(BUILD)/torchvision && $(PYTHON) setup.py --verbose bdist_wheel --dist-dir ../

torchvision-wheel: $(BUILD)/$(TORCHVISION_WHEEL)

setup-jp6: torchvision-wheel
	@echo "Installing requirements for JP6"
	source $(VENV_DIR)/bin/activate &&pip install -r $(JP6_REQUIREMENTS)


pyinstaller: $(PYINSTALLER_DISTPATH)/main

$(PYINSTALLER_DISTPATH)/main: setup
	$(PYTHON) -m PyInstaller --workpath "$(PYINSTALLER_WORKPATH)" --distpath "$(PYINSTALLER_DISTPATH)" main.spec 

module.tar.gz: $(PYINSTALLER_DISTPATH)/main
	cp $(PYINSTALLER_DISTPATH)/main ./
	tar -czvf module.tar.gz main meta.json

clean:
	rm -rf $(BUILD)
	rm -rf $(VENV_DIR)
clean-pyinstaller:
	rm -rf $(PYINSTALLER_WORKPATH)
	rm -rf $(PYINSTALLER_DISTPATH)
