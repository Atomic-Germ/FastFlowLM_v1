BUILD_DIR := build
OUT_DIR   := out
PWSH      := powershell.exe


.PHONY: all clean run serve install model-build model-install model-test model-list model-venv

ifeq ($(OS),Windows_NT)

all:
	@$(PWSH) "if (!(Test-Path '$(BUILD_DIR)')) { New-Item -ItemType Directory -Path '$(BUILD_DIR)' }"
	@$(PWSH) "if (!(Test-Path '$(OUT_DIR)'))   { New-Item -ItemType Directory -Path '$(OUT_DIR)' }"
	@$(PWSH) "cmake --preset windows-vs18"
	@$(PWSH) "cmake --build build --preset windows-default"

run:
	@$(PWSH) "Copy-Item 'lib\\xrt\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'lib\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_list.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_info.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'xclbins' '$(OUT_DIR)' -Recurse -Force"
	@$(PWSH) "cd '$(OUT_DIR)'; .\\flm.exe run llama3.2:1b -a 0"

serve:
	@$(PWSH) "Copy-Item 'lib\\*.dll' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'model_list.json' '$(OUT_DIR)' -Force"
	@$(PWSH) "Copy-Item 'xclbins' '$(OUT_DIR)' -Recurse -Force"
	@$(PWSH) "cd '$(OUT_DIR)'; .\\flm.exe serve qwen2.5vl-it -c 32768" 

clean:
	@$(PWSH) "Remove-Item -Recurse -Force '$(BUILD_DIR)'"
	@$(PWSH) "Remove-Item -Recurse -Force '$(OUT_DIR)'"

install:
	@echo "Installing using CMake..."
	@cd $(BUILD_DIR) && cmake --install . --prefix $(PREFIX)

else

all:
	@mkdir -p $(OUT_DIR)
	@cmake --preset linux-default
	@cmake --build build
	@cp build/flm $(OUT_DIR)

run:
	@cp lib/*.so $(OUT_DIR)
	@cp model_list.json $(OUT_DIR)
	@cp model_info.json $(OUT_DIR)
	@cd $(OUT_DIR) && ./flm run gpt-oss -a 0

serve:
	@cp lib/*.so $(OUT_DIR)
	@cp model_list.json $(OUT_DIR)
	@cd $(OUT_DIR) && ./flm serve -a 1 -e 1

clean:
	@rm -rf $(BUILD_DIR)
	@rm -rf $(OUT_DIR)

install:
	@echo "Installing using CMake..."
	@cd $(BUILD_DIR) && sudo cmake --install . 

endif

# ---------------------------------------------------------------------------
# Model management targets — shared across platforms
# ---------------------------------------------------------------------------

MODEL_PY ?= python3
FLM_MODEL_ARGS ?=

model-venv:
	@if command -v uv >/dev/null 2>&1; then \
		echo "[model-venv] using uv"; \
		uv venv .venv; \
		uv pip install --python .venv/bin/python \
			../utilities/flm-add \
			../utilities/flm-test \
			../utilities/q4nx-build; \
	else \
		echo "[model-venv] uv not found, using pip"; \
		$(MODEL_PY) -m venv .venv; \
		.venv/bin/pip install \
			../utilities/flm-add \
			../utilities/flm-test \
			../utilities/q4nx-build; \
	fi

model-build:
	$(MODEL_PY) flm_model.py build $(FLM_MODEL_ARGS)

model-install:
	$(MODEL_PY) flm_model.py install $(FLM_MODEL_ARGS)

model-test:
	$(MODEL_PY) flm_model.py test $(FLM_MODEL_ARGS)

model-list:
	$(MODEL_PY) flm_model.py list
