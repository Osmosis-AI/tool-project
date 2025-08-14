install:
	python3 -m venv venv
	./venv/bin/pip install -r requirements.txt

run:
	./venv/bin/python main.py

run-battery:
	@if [ -z "$(BATTERY)" ]; then \
		echo "Usage: make run-battery BATTERY=battery_name [TEST=test_index]"; \
		echo "Available batteries: browsing, info_gathering, programming, visual"; \
		exit 1; \
	fi
	@mkdir -p output/$(BATTERY)
	@if [ -n "$(TEST)" ]; then \
		./venv/bin/python main.py --battery $(BATTERY) --test $(TEST) --output-dir output/$(BATTERY); \
	else \
		./venv/bin/python main.py --battery $(BATTERY) --output-dir output/$(BATTERY); \
	fi

evaluate:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make evaluate FILE=path/to/results.json"; \
		echo "Example: make evaluate FILE=output/20250805_123456.json"; \
		exit 1; \
	fi
	./venv/bin/python evaluate_results.py $(FILE)

evaluate-latest:
	@FILES=$$(find output -name "*.json" -not -path "*/evaluations/*" 2>/dev/null); \
	if [ -z "$$FILES" ]; then \
		echo "No result files found in output/ (excluding evaluations/)"; \
		exit 1; \
	fi; \
	echo "Evaluating all files in output/: $$FILES"; \
	./venv/bin/python evaluate_results.py $$FILES

convert-to-csv:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make convert-to-csv FILE=path/to/evaluated_results.json [OUTPUT=output.csv]"; \
		echo "Example: make convert-to-csv FILE=output/evaluations/evaluated_*.json OUTPUT=results.csv"; \
		exit 1; \
	fi
	@if [ -n "$(OUTPUT)" ]; then \
		./venv/bin/python convert_results_to_csv.py $(FILE) --output $(OUTPUT); \
	else \
		./venv/bin/python convert_results_to_csv.py $(FILE); \
	fi

convert-latest-to-csv:
	@LATEST=$$(ls -t output/evaluations/evaluated_*.json 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "No evaluated files found in output/evaluations/"; \
		exit 1; \
	fi; \
	echo "Converting latest evaluated file: $$LATEST"; \
	./venv/bin/python convert_results_to_csv.py $$LATEST --output evaluation_results_$$(date +%Y%m%d_%H%M%S).csv

pipeline:
	@echo "Running complete evaluation pipeline..."
	@FILES=$$(find output -name "*.json" -not -path "*/evaluations/*" 2>/dev/null); \
	if [ -z "$$FILES" ]; then \
		echo "No result files found. Run 'make run' or 'make run-battery' first."; \
		exit 1; \
	fi; \
	echo "Step 1: Evaluating all files: $$FILES"; \
	./venv/bin/python evaluate_results.py $$FILES; \
	echo "Step 2: Converting to CSV"; \
	EVAL_LATEST=$$(ls -t output/evaluations/evaluated_*.json 2>/dev/null | head -1); \
	if [ -z "$$EVAL_LATEST" ]; then \
		echo "Evaluation failed - no evaluated files found"; \
		exit 1; \
	fi; \
	OUTPUT_FILE="evaluation_results_$$(date +%Y%m%d_%H%M%S).csv"; \
	./venv/bin/python convert_results_to_csv.py $$EVAL_LATEST --output $$OUTPUT_FILE; \
	echo "Pipeline complete! CSV saved as: $$OUTPUT_FILE"

charts-dev:
	cd recharts && npm run dev

charts-build:
	cd recharts && npm run build

charts-preview:
	cd recharts && npm run preview

charts-lint:
	cd recharts && npm run lint

run-all-experiments:
	@echo "Running all experiment batteries..."
	@BATTERIES="test_pretrain_knowledge_with_no_mcp_servers test_pretrain_knowledge_with_all_mcp_servers test_tool_usage_with_only_info_gathering_mcp_servers test_tool_usage_with_only_visual_mcp_servers test_tool_usage_with_only_programming_mcp_servers test_tool_usage_browsing_with_all_mcp_servers test_tool_usage_info_gathering_with_all_mcp_servers test_tool_usage_programming_with_all_mcp_servers test_tool_usage_visual_with_all_mcp_servers test_tool_usage_with_only_browsing_mcp_servers test_tool_usage_with_only_info_gathering_mcp_servers test_tool_usage_with_only_visual_mcp_servers test_tool_usage_with_only_programming_mcp_servers"; \
	for battery in $$BATTERIES; do \
		echo "Running battery: $$battery"; \
		mkdir -p output/$$battery; \
		./venv/bin/python main.py --battery $$battery --output-dir output/$$battery || echo "Warning: Battery $$battery failed"; \
	done; \
	echo "All experiments completed!"

evaluate-all-and-csv:
	@echo "Running complete evaluation and CSV generation pipeline..."
	@FILES=$$(find output -name "*.json" -not -path "*/evaluations/*" 2>/dev/null); \
	if [ -z "$$FILES" ]; then \
		echo "No result files found. Run 'make run-all-experiments' first."; \
		exit 1; \
	fi; \
	echo "Step 1: Evaluating all files: $$FILES"; \
	./venv/bin/python evaluate_results.py $$FILES; \
	echo "Step 2: Converting all evaluated files to CSV"; \
	EVAL_FILES=$$(find output/evaluations -name "evaluated_*.json" 2>/dev/null); \
	if [ -z "$$EVAL_FILES" ]; then \
		echo "Evaluation failed - no evaluated files found"; \
		exit 1; \
	fi; \
	for eval_file in $$EVAL_FILES; do \
		OUTPUT_FILE="csv_results_$$(basename $$eval_file .json)_$$(date +%Y%m%d_%H%M%S).csv"; \
		./venv/bin/python convert_results_to_csv.py $$eval_file --output $$OUTPUT_FILE; \
		echo "CSV generated: $$OUTPUT_FILE"; \
	done; \
	echo "Pipeline complete! All CSV files generated."

full-experiment-pipeline:
	@echo "Running full experiment pipeline..."
	make run-all-experiments
	make evaluate-all-and-csv
	@echo "Full experiment pipeline completed!"

how to:
	@echo "Available commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make run              - Run all experiments"
	@echo "  make run-battery BATTERY=name [TEST=n] - Run specific battery"
	@echo "  make run-all-experiments - Run all experiment batteries sequentially"
	@echo "  make evaluate-all-and-csv - Evaluate all results and generate CSV files"
	@echo "  make full-experiment-pipeline - Run all experiments, evaluate, and generate CSVs"
	@echo ""
	@echo "Evaluation commands:"
	@echo "  make evaluate FILE=path/to/results.json - Evaluate specific results file"
	@echo "  make evaluate-latest  - Evaluate the most recent results file"
	@echo "  make convert-to-csv FILE=path/to/evaluated.json [OUTPUT=file.csv] - Convert to CSV"
	@echo "  make convert-latest-to-csv - Convert latest evaluated file to CSV"
	@echo "  make pipeline         - Run complete evaluation pipeline (evaluate latest + convert to CSV)"
	@echo ""
	@echo "Chart visualization commands:"
	@echo "  make charts-dev       - Start development server for charts"
	@echo "  make charts-build     - Build charts for production"
	@echo "  make charts-preview   - Preview built charts"
	@echo "  make charts-lint      - Lint chart code"
	@echo ""
	@echo "Example workflows:"
	@echo "  make run-battery BATTERY=test_pretrain_knowledge_with_no_mcp_servers"
	@echo "  make pipeline         # Evaluates latest results and creates CSV"
	@echo "  make evaluate-latest  # Just evaluate without CSV conversion"
	@echo "  make convert-latest-to-csv # Just convert to CSV"
