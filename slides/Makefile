# LLM Training Slides Makefile
# Quick commands to setup and run the presentation using Vite

# Default target
.DEFAULT_GOAL := help

# Variables
NODE_VERSION := 18

# Colors for output
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

## Help target
help:
	@echo "$(BLUE)🚀 LLM Training Slides$(NC)"
	@echo "$(GREEN)Available commands:$(NC)"
	@echo "  $(YELLOW)make install$(NC)   - Install dependencies"
	@echo "  $(YELLOW)make start$(NC)     - Start the presentation (dev mode)"
	@echo "  $(YELLOW)make dev$(NC)       - Same as start (Vite convention)"
	@echo "  $(YELLOW)make build$(NC)     - Build for production"
	@echo "  $(YELLOW)make preview$(NC)   - Preview production build"
	@echo "  $(YELLOW)make clean$(NC)     - Clean node_modules and build files"
	@echo "  $(YELLOW)make check$(NC)     - Check if Node.js and npm are installed"
	@echo "  $(YELLOW)make pdf$(NC)       - Generate PDF from markdown"
	@echo ""
	@echo "$(GREEN)🎯 Quick start:$(NC)"
	@echo "  $(BLUE)make install && make start$(NC)"
	@echo ""
	@echo "$(GREEN)📖 Navigation:$(NC)"
	@echo "  • Use arrow keys or spacebar to navigate slides"
	@echo "  • Presenter notes are shown below each slide"

## Check prerequisites
check:
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v node >/dev/null 2>&1 || { echo "$(RED)Node.js is required but not installed.$(NC)"; exit 1; }
	@command -v npm >/dev/null 2>&1 || { echo "$(RED)npm is required but not installed.$(NC)"; exit 1; }
	@echo "$(GREEN)✓ Node.js version: $(shell node --version)$(NC)"
	@echo "$(GREEN)✓ npm version: $(shell npm --version)$(NC)"

## Install dependencies
install: check
	@echo "$(BLUE)Installing dependencies...$(NC)"
	@npm install
	@echo "$(GREEN)✓ Dependencies installed!$(NC)"

## Start development server (Vite)
start: check
	@echo "$(BLUE)Starting LLM Training Presentation...$(NC)"
	@if [ ! -f "package.json" ]; then \
		echo "$(RED)No package.json found. Dependencies missing!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)🚀 Opening presentation at http://localhost:3000$(NC)"
	@echo "$(YELLOW)Use arrow keys or spacebar to navigate slides$(NC)"
	@echo "$(YELLOW)Presenter notes are shown below each slide$(NC)"
	@npm run dev

## Development server (alias for start)
dev: start

## Build for production
build: check
	@echo "$(BLUE)Building production version...$(NC)"
	@npm run build
	@echo "$(GREEN)✓ Build complete! Files are in ./build/$(NC)"

## Clean build files and node_modules
clean:
	@echo "$(BLUE)Cleaning up...$(NC)"
	@rm -rf node_modules build .eslintcache
	@echo "$(GREEN)✓ Cleanup complete!$(NC)"

## Build and preview production version
preview: build
	@echo "$(BLUE)Starting production preview...$(NC)"
	@npm run preview
	@echo "$(GREEN)✓ Preview server started!$(NC)"

## Generate PDF from markdown
pdf:
	@echo "$(BLUE)Generating PDF from markdown...$(NC)"
	@command -v pandoc >/dev/null 2>&1 || { echo "$(RED)pandoc is required. Install with: brew install pandoc$(NC)"; exit 1; }
	@command -v weasyprint >/dev/null 2>&1 || { echo "$(RED)weasyprint is required. Install with: brew install weasyprint$(NC)"; exit 1; }
	@pandoc llm-training-session.md -o llm-training-session.pdf --pdf-engine=weasyprint
	@echo "$(GREEN)✓ PDF generated: llm-training-session.pdf$(NC)"

# Phony targets
.PHONY: help check install start dev build clean preview pdf