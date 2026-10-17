##
## EPITECH PROJECT, 2026
## OurTorch
## File description:
## Makefile
##

ANALYZER_BIN	=	my_torch_analyzer
GENERATOR_BIN	=	my_torch_generator

ANALYZER_SRC	=	analyzer/main.py
GENERATOR_SRC	=	scripts/generator.py

all: $(ANALYZER_BIN) $(GENERATOR_BIN)

$(ANALYZER_BIN): $(ANALYZER_SRC)
	cp $(ANALYZER_SRC) $(ANALYZER_BIN)
	chmod +x $(ANALYZER_BIN)

$(GENERATOR_BIN): $(GENERATOR_SRC)
	cp $(GENERATOR_SRC) $(GENERATOR_BIN)
	chmod +x $(GENERATOR_BIN)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

fclean: clean
	rm -f $(ANALYZER_BIN) $(GENERATOR_BIN)

re: fclean all

.PHONY: all clean fclean re
